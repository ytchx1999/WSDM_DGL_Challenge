#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Xuhong Wang
# @Email  : wxuhong@amazon.com or wang_xuhong@sjtu.edu.cn
# Feel free to send me an email if you have any question.
# You can also CC Quan Gan (quagan@amazon.com).
from genericpath import exists
import torch.nn.functional as F
import torch
import dgl
import dgl.nn.pytorch as dglnn
from torch import nn
import argparse
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import os
import dgl.function as fn
import random
from model import HeteroConv
from dgl.dataloading import EdgeDataLoader, NodeDataLoader, MultiLayerFullNeighborSampler, MultiLayerNeighborSampler
from tqdm import tqdm

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)


def preprocess(args, directed_g):
    # this function is used to add reverse edges for model computing
    if args.dataset == 'A':
        g = dgl.add_reverse_edges(directed_g, copy_edata=True)
    if args.dataset == 'B':
        graph_dict = {}
        for (src_type, event_type, dst_type) in directed_g.canonical_etypes:
            graph_dict[(src_type, event_type, dst_type)] = directed_g.edges(etype=(src_type, event_type, dst_type))
            src_nodes_reversed = directed_g.edges(etype=(src_type, event_type, dst_type))[1]
            dst_nodes_reversed = directed_g.edges(etype=(src_type, event_type, dst_type))[0]
            graph_dict[(dst_type, event_type+'_reversed', src_type)] = (src_nodes_reversed, dst_nodes_reversed)
        g = dgl.heterograph(graph_dict)
        for etype in g.etypes:
            g.edges[etype].data['ts'] = directed_g.edges[etype.split('_')[0]].data['ts']
            if 'feat' in directed_g.edges[etype.split('_')[0]].data.keys():
                g.edges[etype].data['feat'] = directed_g.edges[etype.split('_')[0]].data['feat']
    return g


def get_args():
    # Argument and global variables
    parser = argparse.ArgumentParser('Base')
    parser.add_argument('--dataset', type=str, choices=["A", "B"], default='A', help='Dataset name')
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--node_enc_dim', type=int, default=128, help='embedding dim of node feature in A')
    parser.add_argument("--hid_dim", type=int, default=32, help="number of hidden gnn units")
    parser.add_argument("--emb_dim", type=int, default=10, help="number of final gnn embedding units")
    parser.add_argument("--time_dim", type=int, default=3, help="number of time encoding dims")
    parser.add_argument("--n_layers", type=int, default=2, help="number of hidden gnn layers")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight for L2 loss")
    parser.add_argument("--gpu", type=int, default=1, help="number of GPU")
    parser.add_argument("--seed", type=int, default=0, help="number of seed")
    parser.add_argument("--batch_size", type=int, default=20000, help="batch size")
    parser.add_argument("--num_heads", type=int, default=1, help="number of heads")

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args


def main():
    args = get_args()
    print(args, flush=True)
    set_seed(args.seed)

    if not os.path.exists('./outputs'):
        os.makedirs('./outputs')
    if args.dataset == 'B':
        g = dgl.load_graphs(f'DGLgraphs/Dataset_{args.dataset}.bin')[0][0]
    elif args.dataset == 'A':
        g, etype_feat = dgl.load_graphs(f'DGLgraphs/Dataset_{args.dataset}.bin')
        g = g[0]

    g = preprocess(args, g)
    g, model = train(args, g)
    test(args, g, model)

    print("Saving results...", flush=True)
    test_and_save(args, g, model)
    print("Done!", flush=True)


def train(args, g):
    if args.dataset == 'B':
        # dim_nfeat = args.emb_dim*2
        # for ntype in g.ntypes:
        #     g.nodes[ntype].data['feat'] = torch.randn((g.number_of_nodes(ntype), dim_nfeat)) * 0.05
        funcs = {}
        for c_etype in g.canonical_etypes:
            srctype, etype, dsttype = c_etype
            funcs[etype] = (fn.copy_e('feat', 'feat_copy'), fn.mean('feat_copy', 'feat'))
        # 将每个类型消息聚合的结果相加
        g.multi_update_all(funcs, 'sum')
        dim_nfeat = g.ndata['feat'][g.ntypes[0]].shape[1]
        args.time_dim = 1

        for ntype in g.ntypes:
            # ndata_mean = g.nodes[ntype].data['feat'].mean(dim=0)
            # feat = g.nodes[ntype].data['feat']
            # feat_dim = feat.shape[1]
            # for i in range(feat.shape[0]):
            #     if torch.equal(feat[i], torch.zeros((feat_dim,))):
            #         feat[i] = ndata_mean
            # g.nodes[ntype].data['feat'] = feat
            g.nodes[ntype].data['feat'] += torch.randn((g.number_of_nodes(ntype), dim_nfeat)) * 0.01
    else:
        dim_nfeat = g.ndata['feat'].shape[1]

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    # g = g.to(device)

    sampler = MultiLayerFullNeighborSampler(args.n_layers + 1)
    train_nid_dict = {ntype: g.nodes(ntype) for ntype in g.ntypes}
    train_loader = NodeDataLoader(
        g, train_nid_dict, sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=4
    )

    model = HeteroConv(g.etypes, args.n_layers, dim_nfeat, args.hid_dim, args.emb_dim, F.relu, dropout=0.2,
                       args=args, edge_dim=g.edata['feat'][g.canonical_etypes[0]].shape[1], num_heads=args.num_heads, time_dim=args.time_dim)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)
    loss_fcn = nn.BCEWithLogitsLoss().to(device)
    loss_values = []

    best_test_auc = 0

    for epoch in range(args.epochs):
        model.train()

        ndata_emb = {}
        for ntype in g.ntypes:
            ndata_emb[ntype] = torch.zeros((g.nodes[ntype].data['feat'].shape[0], args.emb_dim * args.num_heads))

        for input_nodes, output_nodes, blocks in tqdm(train_loader, desc=f'Epoch {epoch}'):
            blocks = [block.to(device) for block in blocks]
            node_emb = model(blocks)
            loss = 0

            sub_g = dgl.node_subgraph(g, {ntype: blocks[-1].dstnodes(ntype).cpu() for ntype in blocks[-1].dsttypes}).to(device)
            for ntype in sub_g.ntypes:
                sub_g.nodes[ntype].data['emb'] = node_emb[ntype]
                if not isinstance(output_nodes, dict):
                    ndata_emb[ntype][output_nodes] = node_emb[ntype].detach().cpu()
                else:
                    ndata_emb[ntype][output_nodes[ntype]] = node_emb[ntype].detach().cpu()

            total = 0
            for i, etype in enumerate(sub_g.etypes):
                if etype.split('_')[-1] == 'reversed':
                    # Etype that end with 'reversed' is the reverse edges we added for GNN message passing.
                    # So we do not need to compute loss in training.
                    continue
                if sub_g.edges[etype].data['feat'].shape[0] == 0:
                    continue
                total += 1
                emb_cat = model.emb_concat(sub_g, etype, args)
                ts = sub_g.edges[etype].data['ts']
                idx = torch.randperm(ts.shape[0]).to(device)
                ts_shuffle = ts[idx]
                neg_label = torch.zeros_like(ts).to(device)
                neg_label[ts_shuffle >= ts] = 1

                time_emb = model.time_encoding(ts)
                time_emb_shuffle = model.time_encoding(ts_shuffle)

                pos_exist_prob = model.time_predict(emb_cat, time_emb, args).squeeze()
                neg_exist_prob = model.time_predict(emb_cat, time_emb_shuffle, args).squeeze()

                probs = torch.cat([pos_exist_prob, neg_exist_prob], 0)
                label = torch.cat([torch.ones_like(ts), neg_label], 0).float()
                loss += loss_fcn(probs, label)  # /len(sub_g.etypes)

            loss /= float(total)
            loss_values.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # print('Loss:', loss_values[-1], flush=True)
            torch.cuda.empty_cache()

        for ntype in g.ntypes:
            g.nodes[ntype].data['emb'] = ndata_emb[ntype]
        # test every epoch
        test_auc = test(args, g, model.cpu())
        print(f'Epoch: {epoch:02d}, Loss: {loss_values[-1]:.4f}, test AUC: {round(test_auc, 5)}', flush=True)
        if test_auc > best_test_auc:
            best_test_auc = test_auc
            # save ckpt
            torch.save(model.state_dict(), f'./outputs/best_auc_{args.dataset}.pkl')
        model.to(device)

    print(f'Best test AUC: {round(best_test_auc, 5)}', flush=True)
    # load best model
    model.cpu()
    model.load_state_dict(torch.load(f'./outputs/best_auc_{args.dataset}.pkl'))

    return g, model


@torch.no_grad()
def test(args, g, model):
    model.eval()
    test_csv = pd.read_csv(f'test_csvs/input_{args.dataset}_initial.csv', names=['src', 'dst', 'type', 'start_at', 'end_at', 'exist'])
    label = test_csv.exist.values
    etype = test_csv.type.values
    src = test_csv.src.values
    dst = test_csv.dst.values
    start_at = torch.tensor(test_csv.start_at.values)
    end_at = torch.tensor(test_csv.end_at.values)

    if args.dataset == 'A':
        # efeat = g.edges[str(etype)].data['feat'][src]
        # collect = []
        # for i, encoder in enumerate(model.edge_encoders):
        #     collect.append(encoder(efeat[:, i]))
        # efeat = torch.stack(collect, dim=0).sum(dim=0)
        emb_cats = torch.cat([g.ndata['emb'][src], g.ndata['emb'][dst]], 1)
    if args.dataset == 'B':
        emb_cats = torch.cat([g.ndata['emb']['User'][src], g.ndata['emb']['Item'][dst]], 1)

    start_time_emb = model.time_encoding(start_at)
    end_time_emb = model.time_encoding(end_at)
    start_prob = model.time_predict(emb_cats, start_time_emb, args).squeeze()
    end_prob = model.time_predict(emb_cats, end_time_emb, args).squeeze()
    exist_prob = end_prob - start_prob

    AUC = roc_auc_score(label, exist_prob)
    # print(f'AUC is {round(AUC,5)}', flush=True)

    return AUC


@torch.no_grad()
def test_and_save(args, g, model):
    model.eval()
    test_csv = pd.read_csv(f'test_csvs/input_{args.dataset}.csv', names=['src', 'dst', 'type', 'start_at', 'end_at'])
    # label = test_csv.exist.values
    etype = test_csv.type.values
    src = test_csv.src.values
    dst = test_csv.dst.values
    start_at = torch.tensor(test_csv.start_at.values)
    end_at = torch.tensor(test_csv.end_at.values)
    if args.dataset == 'A':
        emb_cats = torch.cat([g.ndata['emb'][src], g.ndata['emb'][dst]], 1)
    if args.dataset == 'B':
        emb_cats = torch.cat([g.ndata['emb']['User'][src], g.ndata['emb']['Item'][dst]], 1)

    start_time_emb = model.time_encoding(start_at)
    end_time_emb = model.time_encoding(end_at)
    start_prob = model.time_predict(emb_cats, start_time_emb, args).squeeze()
    end_prob = model.time_predict(emb_cats, end_time_emb, args).squeeze()
    exist_prob = end_prob - start_prob

    exist_prob = exist_prob.reshape(-1, 1).numpy()
    np.savetxt(f'outputs/output_{args.dataset}.csv', exist_prob, delimiter=None)

    # AUC = roc_auc_score(label, exist_prob)
    # print(f'AUC is {round(AUC,5)}', flush=True)


if __name__ == "__main__":
    main()
