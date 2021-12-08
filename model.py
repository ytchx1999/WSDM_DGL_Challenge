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
from dgl.nn.pytorch import SAGEConv, GATConv


class HeteroConv(nn.Module):
    def __init__(self, etypes, n_layers, in_feats, hid_feats, emb_feats, activation, dropout=0.2, args=None, edge_dim=None, num_heads=1, time_dim=1):
        super(HeteroConv, self).__init__()
        self.etypes = etypes
        self.n_layers = n_layers
        self.in_feats = in_feats
        self.hid_feats = hid_feats
        self.emb_feats = emb_feats
        self.act = activation
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.args = args
        self.num_heads = num_heads
        self.time_dim = time_dim
        self.hconv_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(n_layers):
            self.norms.append(nn.BatchNorm1d(hid_feats * num_heads))
        self.norms.append(nn.BatchNorm1d(emb_feats * num_heads))

        # A: feature and edge encoder
        if self.args.dataset == 'A':
            self.node_encoders = nn.ModuleList()
            for i in range(in_feats):
                self.node_encoders.append(nn.Embedding(420, embedding_dim=self.args.node_enc_dim, padding_idx=417))
            in_feats = self.args.node_enc_dim  # embedding dim

            # self.edge_encoders = nn.ModuleList()
            # for i in range(edge_dim):
            #     self.edge_encoders.append(nn.Embedding(250, embedding_dim=10))
            # edge_feats = 10
            edge_feats = 0
        else:
            edge_feats = 0

        self.time_encoder = nn.ModuleList()
        bits = 10
        for i in range(bits):
            self.time_encoder.append(nn.Embedding(20, embedding_dim=self.time_dim))  # time digit embedding dim

        # input layer
        self.hconv_layers.append(self.build_hconv(in_feats, hid_feats, activation=self.act, num_heads=self.num_heads))
        # hidden layers
        for i in range(n_layers - 1):
            self.hconv_layers.append(self.build_hconv(hid_feats * num_heads, hid_feats, activation=self.act, num_heads=self.num_heads))
        # output layer
        self.hconv_layers.append(self.build_hconv(hid_feats * num_heads, emb_feats, num_heads=self.num_heads))  # activation None

        # self.fc1 = nn.Linear(hid_feats*2+10+edge_feats, hid_feats)
        # self.fc2 = nn.Linear(hid_feats, 1)
        if self.args.dataset == 'A':
            time_dim = (bits * self.time_dim)
        else:
            time_dim = 45
        self.mlp = MLP(emb_feats * 2 * num_heads + time_dim + edge_feats, emb_feats, 1, num_layers=3)

    def build_hconv(self, in_feats, out_feats, activation=None, num_heads=1):
        GNN_dict = {}
        for event_type in self.etypes:
            GNN_dict[event_type] = GATConv(in_feats=in_feats, out_feats=out_feats, num_heads=num_heads, residual=True, activation=activation)
            # GNN_dict[event_type] = SAGEConv(in_feats=in_feats, out_feats=out_feats, aggregator_type='mean', activation=activation)
        return dglnn.HeteroGraphConv(GNN_dict, aggregate='sum')

    def forward(self, blocks, feat_key='feat'):
        h = blocks[0].ndata[feat_key]

        # A: feature encoding
        if self.args.dataset == 'A':
            h = h['Node']
            collect = []
            for i in range(h.shape[1]):
                collect.append(self.node_encoders[i](h[:, i]))
            h = torch.stack(collect, dim=0).sum(dim=0)

        if not isinstance(h, dict):
            h = {'Node': h}
        for i, layer in enumerate(self.hconv_layers):
            h = layer(blocks[i], h)
            for key in h.keys():
                h[key] = self.norms[i](h[key].flatten(1, -1))
                # h[key] = self.act(h[key])
                # h[key] = self.dropout(h[key])
        return h

    def emb_concat(self, g, etype, args):
        if self.args.dataset == 'A':
            def cat(edges):
                # efeat = edges.data['feat']
                # collect = []
                # for i, encoder in enumerate(self.edge_encoders):
                #     collect.append(encoder(efeat[:, i]))
                # efeat = torch.stack(collect, dim=0).sum(dim=0)
                # return {'emb_cat': torch.cat([edges.src['emb'], efeat, edges.dst['emb']], 1)}
                return {'emb_cat': torch.cat([edges.src['emb'], edges.dst['emb']], 1)}
        else:
            def cat(edges):
                return {'emb_cat': torch.cat([edges.src['emb'], edges.dst['emb']], 1)}
        with g.local_scope():
            g.apply_edges(cat, etype=etype)
            emb_cat = g.edges[etype].data['emb_cat']
        return emb_cat

    def time_encoding(self, x, bits=10):
        '''
        This function is designed to encode a unix timestamp to a 10-dim vector.
        And it is only one of the many options to encode timestamps.
        Users can also define other time encoding methods such as Neural Network based ones.

        时间戳的第一位都是1，没用，所以舍弃
        '''
        inp = x.repeat(10, 1).transpose(0, 1)
        div = torch.cat([torch.ones((x.shape[0], 1), dtype=torch.long) * 10 ** (bits-1-i) for i in range(bits)], 1).to(inp.device)
        if self.args.dataset == 'A':
            h = (inp // div) % 10
            # h = h[:, 1:]
            collect = []
            for i, encoder in enumerate(self.time_encoder):
                collect.append(encoder(h[:, i]))
            h = torch.cat(collect, dim=1)
        else:
            h = (((inp // div) % 10) * 0.1).float()
            h = h[:, 1:]
            collect = []
            for i in range(h.shape[1]):
                collect.append(h[:, i].repeat(h.shape[1] - i, 1).transpose(0, 1))
            h = torch.cat(collect, dim=1)
            # h = F.normalize(h, p=2, dim=1)
        return h

    def time_predict(self, node_emb_cat, time_emb, args):
        h = torch.cat([node_emb_cat, time_emb], 1)
        h = self.dropout(h)
        # h = self.fc1(h)
        # h = self.act(h)
        # h = self.fc2(h)
        h = self.mlp(h, args)
        return h


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()

        self.lins.append(nn.Linear(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        for i in range(self.num_layers - 2):
            self.lins.append(nn.Linear(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.lins.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x, args):
        for i in range(self.num_layers - 1):
            x = F.relu(self.lins[i](x))
            if args.dataset == 'B':
                x = self.bns[i](x)  # batch norm
                x = F.dropout(x, p=0.2, training=self.training)
        x = self.lins[-1](x)
        return x
