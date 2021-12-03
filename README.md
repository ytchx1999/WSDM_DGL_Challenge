# WSDM_DGL_Challenge
WSDM Cup 2022: Temporal Link Prediction Task (https://www.dgl.ai/WSDM2022-Challenge/).

# WSDM 2022 Large-scale Temporal Graph Link Prediction - Baseline and Initial Test Set

[WSDM Cup Website link](https://www.wsdm-conference.org/2022/call-for-wsdm-cup-proposals/)

[Link to this challenge](https://www.dgl.ai/WSDM2022-Challenge/)

This branch offers

* An initial test set having 10,000 test examples for each dataset, together with their labels in `exist` column.  Note that this test set only serves for development purposes.  So
  * The intermediate and final dataset will **not** contain the `exist` column.
  * This is **not** the intermediate dataset we will be using for ranking solutions.
* A simple baseline that trains on both datasets.

Download links to initial test set: [Dataset A](https://data.dgl.ai/dataset/WSDMCup2022/input_A_initial.csv.gz) [Dataset B](https://data.dgl.ai/dataset/WSDMCup2022/input_B_initial.csv.gz)

## Baseline description

| Date | Method | **Best** initial test AUC of **A** | **Best** initial test AUC of **B** |
|:-:|:-:|:-:|:-:|
| 2021.12.03 | minibatch  | 0.6113 | 0.58478 |
| 2021.11.31 | new time encoding  | 0.57364 | 0.57479 |
| 2021.11.29 | 数据预处理 | 0.52814 | 0.53116 |
| ----       | raw baseline | 0.511 | 0.510 |



#### 数据预处理
Time encoding：
+ A: 时间戳为10位十进制数，抽出每一位分别进行`nn.Embedding`的映射，然后从左到右进行concat得到`time_emb`。
+ B: 时间戳为10位十进制数，从左到右优先级依次降低，每列重复10-i次，然后从左到右进行concat得到`time_emb`。
+ 均舍弃了原始时间戳的第一位`1`（2021年才是16开头的数）。

A:
+ node_feat所有缺失值用max+1（417）来填充，包括csv的整行确实
+ train: ndata['feat']8列分别进行encoding，得到可学习的embedding后，stack+sum
+ g.edata['feat']进行手动广播，和边进行一一对应
+ ~~def emb_conccat() --> cat(src['emb'], edge_feat_emb, dst['emb])~~

B: 
+ g.edata['feat']缺失值进行0填充
+ g.ndata['feat']聚合异构图中邻边的edata['feat']


#### 异构图的构造

A:
+ edge: `{('Node', 'e_type', 'Node'): (src, dst)}`
+ edata['ts']: `{('Node', 'e_type', 'Node'): (time)}`
+ ndata['feat']: `{'Node': 部分节点有特征，部分节点没有特征为全0}`
+ etype_feat: 目前没用到，边类型的特征

B: 
+ edge: `{('User', 'e_type', 'Item'): (src, dst)}`
+ edata['ts']: `{('User', 'e_type', 'Item'): (time)}` 
+ edata['feat']: `{('User', 'e_type', 'Item'): 单纯的边特征}`
+ etype_feat: None

```bash
# etype = ('User', '1', 'Item')
>>> g.edata['ts'][('User', '1', 'Item')].shape
torch.Size([29457])
>>> g.edata['feat'][('User', '1', 'Item')].shape
torch.Size([29457, 768])
>>> g.edges[('User', '1', 'Item')].data['feat'].shape
torch.Size([29457, 768])
```

#### 异构图GNN

每种类型的边分别定义GNN算子：
```bash
{
   ('User', '1', 'Item'): dgl.nn.SAGEConv(...),
   ...
   ('User', '1_reversed', 'Item'): dgl.nn.SAGEConv(...),
}
```


#### ~~时间编码 (time encoding)： ~~
时间戳为10位十进制数，抽出每一位乘0.1组成一个10维向量。

例如，时间戳为`1420079360`, encoding后变成10维向量为 `[0.1, 0.4, 0.2, 0.0, 0.0, 0.7, 0.9, 0.3, 0.6, 0.0]`

#### 负采样时间戳（random index）--> `t'`
+ `t <= t', label = 1`
+ `t >  t', label = 0`

求解 `P(t <= t' | s, d, r)`，表示在时间`t'`**之前**，从源节点s到目标节点d之前存在r类型的边的概率。最终inference：`t_start ~ t_end` 之间，从源节点s到目标节点d之前存在r类型的边的概率。

**`P(t_start <= t <= t_end | s, d, r) = P(t <= t_end | s, d, r) - P(t <= t_start | s, d, r)`**

#### 训练流程
1. 根据 heterogeneous graph 结构和 ndata['feat']，通过 HGNN 训练得到节点的 `node_emb`
2. 对于每条边，`edge_emb = cat([src_node_emb, dst_node_emb])` 
3. 正/负采样，得到正样本和负样本的 `timestamp` 和 `label`
4. time encoding 得到 `time_emb`
5. 对于每条边，`cat([edge_emb, time_emb])` 之后，过Linear层得到出现的概率 `probs`
6. BCEWithLogitsLoss() + backward()更新参数

#### 一些细节
baseline中，A没有使用 etype_feat，B没有使用 edata['feat']

---

The baseline is only a minimal working example for both datasets, and it is certainly not optimal.  **You are encouraged to tweak it or propose your own solutions from scratch!**

Here we summarize our baseline:
The baseline is an [RGCN](https://arxiv.org/abs/1703.06103)-like GNN model trained on the entire graph.
Event timestamps on the graph are encoded by decomposing the 10-digit decimal integers into 10-dimensional vectors, each element representing a digit.
We train the model as binary classification using a negative-sampling-like strategy.
Given a ground truth event `(s, d, r, t)` with source node `s`, destination node `d`, event type `r` and timestamp `t`, we perturb `t` to obtain a new value `t'`.
We label the quadruplet with 1 if the new timestamp is larger than the original timestamp, and 0 otherwise.  The model is essentially trained to
predict `p(t < t' | s, d, r)`, i.e. the probability that an edge with type `r` exists from source `s` and destination `d` before timestamp `t'`.

## Baseline usage

不需要手动下载数据集，直接运行程序即可。

To use the baseline you need to install [DGL](https://www.dgl.ai).

~~ou also need at least 64GB of CPU memory.  GPU is not required.~~

**mini-batch train in GPU, full batch inference in CPU!**

1. Convert csv file to DGL graph objects.

   ```bash
   python3 csv2DGLgraph.py --dataset [A or B]
   ```

2. Training.

   ```bash
   cd scripts/
   nohup bash trainA.sh > ../outputs/a.log 2>&1 &
   nohup bash trainB.sh > ../outputs/b.log 2>&1 &
   ```

#### Args
```bash
usage: Base [-h] [--dataset {A,B}] [--lr LR] [--epochs EPOCHS]
            [--node_enc_dim NODE_ENC_DIM] [--emb_dim EMB_DIM]
            [--time_dim TIME_DIM] [--n_layers N_LAYERS]
            [--weight_decay WEIGHT_DECAY] [--gpu GPU]
            [--batch_size BATCH_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --dataset {A,B}       Dataset name
  --lr LR               learning rate
  --epochs EPOCHS       Number of epochs
  --node_enc_dim NODE_ENC_DIM
                        embedding dim of node feature in A
  --emb_dim EMB_DIM     number of hidden gnn units
  --time_dim TIME_DIM   number of time encoding dims
  --n_layers N_LAYERS   number of hidden gnn layers
  --weight_decay WEIGHT_DECAY
                        Weight for L2 loss
  --gpu GPU             number of GPU
  --batch_size BATCH_SIZE
                        batch size
```

## Performance on Initial Test Set

The baseline got AUC of 0.511 on Dataset A and 0.510 on Dataset B.

## Tree
```bash
.
├── base_pipeline.py
├── csv2DGLgraph.py
├── DGLgraphs
│   ├── Dataset_A.bin
│   └── Dataset_B.bin
├── outputs
│   └── a.log
├── README 2.md
├── README.md
├── test_csvs
│   ├── input_A_initial.csv
│   └── input_B_initial.csv
└── train_csvs
    ├── edges_train_A.csv
    ├── edges_train_B.csv
    ├── edge_type_features.csv
    └── node_features.csv

4 directories, 13 files
```