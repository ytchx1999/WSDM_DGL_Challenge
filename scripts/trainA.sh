cd "$(dirname $0)"

python3 ../main.py \
--dataset A \
--gpu 0 \
--epochs 50 \
--time_dim 3 \
--n_layers 2 \
--emb_dim 10 \
--node_enc_dim 128 \
--lr 1e-3 \
--batch_size 10000