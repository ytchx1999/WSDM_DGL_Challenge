cd "$(dirname $0)"

python3 ../main.py \
--dataset A \
--seed 0 \
--gpu 0 \
--epochs 10 \
--time_dim 3 \
--n_layers 2 \
--emb_dim 10 \
--hid_dim 20 \
--node_enc_dim 16 \
--lr 1e-3 \
--batch_size 20000 \
--weight_decay 5e-4 \
--num_heads 3