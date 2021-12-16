cd "$(dirname $0)"

python3 ../main.py \
--dataset B \
--seed 0 \
--gpu 1 \
--epochs 5 \
--time_dim 1 \
--n_layers 2 \
--emb_dim 10 \
--hid_dim 30 \
--lr 1e-3 \
--batch_size 60000 \
--weight_decay 0. \
--num_heads 3
