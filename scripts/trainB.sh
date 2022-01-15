cd "$(dirname $0)"

python3 ../main.py \
--dataset B \
--seed 0 \
--gpu 1 \
--epochs 10 \
--time_dim 1 \
--n_layers 2 \
--emb_dim 10 \
--hid_dim 10 \
--lr 1e-3 \
--batch_size 60000 \
--weight_decay 5e-4 \
--num_heads 3
