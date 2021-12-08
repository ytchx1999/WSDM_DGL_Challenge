cd "$(dirname $0)"

python3 ../main.py \
--dataset B \
--gpu 1 \
--epochs 10 \
--time_dim 1 \
--n_layers 2 \
--emb_dim 10 \
--hid_dim 32 \
--lr 1e-3 \
--batch_size 50000 \
--num_heads 3