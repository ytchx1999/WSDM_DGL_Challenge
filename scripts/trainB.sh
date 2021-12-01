cd "$(dirname $0)"

python3 ../main.py \
--dataset B \
--gpu 1 \
--epochs 50 \
--time_dim 1 \
--n_layers 2 \
--emb_dim 10 \
--lr 1e-3 \
--batch_size 50000