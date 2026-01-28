set -e

python infer.py ../ISLES-2022-npz/all/sub-strokecase0003.npz ../runs/mae_decoder/best.pth --out_dir ../runs/mae_decoder_v1 \
    --threshold 0.5