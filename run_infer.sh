set -e

python infer.py ../ISLES-2022-npz-multimodal/all/sub-strokecase0003.npz ../runs/mae_decoder_multimodal/best.pth \
    --out_dir ../runs/mae_multimodal_infer --threshold 0.5