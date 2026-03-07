set -e

python infer.py ../ISLES-2022-npz-multimodal_clean/all/sub-strokecase0004.npz ../runs/terence_strategy/best_terence.pth \
    "../inferences/tinger_code/sub-strokecase0004" --cls_ckpt ../runs/mlp_slice_classifier/best_mlp.pth