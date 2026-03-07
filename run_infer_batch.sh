set -e

data_dir="../ISLES-2022-npz-multimodal_clean"
model_path="../runs/terence_strategy/best_terence.pth"
python infer_batch.py $data_dir/splits/val.txt $data_dir/all/ $model_path

# python infer_batch.py $data_dir/all/sub-strokecase0004.npz $model_path --cls_ckpt ../runs/mlp_slice_classifier/best_mlp.pth