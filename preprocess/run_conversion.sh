set -e

abs_path="/mnt/d/Work/UCSD/Master/WI26/BENG_280B/BENG280B_Grp1_WI26"

python "${abs_path}/preprocess/convert_to_npz.py" "$abs_path/../ISLES-2022/" "$abs_path/../ISLES-2022-npz-multimodal/" -rs