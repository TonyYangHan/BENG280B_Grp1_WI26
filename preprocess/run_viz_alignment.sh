set -e
abs_path="/mnt/d/Work/UCSD/Master/WI26/BENG_280B/BENG280B_Grp1_WI26"
python "${abs_path}/preprocess/viz_npz_alignment.py" "${abs_path}/../ISLES-2022-npz-multimodal/all" \
    --out_dir "${abs_path}/../qc_align" --cases 10 --per_case 4 --mode lesion