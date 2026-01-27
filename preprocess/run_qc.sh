set -e

dir="../ISLES-2022/"

python isles_qc.py --img ${dir}sub-strokecase0001/ses-0001/dwi/sub-strokecase0001_ses-0001_dwi.nii.gz \
    --mask ${dir}derivatives/sub-strokecase0001/ses-0001/sub-strokecase0001_ses-0001_msk.nii.gz \
    --vol_idx 0 --display_scale 2