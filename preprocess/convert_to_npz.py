import csv
import argparse
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import nibabel as nib
from tqdm import tqdm


def affine_close(a, b, tol=1e-3) -> bool:
    return np.allclose(a, b, atol=tol, rtol=0)


def load_canonical_nii(path: Path) -> Tuple[nib.Nifti1Image, np.ndarray]:
    nii = nib.load(str(path))
    nii_c = nib.as_closest_canonical(nii)
    data = nii_c.get_fdata(dtype=np.float32)  # float32 for compactness
    return nii_c, data


def ensure_3d(x: np.ndarray, name: str, path: Path) -> np.ndarray:
    if x.ndim == 3:
        return x
    if x.ndim == 4:
        print(f"[WARN] {name} is 4D at {path}; using volume 0.")
        return x[..., 0]
    raise ValueError(f"{name} at {path} must be 3D or 4D, got shape {x.shape}")


def resample_to_ref(moving_nii: nib.Nifti1Image, ref_nii: nib.Nifti1Image, order: int) -> nib.Nifti1Image:
    """
    Resample moving -> ref grid. order=0 nearest (masks), order=1 linear (continuous).
    Requires scipy via nibabel.processing.
    """
    from nibabel.processing import resample_from_to  # may require scipy
    return resample_from_to(moving_nii, ref_nii, order=order)


def to_ZHW(arr_xyz: np.ndarray) -> np.ndarray:
    """
    Convert nibabel-style [X,Y,Z] -> [Z,H,W] (H=X, W=Y).
    """
    if arr_xyz.ndim != 3:
        raise ValueError(f"Expected 3D array, got {arr_xyz.ndim}D")
    return np.moveaxis(arr_xyz, -1, 0)  # [Z,X,Y]


def find_quadruples(isles_root: Path) -> List[Tuple[str, Path, Path, Path, Path]]:
    """
    Returns list of (case_id, dwi_path, adc_path, flair_path, mask_path)

    Example paths:
      DWI:   ISLES-2022/sub-strokecase0001/ses-0001/dwi/sub-strokecase0001_ses-0001_dwi.nii.gz
      ADC:   ISLES-2022/sub-strokecase0001/ses-0001/dwi/sub-strokecase0001_ses-0001_adc.nii.gz
      FLAIR: ISLES-2022/sub-strokecase0001/ses-0001/anat/sub-strokecase0001_ses-0001_FLAIR.nii.gz
      MASK:  ISLES-2022/derivatives/sub-strokecase0001/ses-0001/sub-strokecase0001_ses-0001_msk.nii.gz
    """
    out = []
    for case_dir in sorted(isles_root.glob("sub-strokecase*")):
        if not case_dir.is_dir():
            continue
        case_id = case_dir.name

        dwi_path = isles_root / case_id / "ses-0001" / "dwi" / f"{case_id}_ses-0001_dwi.nii.gz"
        adc_path = isles_root / case_id / "ses-0001" / "dwi" / f"{case_id}_ses-0001_adc.nii.gz"
        flair_path = isles_root / case_id / "ses-0001" / "anat" / f"{case_id}_ses-0001_FLAIR.nii.gz"
        msk_path = isles_root / "derivatives" / case_id / "ses-0001" / f"{case_id}_ses-0001_msk.nii.gz"

        missing = []
        for name, p in [("dwi", dwi_path), ("adc", adc_path), ("flair", flair_path), ("mask", msk_path)]:
            if not p.exists():
                missing.append(name)

        if missing:
            print(f"[WARN] Skip {case_id}: missing {','.join(missing)}")
            continue

        out.append((case_id, dwi_path, adc_path, flair_path, msk_path))
    return out


def write_npz_for_case(
    case_id: str,
    dwi_path: Path,
    adc_path: Path,
    flair_path: Path,
    msk_path: Path,
    out_npz_path: Path,
    allow_resample: bool = True,
) -> Dict[str, Any]:
    # Load
    dwi_nii, dwi = load_canonical_nii(dwi_path)
    adc_nii, adc = load_canonical_nii(adc_path)
    flair_nii, flair = load_canonical_nii(flair_path)
    msk_nii, msk = load_canonical_nii(msk_path)

    dwi = ensure_3d(dwi, "dwi", dwi_path)
    adc = ensure_3d(adc, "adc", adc_path)
    flair = ensure_3d(flair, "flair", flair_path)
    msk = ensure_3d(msk, "mask", msk_path)

    # Make everything match DWI grid (shape+affine)
    def _need_match(img_arr, img_nii) -> bool:
        return (img_arr.shape != dwi.shape) or (not affine_close(img_nii.affine, dwi_nii.affine))

    if _need_match(adc, adc_nii) or _need_match(flair, flair_nii) or _need_match(msk, msk_nii):
        if not allow_resample:
            raise ValueError(
                f"{case_id}: modalities/mask misaligned with DWI.\n"
                f"  dwi shape {dwi.shape}\n"
                f"  adc shape {adc.shape}, affine close={affine_close(adc_nii.affine, dwi_nii.affine)}\n"
                f"  flair shape {flair.shape}, affine close={affine_close(flair_nii.affine, dwi_nii.affine)}\n"
                f"  mask shape {msk.shape}, affine close={affine_close(msk_nii.affine, dwi_nii.affine)}"
            )

        # Resample continuous images with linear interpolation; mask with nearest
        print(f"[Warning] {case_id}: resampling modalities/mask to match DWI grid.")
        try:
            if _need_match(adc, adc_nii):
                adc_nii_r = resample_to_ref(adc_nii, dwi_nii, order=1)
                adc = ensure_3d(adc_nii_r.get_fdata(dtype=np.float32), "adc(resampled)", adc_path)

            if _need_match(flair, flair_nii):
                flair_nii_r = resample_to_ref(flair_nii, dwi_nii, order=1)
                flair = ensure_3d(flair_nii_r.get_fdata(dtype=np.float32), "flair(resampled)", flair_path)

            if _need_match(msk, msk_nii):
                msk_nii_r = resample_to_ref(msk_nii, dwi_nii, order=0)
                msk = ensure_3d(msk_nii_r.get_fdata(dtype=np.float32), "mask(resampled)", msk_path)

        except Exception as e:
            raise RuntimeError(
                f"{case_id}: resampling failed. Install scipy if missing.\n"
                f"Original error: {repr(e)}"
            )

    # Convert to [Z,H,W]
    dwi_zhw = to_ZHW(dwi).astype(np.float32)
    adc_zhw = to_ZHW(adc).astype(np.float32)
    flair_zhw = to_ZHW(flair).astype(np.float32)
    msk_zhw = (to_ZHW(msk) > 0.5).astype(np.uint8)

    nonzero = int(msk_zhw.sum())
    np.savez_compressed(out_npz_path, dwi=dwi_zhw, adc=adc_zhw, flair=flair_zhw, mask=msk_zhw)

    return {
        "case_id": case_id,
        "dwi_path": str(dwi_path),
        "adc_path": str(adc_path),
        "flair_path": str(flair_path),
        "mask_path": str(msk_path),
        "out_npz": str(out_npz_path),
        "shape_ZHW": str(tuple(dwi_zhw.shape)),
        "mask_nonzero": nonzero,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="Path to ISLES-2022 root folder")
    ap.add_argument("out", help="Output folder")
    ap.add_argument("--val_frac", "-f", type=float, default=0.2, help="Validation fraction (by case)")
    ap.add_argument("--seed", "-s", type=int, default=42, help="Split seed")
    ap.add_argument("--allow_resample", "-rs", action="store_true", help="If set, resample ADC/FLAIR/MASK if misaligned with DWI")
    ap.add_argument("--copy_split_files", action="store_true",
                    help="If set, copies NPZ into OUT/train and OUT/val. (Always writes OUT/all)")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()

    out_all = out / "all"
    out_train = out / "train"
    out_val = out / "val"
    out_splits = out / "splits"
    out.mkdir(parents=True, exist_ok=True)
    out_all.mkdir(parents=True, exist_ok=True)
    out_splits.mkdir(parents=True, exist_ok=True)
    if args.copy_split_files:
        out_train.mkdir(parents=True, exist_ok=True)
        out_val.mkdir(parents=True, exist_ok=True)

    quads = find_quadruples(root)
    if not quads:
        raise FileNotFoundError(f"No (DWI,ADC,FLAIR,MASK) sets found under {root}. Check structure and filenames.")

    rng = random.Random(args.seed)
    idx = list(range(len(quads)))
    rng.shuffle(idx)
    n_val = int(round(len(idx) * args.val_frac))
    val_set = set(idx[:n_val])

    train_cases = []
    val_cases = []
    for i, quad in enumerate(quads):
        (val_cases if i in val_set else train_cases).append(quad)

    manifest_rows: List[Dict[str, Any]] = []
    errors = 0

    for split_name, split_list in [("train", train_cases), ("val", val_cases)]:
        for case_id, dwi_path, adc_path, flair_path, msk_path in tqdm(split_list, desc=f"Converting {split_name}"):
            out_npz = out_all / f"{case_id}.npz"
            try:
                row = write_npz_for_case(
                    case_id=case_id,
                    dwi_path=dwi_path,
                    adc_path=adc_path,
                    flair_path=flair_path,
                    msk_path=msk_path,
                    out_npz_path=out_npz,
                    allow_resample=args.allow_resample,
                )
                row["split"] = split_name
                manifest_rows.append(row)

                if args.copy_split_files:
                    dst = (out_train if split_name == "train" else out_val) / out_npz.name
                    if not dst.exists():
                        dst.write_bytes(out_npz.read_bytes())

            except Exception as e:
                errors += 1
                print(f"[ERROR] {case_id}: {e}")

    # Write split lists
    train_txt = out_splits / "train.txt"
    val_txt = out_splits / "val.txt"
    with open(train_txt, "w", encoding="utf-8") as f:
        for cid, *_ in train_cases:
            f.write(cid + "\n")
    with open(val_txt, "w", encoding="utf-8") as f:
        for cid, *_ in val_cases:
            f.write(cid + "\n")

    # Write manifest.csv
    manifest_path = out / "manifest.csv"
    if manifest_rows:
        keys = list(manifest_rows[0].keys())
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in manifest_rows:
                w.writerow(r)
    else:
        manifest_path.write_text("", encoding="utf-8")

    print("\nDone.")
    print(f"  Found sets : {len(quads)}")
    print(f"  Train cases: {len(train_cases)}")
    print(f"  Val cases  : {len(val_cases)}")
    print(f"  Errors     : {errors}")
    print(f"  NPZ folder : {out_all}")
    print(f"  Splits     : {out_splits}")
    if args.copy_split_files:
        print(f"  Train/Val copies: {out_train} | {out_val}")
    print(f"  Manifest   : {manifest_path}")


if __name__ == "__main__":
    main()