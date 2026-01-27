import csv
import argparse
import random
from pathlib import Path

import numpy as np
import nibabel as nib
from tqdm import tqdm


def affine_close(a, b, tol=1e-3) -> bool:
    return np.allclose(a, b, atol=tol, rtol=0)


def load_canonical_nii(path: Path) -> tuple:
    nii = nib.load(str(path))
    nii_c = nib.as_closest_canonical(nii)
    data = nii_c.get_fdata(dtype=np.float32)  # float32 for compactness
    return nii_c, data


def ensure_3d(x: np.ndarray, name: str, path: Path) -> np.ndarray:
    # For ISLES DWI this should be 3D; if 4D, we pick volume 0.
    if x.ndim == 3:
        return x
    if x.ndim == 4:
        print("[Warning] The dataset is in 4D format; using volume 0.")
        return x[..., 0]
    raise ValueError(f"{name} at {path} must be 3D or 4D, got shape {x.shape}")


def maybe_resample_mask_to_image(mask_nii: nib.Nifti1Image, img_nii: nib.Nifti1Image) -> nib.Nifti1Image:
    """
    If mask and image aren't aligned, resample mask to image grid with nearest-neighbor.
    Requires scipy via nibabel.processing.
    """
    from nibabel.processing import resample_from_to  # may require scipy

    # resample mask -> image space using order=0 (nearest)
    res = resample_from_to(mask_nii, img_nii, order=0)
    return res


def to_ZHW(arr_xyz: np.ndarray) -> np.ndarray:
    """
    Convert nibabel-style [X,Y,Z] to [Z,H,W] (H=X, W=Y here).
    """
    if arr_xyz.ndim != 3:
        raise ValueError(f"Expected 3D array, got {arr_xyz.ndim}D")
    return np.moveaxis(arr_xyz, -1, 0)  # [Z,X,Y]


def find_pairs(isles_root: Path) -> list:
    """
    Returns list of (case_id, img_path, mask_path)
    Pattern:
      Image:  ISLES-2022/sub-strokecase0001/ses-0001/dwi/sub-strokecase0001_ses-0001_dwi.nii.gz
      Mask:   ISLES-2022/derivatives/sub-strokecase0001/ses-0001/sub-strokecase0001_ses-0001_msk.nii.gz
    """
    pairs = []
    # case dirs are direct children: sub-strokecaseXXXX
    for case_dir in sorted(isles_root.glob("sub-strokecase*")):
        if not case_dir.is_dir():
            continue
        case_id = case_dir.name  # e.g., sub-strokecase0001

        img_path = isles_root / case_id / "ses-0001" / "dwi" / f"{case_id}_ses-0001_dwi.nii.gz"
        msk_path = isles_root / "derivatives" / case_id / "ses-0001" / f"{case_id}_ses-0001_msk.nii.gz"

        if img_path.exists() and msk_path.exists():
            pairs.append((case_id, img_path, msk_path))
        else:
            # Some cases might be missing (or naming differs). We skip with a warning.
            missing = []
            if not img_path.exists():
                missing.append("img")
            if not msk_path.exists():
                missing.append("mask")
            print(f"[WARN] Skip {case_id}: missing {','.join(missing)}")
    return pairs


def write_npz_for_case(
    case_id: str,
    img_path: Path,
    msk_path: Path,
    out_npz_path: Path,
    allow_resample: bool = True,
) -> dict:
    img_nii, img = load_canonical_nii(img_path)
    msk_nii, msk = load_canonical_nii(msk_path)

    img = ensure_3d(img, "image", img_path)
    msk = ensure_3d(msk, "mask", msk_path)

    # If not aligned, try resampling mask to image grid
    if (img.shape != msk.shape) or (not affine_close(img_nii.affine, msk_nii.affine)):
        if not allow_resample:
            raise ValueError(
                f"{case_id}: image/mask misaligned.\n"
                f"  img shape {img.shape} mask shape {msk.shape}\n"
                f"  affine close: {affine_close(img_nii.affine, msk_nii.affine)}"
            )
        try:
            msk_res = maybe_resample_mask_to_image(msk_nii, img_nii)
            msk = msk_res.get_fdata(dtype=np.float32)
            msk = ensure_3d(msk, "mask(resampled)", msk_path)
            # update for reporting
            msk_nii = msk_res
        except Exception as e:
            raise RuntimeError(
                f"{case_id}: mask resampling failed. Install scipy if missing.\n"
                f"Original error: {repr(e)}"
            )

    # Convert to [Z,H,W] and binarize mask
    img_zhw = to_ZHW(img).astype(np.float32)
    msk_zhw = (to_ZHW(msk) > 0.5).astype(np.uint8)

    nonzero = int(msk_zhw.sum())
    np.savez_compressed(out_npz_path, img=img_zhw, mask=msk_zhw)

    return {
        "case_id": case_id,
        "img_path": str(img_path),
        "mask_path": str(msk_path),
        "out_npz": str(out_npz_path),
        "shape_ZHW": str(tuple(img_zhw.shape)),
        "mask_nonzero": nonzero,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="Path to ISLES-2022 root folder")
    ap.add_argument("out", help="Output folder")
    ap.add_argument("--val_frac", type=float, default=0.2, help="Validation fraction (by case)")
    ap.add_argument("--seed", type=int, default=42, help="Split seed")
    ap.add_argument("--allow_resample", action="store_true", help="If set, resample masks if misaligned")
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

    pairs = find_pairs(root)
    if not pairs:
        raise FileNotFoundError(f"No image/mask pairs found under {root}. Check the folder name and structure.")

    # Split by case_id
    rng = random.Random(args.seed)
    case_ids = [cid for cid, _, _ in pairs]
    idx = list(range(len(case_ids)))
    rng.shuffle(idx)
    n_val = int(round(len(idx) * args.val_frac))
    val_set = set(idx[:n_val])

    train_cases = []
    val_cases = []
    for i, (case_id, img_path, msk_path) in enumerate(pairs):
        if i in val_set:
            val_cases.append((case_id, img_path, msk_path))
        else:
            train_cases.append((case_id, img_path, msk_path))

    # Convert all -> OUT/all/*.npz
    manifest_rows = []
    errors = 0

    for split_name, split_list in [("train", train_cases), ("val", val_cases)]:
        for case_id, img_path, msk_path in tqdm(split_list, desc=f"Converting {split_name}"):
            out_npz = out_all / f"{case_id}.npz"
            try:
                row = write_npz_for_case(
                    case_id=case_id,
                    img_path=img_path,
                    msk_path=msk_path,
                    out_npz_path=out_npz,
                    allow_resample=args.allow_resample,
                )
                row["split"] = split_name
                manifest_rows.append(row)

                # Optional: copy into split folders
                if args.copy_split_files:
                    dst = (out_train if split_name == "train" else out_val) / out_npz.name
                    # Copy bytes (not symlink) for max compatibility
                    if not dst.exists():
                        dst.write_bytes(out_npz.read_bytes())

            except Exception as e:
                errors += 1
                print(f"[ERROR] {case_id}: {e}")

    # Write split lists
    train_txt = out_splits / "train.txt"
    val_txt = out_splits / "val.txt"
    with open(train_txt, "w", encoding="utf-8") as f:
        for cid, _, _ in train_cases:
            f.write(cid + "\n")
    with open(val_txt, "w", encoding="utf-8") as f:
        for cid, _, _ in val_cases:
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
    print(f"  Found pairs: {len(pairs)}")
    print(f"  Train cases: {len(train_cases)}")
    print(f"  Val cases  : {len(val_cases)}")
    print(f"  Errors     : {errors}")
    print(f"  NPZ folder : {out_all}")
    print(f"  Splits     : {out_splits}")
    if args.copy_split_files:
        print(f"  Train/Val copies: {out_train} | {out_val}")
    print(f"  Manifest   : {manifest_path}")

    print("\nTo train with your current decoder code, point data_root to OUT/all:")
    print(f"  python train.py --data_root {out_all} --mae_ckpt /path/to/mae.pth --out_dir runs/demo")


if __name__ == "__main__":
    main()
