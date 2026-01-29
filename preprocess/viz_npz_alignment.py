import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def find_lesion_slices(mask_zhw: np.ndarray):
    zsum = mask_zhw.reshape(mask_zhw.shape[0], -1).sum(axis=1)
    return np.where(zsum > 0)[0].tolist()


def normalize_for_display(x: np.ndarray, p_lo=1.0, p_hi=99.0):
    x = x.astype(np.float32, copy=False)
    lo, hi = np.percentile(x, [p_lo, p_hi])
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    x = np.clip(x, lo, hi)
    return (x - lo) / (hi - lo + 1e-8)


def checkerboard(a: np.ndarray, b: np.ndarray, block: int = 16):
    """
    a,b: [H,W] in [0,1]
    returns checkerboard composite
    """
    H, W = a.shape
    yy = (np.arange(H) // block)[:, None]
    xx = (np.arange(W) // block)[None, :]
    mask = ((yy + xx) % 2) == 0
    out = np.where(mask, a, b)
    return out


def overlay_mask(img01: np.ndarray, mask: np.ndarray, alpha: float = 0.35):
    """
    img01: [H,W] in [0,1]
    mask:  [H,W] {0,1}
    returns RGB image [H,W,3]
    """
    rgb = np.stack([img01, img01, img01], axis=-1)
    m = (mask > 0).astype(np.float32)
    # red overlay
    rgb[..., 0] = np.clip(rgb[..., 0] * (1 - alpha * m) + alpha * m, 0, 1)
    rgb[..., 1] = np.clip(rgb[..., 1] * (1 - alpha * m), 0, 1)
    rgb[..., 2] = np.clip(rgb[..., 2] * (1 - alpha * m), 0, 1)
    return rgb


def plot_case(npz_path: Path, out_dir: Path, z_list, checker_block=16, alpha=0.35):
    d = np.load(npz_path, allow_pickle=False)
    required = ["dwi", "adc", "mask"]
    for k in required:
        if k not in d:
            raise KeyError(f"{npz_path}: missing key '{k}'. Found: {list(d.keys())}")

    dwi = d["dwi"]  # [Z,H,W]
    adc = d["adc"]
    mask = d["mask"].astype(np.uint8)

    flair = None
    if "flair" in d:
        flair = d["flair"]

    Z, H, W = dwi.shape
    out_dir.mkdir(parents=True, exist_ok=True)

    for z in z_list:
        z = int(z)
        if z < 0 or z >= Z:
            continue

        dwi_z = normalize_for_display(dwi[z])
        adc_z = normalize_for_display(adc[z])
        m_z = (mask[z] > 0).astype(np.uint8)

        if flair is not None:
            flair_z = normalize_for_display(flair[z])
        else:
            flair_z = None

        fig = plt.figure(figsize=(12, 7))

        # Row 1: modalities + mask overlay
        ax1 = plt.subplot(2, 3, 1)
        ax1.imshow(overlay_mask(dwi_z, m_z, alpha=alpha))
        ax1.set_title(f"DWI z={z} (+mask)")
        ax1.axis("off")

        ax2 = plt.subplot(2, 3, 2)
        ax2.imshow(overlay_mask(adc_z, m_z, alpha=alpha))
        ax2.set_title("ADC (+mask)")
        ax2.axis("off")

        ax3 = plt.subplot(2, 3, 3)
        if flair_z is not None:
            ax3.imshow(overlay_mask(flair_z, m_z, alpha=alpha))
            ax3.set_title("FLAIR (+mask)")
        else:
            ax3.imshow(m_z, cmap="gray")
            ax3.set_title("FLAIR missing; showing mask")
        ax3.axis("off")

        # Row 2: alignment diagnostics
        ax4 = plt.subplot(2, 3, 4)
        ax4.imshow(dwi_z, cmap="gray")
        ax4.set_title("DWI (display norm)")
        ax4.axis("off")

        ax5 = plt.subplot(2, 3, 5)
        if flair_z is not None:
            cb = checkerboard(dwi_z, flair_z, block=checker_block)
            ax5.imshow(cb, cmap="gray")
            ax5.set_title(f"Checkerboard DWI vs FLAIR (block={checker_block})")
        else:
            ax5.imshow(adc_z, cmap="gray")
            ax5.set_title("ADC (no FLAIR to compare)")
        ax5.axis("off")

        ax6 = plt.subplot(2, 3, 6)
        ax6.imshow(m_z, cmap="gray")
        ax6.set_title("Mask")
        ax6.axis("off")

        fig.suptitle(npz_path.stem)
        fig.tight_layout()

        out_path = out_dir / f"{npz_path.stem}_z{z:03d}.png"
        plt.savefig(out_path, dpi=200)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz_root", type=str, help="Folder containing case .npz files (e.g., OUT/all)")
    ap.add_argument("--out_dir", type=str, default="qc_align_npz", help="Where to save PNGs")
    ap.add_argument("--cases", type=int, default=12, help="How many cases to sample")
    ap.add_argument("--per_case", type=int, default=4, help="How many z-slices to plot per case")
    ap.add_argument("--mode", type=str, default="lesion", choices=["lesion", "uniform", "mixed"],
                    help="How to pick z-slices: lesion=only lesion slices; uniform=evenly spaced; mixed=half lesion half uniform")
    ap.add_argument("--checker_block", type=int, default=16, help="Checkerboard block size (pixels)")
    ap.add_argument("--alpha", type=float, default=0.35, help="Mask overlay alpha")
    ap.add_argument("--seed", type=int, default=0, help="Sampling seed")
    args = ap.parse_args()

    npz_root = Path(args.npz_root)
    out_dir = Path(args.out_dir)
    rng = np.random.default_rng(args.seed)

    npz_paths = sorted(npz_root.glob("*.npz"))
    if not npz_paths:
        raise FileNotFoundError(f"No .npz files found in {npz_root}")

    if args.cases < len(npz_paths):
        pick = rng.choice(len(npz_paths), size=args.cases, replace=False)
        npz_paths = [npz_paths[i] for i in sorted(pick)]

    for p in npz_paths:
        d = np.load(p, allow_pickle=False)
        mask = d["mask"].astype(np.uint8)
        Z = mask.shape[0]

        lesion_z = find_lesion_slices(mask)
        z_list = []

        if args.mode == "lesion":
            if lesion_z:
                z_list = lesion_z[:]
                rng.shuffle(z_list)
                z_list = z_list[:args.per_case]
            else:
                # fallback to uniform if no lesion
                z_list = np.linspace(0, Z - 1, args.per_case).round().astype(int).tolist()

        elif args.mode == "uniform":
            z_list = np.linspace(0, Z - 1, args.per_case).round().astype(int).tolist()

        else:  # mixed
            half = args.per_case // 2
            if lesion_z:
                lz = lesion_z[:]
                rng.shuffle(lz)
                z_list.extend(lz[:half])
            z_list.extend(np.linspace(0, Z - 1, args.per_case - len(z_list)).round().astype(int).tolist())

        z_list = sorted(set(int(z) for z in z_list))
        plot_case(p, out_dir, z_list, checker_block=args.checker_block, alpha=args.alpha)

    print(f"Saved QC images to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
