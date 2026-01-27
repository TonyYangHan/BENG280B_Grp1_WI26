import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def load_nii(path):
    nii = nib.load(path)
    # bring to closest canonical orientation (helps interpret overlays)
    nii_c = nib.as_closest_canonical(nii)
    data = nii_c.get_fdata(dtype=np.float32)
    return nii_c, data

def describe(nii, data, name=""):
    zooms = nii.header.get_zooms()
    ax = nib.aff2axcodes(nii.affine)
    print(f"\n[{name}]")
    print("  shape:", data.shape)
    print("  zooms:", zooms)
    print("  axcodes:", ax)
    print("  affine:\n", nii.affine)

def pick_3d(data, vol_idx=0):
    # If 4D, pick one volume
    if data.ndim == 4:
        if vol_idx < 0 or vol_idx >= data.shape[-1]:
            raise ValueError(f"vol_idx {vol_idx} out of range for shape {data.shape}")
        return data[..., vol_idx]
    if data.ndim != 3:
        raise ValueError(f"Expected 3D/4D, got {data.ndim}D")
    return data

def affine_close(a, b, tol=1e-3):
    return np.allclose(a, b, atol=tol, rtol=0)

# ---------------------------
# Display-only resizing helpers
# ---------------------------
def resize_bilinear_2d(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Pure-numpy bilinear resize for grayscale 2D arrays (display only)."""
    in_h, in_w = img.shape
    if (in_h, in_w) == (out_h, out_w):
        return img

    y = np.linspace(0, in_h - 1, out_h)
    x = np.linspace(0, in_w - 1, out_w)

    xi = np.arange(in_w)
    tmp = np.empty((in_h, out_w), dtype=np.float32)
    for i in range(in_h):
        tmp[i, :] = np.interp(x, xi, img[i, :])

    yi = np.arange(in_h)
    out = np.empty((out_h, out_w), dtype=np.float32)
    for j in range(out_w):
        out[:, j] = np.interp(y, yi, tmp[:, j])

    return out

def resize_nearest_2d(mask: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Nearest-neighbor resize for binary/label masks (display only)."""
    in_h, in_w = mask.shape
    if (in_h, in_w) == (out_h, out_w):
        return mask

    y = np.linspace(0, in_h - 1, out_h).round().astype(int)
    x = np.linspace(0, in_w - 1, out_w).round().astype(int)
    y = np.clip(y, 0, in_h - 1)
    x = np.clip(x, 0, in_w - 1)
    return mask[np.ix_(y, x)]

def show_overlay(img3d, msk3d, title="", z=None, display_size: int = 0, display_scale: int = 1):
    # choose a slice: either user-specified or the slice with most mask area
    if z is None:
        areas = msk3d.sum(axis=(0,1))  # if data is [X,Y,Z]
        z = int(np.argmax(areas)) if areas.max() > 0 else img3d.shape[2] // 2

    img2 = img3d[:, :, z]
    msk2 = msk3d[:, :, z]

    # Display-only resizing (does NOT affect alignment checks)
    if display_size and display_size > 0:
        out_h = out_w = int(display_size)
        img_disp = resize_bilinear_2d(img2, out_h, out_w)
        msk_disp = resize_nearest_2d(msk2, out_h, out_w)
    elif display_scale and display_scale > 1:
        out_h = int(img2.shape[0] * display_scale)
        out_w = int(img2.shape[1] * display_scale)
        img_disp = resize_bilinear_2d(img2, out_h, out_w)
        msk_disp = resize_nearest_2d(msk2, out_h, out_w)
    else:
        img_disp, msk_disp = img2, msk2

    # simple windowing for display
    vmin, vmax = np.percentile(img2, [1, 99])  # window based on native slice

    plt.figure(figsize=(10, 4))
    plt.suptitle(f"{title} | z={z} | display={img_disp.shape[0]}x{img_disp.shape[1]}")

    plt.subplot(1, 3, 1)
    plt.title("Image")
    plt.imshow(img_disp.T, origin="lower", vmin=vmin, vmax=vmax, cmap="gray", interpolation="bilinear")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Mask")
    plt.imshow(msk_disp.T, origin="lower", interpolation="nearest")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Overlay")
    plt.imshow(img_disp.T, origin="lower", vmin=vmin, vmax=vmax, cmap="gray", interpolation="bilinear")
    plt.imshow(msk_disp.T, origin="lower", alpha=0.35, interpolation="nearest")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def main(img_path, mask_path, vol_idx=0, display_size=0, display_scale=1):
    img_nii, img = load_nii(img_path)
    msk_nii, msk = load_nii(mask_path)

    describe(img_nii, img, "Image (canonical)")
    describe(msk_nii, msk, "Mask (canonical)")

    img3d = pick_3d(img, vol_idx=vol_idx)
    msk3d = pick_3d(msk, vol_idx=0)  # masks are usually 3D

    # Binarize mask
    msk3d = (msk3d > 0.5).astype(np.uint8)

    print("\nBasic checks:")
    print("  image 3D shape:", img3d.shape)
    print("  mask  3D shape:", msk3d.shape)
    print("  mask nonzero voxels:", int(msk3d.sum()))

    same_shape = (img3d.shape == msk3d.shape)
    same_affine = affine_close(img_nii.affine, msk_nii.affine, tol=1e-3)

    print("\nAlignment checks (in canonical space):")
    print("  same shape :", same_shape)
    print("  affine close:", same_affine)
    if not same_affine:
        diff = np.abs(img_nii.affine - msk_nii.affine).max()
        print("  max |affine diff|:", float(diff))

    if same_shape:
        show_overlay(
            img3d, msk3d,
            title="QC overlay (canonical)",
            display_size=display_size,
            display_scale=display_scale
        )
    else:
        print("\nNOTE: Shapes differ, so direct overlay isn't valid.")
        print("You likely need to resample mask to image grid (nearest-neighbor).")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--vol_idx", type=int, default=0)

    # New display options:
    ap.add_argument("--display_size", type=int, default=0,
                    help="Optional: upscale display to NxN (e.g., 224). 0 disables.")
    ap.add_argument("--display_scale", type=int, default=1,
                    help="Optional: upscale display by integer factor (e.g., 2). Ignored if display_size>0.")

    args = ap.parse_args()
    main(args.img, args.mask, args.vol_idx, args.display_size, args.display_scale)
