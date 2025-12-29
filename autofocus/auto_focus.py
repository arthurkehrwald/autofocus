#!/usr/bin/env python3
"""Auto-focus helper for Siemens star image set.

Usage:
  python3 autofocus/auto_focus.py --dir test-images

The script finds the Siemens star by scanning for the window with maximum
gradient energy, crops that region, computes a sharpness score per image,
and reports the filename with the highest sharpness.
"""
from pathlib import Path
import argparse
import cv2
import numpy as np
import sys


def compute_gradient_energy(gray: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    energy = gx * gx + gy * gy
    return energy


def find_best_crop_center(energy: np.ndarray, crop_w: int, crop_h: int) -> tuple:
    # Sum energy over sliding window using a box filter
    kernel = np.ones((crop_h, crop_w), dtype=np.float64)
    summed = cv2.filter2D(energy, -1, kernel)
    # location of max summed energy
    minv, maxv, minloc, maxloc = cv2.minMaxLoc(summed)
    return maxloc  # (x, y)


def crop_around(img: np.ndarray, center: tuple, crop_w: int, crop_h: int) -> np.ndarray:
    x, y = center
    h, w = img.shape[:2]
    x1 = int(max(0, x - crop_w // 2))
    y1 = int(max(0, y - crop_h // 2))
    x2 = int(min(w, x1 + crop_w))
    y2 = int(min(h, y1 + crop_h))
    # adjust if near border
    if x2 - x1 < crop_w:
        x1 = max(0, x2 - crop_w)
    if y2 - y1 < crop_h:
        y1 = max(0, y2 - crop_h)
    return img[y1:y2, x1:x2]


def sharpness_score(crop_gray: np.ndarray) -> float:
    # Use variance of Laplacian (classic focus measure)
    lap = cv2.Laplacian(crop_gray, cv2.CV_64F)
    return float(lap.var())


def process_image(path: Path, crop_w: int, crop_h: int, save_crop: Path = None) -> tuple:
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Unable to read image {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Try ArUco-based cropping: look for four identical markers from a 4x4 dictionary.
    crop = None
    try:
        # obtain 4x4 dictionary (50 is a common choice); try several if available
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners_list, ids, _ = detector.detectMarkers(gray)
        if ids is not None and len(ids) >= 4:
            ids_flat = ids.flatten()
            uniq, counts = np.unique(ids_flat, return_counts=True)
            # choose an id that appears at least 4 times
            candidates = uniq[counts >= 4]
            if len(candidates) > 0:
                chosen_id = int(candidates[0])
                pts = []
                for i, mid in enumerate(ids_flat):
                    if int(mid) == chosen_id:
                        # corners_list[i] is (1,4,2) array; take first corner
                        c = corners_list[i][0][0]
                        pts.append(c)
                if len(pts) >= 4:
                    pts = np.array(pts[:4], dtype=np.float32)
                    xs = pts[:, 0]
                    ys = pts[:, 1]
                    x1 = int(np.min(xs))
                    y1 = int(np.min(ys))
                    x2 = int(np.max(xs))
                    y2 = int(np.max(ys))
                    # ensure within image
                    h, w = img.shape[:2]
                    x1 = max(0, min(w - 1, x1))
                    x2 = max(0, min(w, x2))
                    y1 = max(0, min(h - 1, y1))
                    y2 = max(0, min(h, y2))
                    if x2 > x1 and y2 > y1:
                        crop = img[y1:y2, x1:x2]
    except Exception:
        crop = None

    # Fallback to gradient-energy-based cropping if ArUco detection failed
    if crop is None:
        energy = compute_gradient_energy(gray)
        center = find_best_crop_center(energy, crop_w, crop_h)
        crop = crop_around(img, center, crop_w, crop_h)
    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    score = sharpness_score(crop_gray)
    if save_crop is not None:
        save_crop.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_crop), crop)
    return score


def collect_images(dirpath: Path):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp")
    files = []
    for e in exts:
        files.extend(sorted(dirpath.glob(e)))
    return files


def main():
    p = argparse.ArgumentParser(description="Auto-select most in-focus Siemens star image")
    p.add_argument("--dir", "-d", type=Path, default="./test-images", help="Directory containing images")
    p.add_argument("--crop-fraction", type=float, default=0.25,
                   help="Fraction of min(image_width,image_height) used for crop (default 0.25)")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    if not args.dir.exists():
        print("Directory not found:", args.dir, file=sys.stderr)
        sys.exit(2)

    imgs = collect_images(args.dir)
    if not imgs:
        print("No images found in:", args.dir, file=sys.stderr)
        sys.exit(2)

    # derive crop size from first image
    sample = cv2.imread(str(imgs[0]))
    if sample is None:
        print("Unable to read sample image", imgs[0], file=sys.stderr)
        sys.exit(2)
    h, w = sample.shape[:2]
    base = int(min(h, w) * args.crop_fraction)
    crop_w = crop_h = max(32, base)

    best = None
    results = []
    for i, path in enumerate(imgs):
        try:
            out_crop = Path("test_artifacts") / "crops" / f"crop_{i:03d}_{path.name}"
            score = process_image(path, crop_w, crop_h, save_crop=out_crop)
            results.append((path, score))
            if args.verbose:
                print(f"{path.name}: score={score:.3f}")
            if best is None or score > best[1]:
                best = (path, score)
        except Exception as e:
            print(f"Skipping {path}: {e}", file=sys.stderr)

    if best:
        print("Best (sharpest) image:", best[0].as_posix())
        print(f"Score: {best[1]:.3f}")
    else:
        print("No usable images found.")


if __name__ == "__main__":
    main()
