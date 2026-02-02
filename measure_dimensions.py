#!/usr/bin/env python3
"""
Measure real-world 2D dimensions of a (roughly) fronto-parallel planar object using
perspective projection:

    width_m  = width_px  * Z / fx
    height_m = height_px * Z / fy

Batch mode:
- Measure one or more images in one run.
- Optionally compute error vs ground-truth (true width/height).
- If no ground-truth is given and you provide 2 images, it computes error of image2 vs image1.

You click 4 corners per image: TL, TR, BR, BL.
The script undistorts first using calibration.

Examples:
  python measure_dimensions.py --image measurements/hawkins_near.jpeg --calib calib.npz --distance 1.2

  python measure_dimensions.py --images measurements/hawkins_near.jpeg measurements/hawkins_3m.jpeg \
      --distances 1.2 3.0 --calib calib.npz --true_width_mm 55 --true_height_mm 55
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np


@dataclass
class Measurement:
    image: str
    distance_m: float
    width_m: float
    height_m: float
    corners_px: List[Tuple[int, int]]
    annotated_path: str
    json_path: str


def _euclid(p, q) -> float:
    return float(np.hypot(p[0] - q[0], p[1] - q[1]))


def _order_points_clockwise(pts: np.ndarray) -> np.ndarray:
    # TL, TR, BR, BL
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _click_4_corners(image_bgr: np.ndarray, window_title: str) -> List[Tuple[int, int]]:
    pts: List[Tuple[int, int]] = []
    vis = image_bgr.copy()

    def redraw():
        nonlocal vis
        vis = image_bgr.copy()
        for i, p in enumerate(pts):
            cv2.circle(vis, p, 6, (0, 255, 0), -1)
            cv2.putText(
                vis, str(i + 1), (p[0] + 8, p[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
        if len(pts) == 4:
            poly = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [poly], True, (0, 255, 0), 2)

    def on_mouse(event, x, y, flags, param):
        nonlocal pts
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((int(x), int(y)))
            redraw()

    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_title, on_mouse)

    redraw()
    while True:
        cv2.imshow(window_title, vis)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q') or key == 27:
            cv2.destroyAllWindows()
            raise SystemExit("User quit.")
        if key == ord('r'):
            pts = []
            redraw()
        if len(pts) == 4:
            break

    cv2.destroyWindow(window_title)

    pts_np = _order_points_clockwise(np.array(pts, dtype=np.float32))
    return [(int(x), int(y)) for x, y in pts_np]


def _measure_one_image(img_path: str, calib_path: str, distance_m: float, out_dir: str) -> Measurement:
    data = np.load(calib_path)
    K = data["camera_matrix"]
    dist = data["dist_coeffs"]

    img = cv2.imread(img_path)
    if img is None:
        raise SystemExit(f"Could not read image: {img_path}")

    h, w = img.shape[:2]
    newK, _roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
    undist = cv2.undistort(img, K, dist, None, newK)

    fx = float(newK[0, 0])
    fy = float(newK[1, 1])

    title = f"Click 4 corners (TL,TR,BR,BL) | {os.path.basename(img_path)} | r=reset q=quit"
    corners_px = _click_4_corners(undist, title)

    tl, tr, br, bl = np.array(corners_px, dtype=np.float32)

    width_px = 0.5 * (_euclid(tl, tr) + _euclid(bl, br))
    height_px = 0.5 * (_euclid(tl, bl) + _euclid(tr, br))

    width_m = width_px * float(distance_m) / fx
    height_m = height_px * float(distance_m) / fy

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(img_path))[0]
    json_path = os.path.join(out_dir, f"{stem}_measurement.json")
    img_out_path = os.path.join(out_dir, f"{stem}_annotated.png")

    ann = undist.copy()
    poly = np.array(corners_px, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(ann, [poly], True, (0, 255, 0), 2)
    cv2.putText(
        ann,
        f"W={width_m:.4f} m, H={height_m:.4f} m (Z={distance_m:.3f} m)",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(img_out_path, ann)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "image": img_path,
                "distance_m": float(distance_m),
                "width_m": float(width_m),
                "height_m": float(height_m),
                "corners_px": corners_px,
                "camera_matrix_used": newK.tolist(),
            },
            f,
            indent=2,
        )

    return Measurement(
        image=img_path,
        distance_m=float(distance_m),
        width_m=float(width_m),
        height_m=float(height_m),
        corners_px=corners_px,
        annotated_path=img_out_path,
        json_path=json_path,
    )


def _pct_err(est: float, true: float) -> float:
    return 100.0 * abs(est - true) / max(abs(true), 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True, help="Path to calib.npz from calibrate_camera.py")

    # Backward compatible single-image args:
    ap.add_argument("--image", help="Single image path")
    ap.add_argument("--distance", type=float, help="Single distance in meters")

    # New batch args:
    ap.add_argument("--images", nargs="+", help="One or more image paths")
    ap.add_argument("--distances", nargs="+", type=float, help="One or more distances (meters), match images")

    ap.add_argument("--out_dir", default="output", help="Output directory")

    # Optional ground-truth:
    ap.add_argument("--true_width_m", type=float, default=None, help="True width in meters (optional)")
    ap.add_argument("--true_height_m", type=float, default=None, help="True height in meters (optional)")
    ap.add_argument("--true_width_mm", type=float, default=None, help="True width in mm (optional)")
    ap.add_argument("--true_height_mm", type=float, default=None, help="True height in mm (optional)")

    args = ap.parse_args()

    # Build image list
    if args.images:
        img_list = args.images
        if not args.distances:
            raise SystemExit("When using --images, you must also provide --distances.")
        dist_list = args.distances
    else:
        if not args.image or args.distance is None:
            raise SystemExit("Provide either (--image and --distance) OR (--images and --distances).")
        img_list = [args.image]
        dist_list = [args.distance]

    # Allow single distance to apply to all images
    if len(dist_list) == 1 and len(img_list) > 1:
        dist_list = dist_list * len(img_list)

    if len(dist_list) != len(img_list):
        raise SystemExit(f"Number of distances ({len(dist_list)}) must match number of images ({len(img_list)}).")

    # Ground truth parsing
    true_w = args.true_width_m
    true_h = args.true_height_m
    if args.true_width_mm is not None:
        true_w = float(args.true_width_mm) / 1000.0
    if args.true_height_mm is not None:
        true_h = float(args.true_height_mm) / 1000.0

    measurements: List[Measurement] = []
    for img_path, Z in zip(img_list, dist_list):
        if Z <= 0:
            raise SystemExit("Distances must be positive.")
        m = _measure_one_image(img_path, args.calib, Z, args.out_dir)
        measurements.append(m)

        print("\n=== Measurement result ===")
        print(f"Image:   {m.image}")
        print(f"Z:       {m.distance_m:.3f} m")
        print(f"Width:   {m.width_m:.6f} m")
        print(f"Height:  {m.height_m:.6f} m")
        print(f"Saved:   {m.annotated_path}")
        print(f"Saved:   {m.json_path}")

        if true_w is not None and true_h is not None:
            print(f"Abs err W: {abs(m.width_m - true_w):.6f} m   (%err { _pct_err(m.width_m, true_w):.2f}%)")
            print(f"Abs err H: {abs(m.height_m - true_h):.6f} m   (%err { _pct_err(m.height_m, true_h):.2f}%)")

    # Summary + comparison for 2 images if no ground-truth
    summary_path = os.path.join(args.out_dir, "batch_summary.json")
    summary = {
        "calib": args.calib,
        "measurements": [
            {
                "image": m.image,
                "distance_m": m.distance_m,
                "width_m": m.width_m,
                "height_m": m.height_m,
                "annotated_path": m.annotated_path,
                "json_path": m.json_path,
            }
            for m in measurements
        ],
    }

    if true_w is not None and true_h is not None:
        summary["ground_truth_m"] = {"width_m": true_w, "height_m": true_h}
        summary["errors"] = [
            {
                "image": m.image,
                "abs_err_width_m": abs(m.width_m - true_w),
                "abs_err_height_m": abs(m.height_m - true_h),
                "pct_err_width": _pct_err(m.width_m, true_w),
                "pct_err_height": _pct_err(m.height_m, true_h),
            }
            for m in measurements
        ]
    elif len(measurements) == 2:
        m1, m2 = measurements
        summary["relative_error_image2_vs_image1"] = {
            "abs_diff_width_m": abs(m2.width_m - m1.width_m),
            "abs_diff_height_m": abs(m2.height_m - m1.height_m),
            "pct_diff_width": 100.0 * abs(m2.width_m - m1.width_m) / max(abs(m1.width_m), 1e-12),
            "pct_diff_height": 100.0 * abs(m2.height_m - m1.height_m) / max(abs(m1.height_m), 1e-12),
        }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved batch summary: {summary_path}\n")


if __name__ == "__main__":
    main()
