
"""
Camera calibration (single camera) using a chessboard.



Notes:
- rows/cols refer to INNER corners (e.g., a 9x6 inner-corner board is common).
- square_size is the REAL size of one square edge in meters (measure with a ruler).
"""

import argparse
import glob
import os
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class CalibrationResult:
    rms: float
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    rvecs: List[np.ndarray]
    tvecs: List[np.ndarray]
    mean_reproj_error_px: float
    image_size: Tuple[int, int]
    used_image_paths: List[str]


def _compute_reprojection_error(objpoints, imgpoints, rvecs, tvecs, K, dist) -> float:
    total_error = 0.0
    total_points = 0
    for objp, imgp, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
        proj = proj.reshape(-1, 2)
        imgp2 = imgp.reshape(-1, 2)
        err = np.linalg.norm(imgp2 - proj, axis=1).sum()
        total_error += err
        total_points += len(objp)
    return float(total_error / max(total_points, 1))


def calibrate_from_images(
    image_paths: List[str],
    board_size: Tuple[int, int],
    square_size_m: float,
    show: bool = False,
    min_images=5,
) -> CalibrationResult:
    cols, rows = board_size  # inner corners

    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_size_m)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

    objpoints = []
    imgpoints = []
    used_paths: List[str] = []
    img_size = None

    for p in image_paths:
        img = cv2.imread(p)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])  # (w,h)

        ret, corners = cv2.findChessboardCorners(
            gray, (cols, rows),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        if not ret:
            continue

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        objpoints.append(objp.copy())
        imgpoints.append(corners2)
        used_paths.append(p)

        if show:
            vis = img.copy()
            cv2.drawChessboardCorners(vis, (cols, rows), corners2, ret)
            cv2.imshow("Detected chessboard corners (press any key)", vis)
            cv2.waitKey(0)

    if show:
        cv2.destroyAllWindows()

    # Need at least 3 views for a meaningful calibration; more is better.
    if len(objpoints) < 3:
        raise RuntimeError(
            f"Not enough valid calibration images ({len(objpoints)} found). "
            "Capture more images with the full chessboard visible."
        )
# Warn (but do not crash) if fewer than requested.
    if len(objpoints) < min_images:
        print(f"WARNING: Only {len(objpoints)} valid images found; "
          f"recommended >= {min_images}. Continuing anyway (accuracy may be poor).")

    

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None
    )
    mean_err = _compute_reprojection_error(objpoints, imgpoints, rvecs, tvecs, K, dist)

    return CalibrationResult(
        rms=float(rms),
        camera_matrix=K,
        dist_coeffs=dist,
        rvecs=rvecs,
        tvecs=tvecs,
        mean_reproj_error_px=mean_err,
        image_size=img_size,
        used_image_paths=used_paths,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help='Glob, e.g. "calib_images/*.jpg"')
    ap.add_argument("--rows", type=int, default=6, help="Inner-corner rows")
    ap.add_argument("--cols", type=int, default=9, help="Inner-corner cols")
    ap.add_argument("--square_size", type=float, required=True, help="Square size in meters")
    ap.add_argument("--output", default="calib.npz", help="Output .npz path")
    ap.add_argument("--show", action="store_true", help="Visualize detected corners")
    ap.add_argument("--min_images", type=int, default=5,
                help="Minimum detected calibration images required (default: 5)")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.images))
    if not paths:
        raise SystemExit(f"No images matched: {args.images}")

    result = calibrate_from_images(
        paths, board_size=(args.cols, args.rows), square_size_m=args.square_size, show=args.show, min_images=args.min_images
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez(
        args.output,
        camera_matrix=result.camera_matrix,
        dist_coeffs=result.dist_coeffs,
        rms=result.rms,
        mean_reproj_error_px=result.mean_reproj_error_px,
        image_width=result.image_size[0],
        image_height=result.image_size[1],
        rows=args.rows,
        cols=args.cols,
        square_size_m=args.square_size,
        used_images=np.array(result.used_image_paths, dtype=object),
    )

    print("\n=== Calibration complete ===")
    print(f"Images matched by glob:        {len(paths)}")
    print(f"Images with corners detected:  {len(result.used_image_paths)}")
    print(f"RMS reprojection error (OpenCV): {result.rms:.4f}")
    print(f"Mean reprojection error (px):    {result.mean_reproj_error_px:.4f}")
    print("Camera matrix K:\n", result.camera_matrix)
    print("Distortion coeffs:\n", result.dist_coeffs.ravel())
    print(f"Saved to: {args.output}\n")


if __name__ == "__main__":
    main()
