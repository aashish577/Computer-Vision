
"""
Tilt-tolerant planar measurement using an ArUco marker to recover the plane pose.

Requires: opencv-contrib-python (for cv2.aruco)

Keys:
  r = reset clicked points
  q / ESC = quit
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class Measurement:
    width_m: float
    height_m: float
    corners_px: List[Tuple[int, int]]
    corners_plane_m: List[Tuple[float, float]]
    marker_tvec_m: Tuple[float, float, float]
    marker_distance_m: float


def _euclid2(a, b) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _order_points_clockwise(pts: np.ndarray) -> np.ndarray:
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _get_aruco_dict(aruco_dict_name: str):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("cv2.aruco not found. Install opencv-contrib-python.")
    dict_map = {
        "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
        "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    }
    if aruco_dict_name not in dict_map:
        raise ValueError(f"Unsupported dict: {aruco_dict_name}. Choose one of: {list(dict_map)}")
    return cv2.aruco.getPredefinedDictionary(dict_map[aruco_dict_name])


def _detect_single_marker_pose(image_bgr, K, dist, marker_length_m, aruco_dict_name: str):
    aruco_dict = _get_aruco_dict(aruco_dict_name)

    if hasattr(cv2.aruco, "ArucoDetector") and hasattr(cv2.aruco, "DetectorParameters"):
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, _ = detector.detectMarkers(image_bgr)
    else:
        params = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(image_bgr, aruco_dict, parameters=params)

    if ids is None or len(ids) == 0:
        raise RuntimeError("No ArUco markers detected. Ensure marker is large & well-lit.")

    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length_m, K, dist)

    rvec = rvecs[0].reshape(3, 1)
    tvec = tvecs[0].reshape(3, 1)
    marker_id = int(ids[0])

    return marker_id, corners[0], rvec, tvec


def _pixel_to_plane_xy(u: float, v: float, K: np.ndarray, dist: np.ndarray, R: np.ndarray, t: np.ndarray):
    pts = np.array([[[u, v]]], dtype=np.float32)
    und = cv2.undistortPoints(pts, K, dist)  # normalized
    x, y = und[0, 0]
    d = np.array([[x], [y], [1.0]], dtype=np.float64)

    n = (R @ np.array([[0.0], [0.0], [1.0]], dtype=np.float64))
    p0 = t

    denom = float(n.T @ d)
    if abs(denom) < 1e-9:
        raise RuntimeError("Ray parallel to plane (unexpected).")

    s = float((n.T @ p0) / denom)
    Xc = s * d

    Xm = R.T @ (Xc - t)
    return float(Xm[0, 0]), float(Xm[1, 0])


def _collect_4_clicks(image_bgr):
    pts: List[Tuple[int, int]] = []
    win = "Click 4 object corners (any order). r=reset, q=quit"
    vis = image_bgr.copy()

    def redraw():
        nonlocal vis
        vis = image_bgr.copy()
        for i, p in enumerate(pts):
            cv2.circle(vis, p, 6, (0, 255, 0), -1)
            cv2.putText(vis, str(i + 1), (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def on_mouse(event, x, y, flags, param):
        nonlocal pts
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((int(x), int(y)))
            redraw()

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    redraw()
    while True:
        cv2.imshow(win, vis)
        key = cv2.waitKey(20) & 0xFF
        if key == ord('q') or key == 27:
            raise SystemExit("User quit.")
        if key == ord('r'):
            pts = []
            redraw()
        if len(pts) == 4:
            break

    cv2.destroyAllWindows()

    pts_np = _order_points_clockwise(np.array(pts, dtype=np.float32))
    return [(int(x), int(y)) for x, y in pts_np]


def _draw_axes(img, K, dist, rvec, tvec, axis_len):
    if hasattr(cv2, "drawFrameAxes"):
        cv2.drawFrameAxes(img, K, dist, rvec, tvec, axis_len)
    else:
        cv2.aruco.drawAxis(img, K, dist, rvec, tvec, axis_len)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--marker_length", type=float, required=True)
    ap.add_argument("--dict", default="DICT_4X4_50")
    ap.add_argument("--out_dir", default="output")
    args = ap.parse_args()

    data = np.load(args.calib)
    K = data["camera_matrix"]
    dist = data["dist_coeffs"]

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Could not read image: {args.image}")

    marker_id, marker_corners, rvec, tvec = _detect_single_marker_pose(
        img, K, dist, args.marker_length, args.dict
    )

    R, _ = cv2.Rodrigues(rvec)
    R = R.astype(np.float64)
    t = tvec.astype(np.float64)
    marker_distance = float(np.linalg.norm(t))

    vis = img.copy()
    cv2.aruco.drawDetectedMarkers(vis, [marker_corners], np.array([[marker_id]], dtype=np.int32))
    _draw_axes(vis, K, dist, rvec, tvec, args.marker_length * 0.75)

    clicks = _collect_4_clicks(vis)

    plane_xy = [_pixel_to_plane_xy(u, v, K, dist, R, t) for (u, v) in clicks]
    tl_xy, tr_xy, br_xy, bl_xy = plane_xy

    width_m = 0.5 * (_euclid2(tl_xy, tr_xy) + _euclid2(bl_xy, br_xy))
    height_m = 0.5 * (_euclid2(tl_xy, bl_xy) + _euclid2(tr_xy, br_xy))

    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.image))[0]
    json_path = os.path.join(args.out_dir, f"{stem}_aruco_measurement.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "image": args.image,
                "marker_id": marker_id,
                "marker_length_m": args.marker_length,
                "marker_tvec_m": [float(t[0, 0]), float(t[1, 0]), float(t[2, 0])],
                "marker_distance_m": marker_distance,
                "width_m": width_m,
                "height_m": height_m,
                "corners_px": clicks,
                "corners_plane_xy_m": [(float(x), float(y)) for (x, y) in plane_xy],
            },
            f,
            indent=2,
        )

    print("\n=== ArUco-plane measurement result ===")
    print(f"||t|| (camera->marker): {marker_distance:.6f} m  (compare to tape-measured distance)")
    print(f"Width:                  {width_m:.6f} m")
    print(f"Height:                 {height_m:.6f} m")
    print(f"Saved JSON:             {json_path}\n")


if __name__ == "__main__":
    main()
