from pathlib import Path
import argparse
import numpy as np
import imageio.v2 as imageio
import os


def compute_intrinsics_from_image(img_path: Path, fov_deg: float = 60.0) -> np.ndarray:
    """Compute 3x3 intrinsic matrix assuming square pixels and centered principal point."""
    img = imageio.imread(img_path)
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    # Use the larger dimension as img_size proxy (matches provided formula)
    img_size = max(h, w)
    focal = img_size / (2.0 * np.tan(np.radians(fov_deg) / 2.0))

    intrinsic = np.array([[focal, 0.0, cx],
                          [0.0, focal, cy],
                          [0.0, 0.0, 1.0]], dtype=np.float32)
    return intrinsic


def export_cameras(root_save_dir: Path):
    frames_dir = root_save_dir / "frames"
    png_frames = sorted([p for p in frames_dir.glob("*.png")])
    if not png_frames:
        raise FileNotFoundError(f"No .png frames found in {frames_dir}")

    intrinsic = compute_intrinsics_from_image(png_frames[0])
    motion_dir = root_save_dir / "motion"
    track_ids = sorted(os.listdir(motion_dir))

    for track_id in track_ids:
        track_dir = motion_dir / track_id
        smpl_dir = track_dir / "smplx_params"
        frame_files = sorted(smpl_dir.glob("*.json"))
        num_frames = len(frame_files)
        if num_frames == 0:
            continue

        extrinsic = np.eye(4, dtype=np.float32)
        extrinsics = np.repeat(extrinsic[None, ...], num_frames, axis=0)
        intrinsics = np.repeat(intrinsic[None, ...], num_frames, axis=0)

        out_path = track_dir / "cameras.npz"
        np.savez(out_path, extrinsics=extrinsics, intrinsics=intrinsics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-save-dir", type=Path, required=True, help="Path to root save directory")
    args = parser.parse_args()

    export_cameras(args.root_save_dir)
