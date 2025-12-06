from pathlib import Path
import argparse
import json
import numpy  as np
import imageio.v2 as imageio
import torch
from os.path import join
import os
import sys
sys.path.append('../')
import smplx
import trimesh

# ensure PyOpenGL uses EGL (headless)
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
from posmap_generator.lib.renderer.mesh import load_obj_mesh

def render_posmap(v_minimal, faces, uvs, faces_uvs, img_size=32):
    '''
    v_minimal: vertices of the minimally-clothed SMPL body mesh
    faces: faces (triangles) of the minimally-clothed SMPL body mesh
    uvs: the uv coordinate of vertices of the SMPL body model
    faces_uvs: the faces (triangles) on the UV map of the SMPL body model
    '''
    from posmap_generator.lib.renderer.egl.egl_pos_render import PosRender

    # instantiate renderer
    rndr = PosRender(width=img_size, height=img_size)

    # set mesh data on GPU
    rndr.set_mesh(v_minimal, faces, uvs, faces_uvs)

    # render
    rndr.draw()

    # retrieve the rendered buffer
    uv_pos = rndr.get_color()
    uv_mask = uv_pos[:, :, 3]
    uv_pos = uv_pos[:, :, :3]

    uv_mask_flat = uv_mask.reshape(-1)
    uv_pos_flat = uv_pos.reshape(-1, 3)

    rendered_pos = uv_pos_flat[uv_mask_flat != 0.0]

    uv_pos = uv_pos_flat.reshape(img_size, img_size, 3)
    uv_mask_img = uv_mask_flat.reshape(img_size, img_size)

    # get face_id (triangle_id) per pixel
    face_id = uv_mask_flat[uv_mask_flat != 0].astype(np.int32) - 1

    assert len(face_id) == len(rendered_pos)

    return uv_pos, uv_mask_img, face_id


def save_human_meshes(root_save_dir : Path):

    motion_dir = root_save_dir / "motion"
    track_ids = sorted(os.listdir(motion_dir))

    smplx_model_path = "/home/cizinsky/body_models/smplx"
    smpl_model = smplx.SMPLX(model_path=smplx_model_path,
                             batch_size=1,
                             use_pca=False)  # hands provided as full 45D axis-angle

    def load_smplx_frame(json_path: Path):
        with open(json_path, "r") as f:
            data = json.load(f)

        # SMPL-X expects axis-angle vectors flattened per component
        body_pose = torch.tensor(data.get("body_pose", [[0, 0, 0]] * 21), dtype=torch.float32).reshape(1, -1)
        left_hand = torch.tensor(data.get("lhand_pose", [[0, 0, 0]] * 15), dtype=torch.float32).reshape(1, -1)
        right_hand = torch.tensor(data.get("rhand_pose", [[0, 0, 0]] * 15), dtype=torch.float32).reshape(1, -1)

        return {
            "betas": torch.tensor(data["betas"], dtype=torch.float32).unsqueeze(0),
            "global_orient": torch.tensor(data["root_pose"], dtype=torch.float32).unsqueeze(0),
            "body_pose": body_pose,
            "left_hand_pose": left_hand,
            "right_hand_pose": right_hand,
            "jaw_pose": torch.tensor(data.get("jaw_pose", [0, 0, 0]), dtype=torch.float32).unsqueeze(0),
            "leye_pose": torch.tensor(data.get("leye_pose", [0, 0, 0]), dtype=torch.float32).unsqueeze(0),
            "reye_pose": torch.tensor(data.get("reye_pose", [0, 0, 0]), dtype=torch.float32).unsqueeze(0),
            "transl": torch.tensor(data.get("trans", [0, 0, 0]), dtype=torch.float32).unsqueeze(0),
        }

    for track_id in track_ids:
        
        track_id_smpl_data_dir =  motion_dir / track_id / "smplx_params"
        track_id_norm_obj_dir = motion_dir / track_id / "norm_obj"
        track_id_norm_obj_dir.mkdir(parents=True, exist_ok=True)

        all_smplx_params_paths = sorted(Path(track_id_smpl_data_dir).glob("*.json"))

        for smplx_path in all_smplx_params_paths:
            smplx_params = load_smplx_frame(smplx_path)

            with torch.no_grad():
                cano_smpl = smpl_model.forward(**smplx_params)

            norm_vertices = cano_smpl.vertices.detach().cpu().numpy().squeeze()
            mesh = trimesh.Trimesh(norm_vertices, smpl_model.faces, process=False)
            mesh.export(track_id_norm_obj_dir / f"{smplx_path.stem}.obj")


def save_npz(root_save_dir: Path, res: int = 128, save_png: bool = False):

    path_to_smplx_template = Path("/home/cizinsky/GaussianAvatar/assets/template_mesh_smplx_uv.obj")
    verts, faces, uvs, faces_uvs = load_obj_mesh(str(path_to_smplx_template), with_texture=True)

    motion_dir = root_save_dir / "motion"
    track_ids = sorted(os.listdir(motion_dir))

    for track_id in track_ids:
        norm_obj_dir = motion_dir / track_id / "norm_obj"
        if not norm_obj_dir.exists():
            continue

        inp_map_dir = motion_dir / track_id / "inp_map"
        inp_map_dir.mkdir(parents=True, exist_ok=True)
        inp_png_dir = motion_dir / track_id / "inp_map_png"
        if save_png:
            inp_png_dir.mkdir(parents=True, exist_ok=True)

        obj_paths = sorted(norm_obj_dir.glob("*.obj"))

        for obj_path in obj_paths:
            body_mesh = trimesh.load(str(obj_path), process=False)

            posmap, uv_mask, _ = render_posmap(body_mesh.vertices, body_mesh.faces, uvs, faces_uvs, img_size=res)
            result = {f"posmap{res}": posmap}

            save_fn = inp_map_dir / f"inp_posemap_{res}_{obj_path.stem}.npz"
            np.savez(save_fn, **result)

            if save_png:
                mask_bool = uv_mask > 0
                vis = np.zeros_like(posmap)
                if mask_bool.any():
                    valid = posmap[mask_bool]
                    vmin = valid.min(axis=0)
                    vmax = valid.max(axis=0)
                    scale = np.maximum(vmax - vmin, 1e-8)
                    vis = (np.clip((posmap - vmin) / scale, 0, 1) * 255).astype(np.uint8)
                png_path = inp_png_dir / f"inp_posemap_{res}_{obj_path.stem}.png"
                imageio.imwrite(png_path, vis)


if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument('--root-save-dir', type=Path, required=True, help='Path to the root save directory')
    args.add_argument('--res', type=int, default=128, choices=[128, 256, 512], help='Resolution of the pose map')
    args.add_argument('--save-png', action='store_true', help='Also save a visualized PNG of each pose map')
    args = args.parse_args()

    save_human_meshes(args.root_save_dir)
    save_npz(args.root_save_dir, res=args.res, save_png=args.save_png)
