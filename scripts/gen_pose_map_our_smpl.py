import argparse
import numpy  as np
import torch
from os.path import join
import os
import sys
sys.path.append('../')
import smplx
import trimesh


def render_posmap(v_minimal, faces, uvs, faces_uvs, img_size=32):
    '''
    v_minimal: vertices of the minimally-clothed SMPL body mesh
    faces: faces (triangles) of the minimally-clothed SMPL body mesh
    uvs: the uv coordinate of vertices of the SMPL body model
    faces_uvs: the faces (triangles) on the UV map of the SMPL body model
    '''
    from posmap_generator.lib.renderer.gl.pos_render import PosRender

    # instantiate renderer
    rndr = PosRender(width=img_size, height=img_size)

    # set mesh data on GPU
    rndr.set_mesh(v_minimal, faces, uvs, faces_uvs)

    # render
    rndr.display()

    # retrieve the rendered buffer
    uv_pos = rndr.get_color(0)
    uv_mask = uv_pos[:, :, 3]
    uv_pos = uv_pos[:, :, :3]

    uv_mask = uv_mask.reshape(-1)
    uv_pos = uv_pos.reshape(-1, 3)

    rendered_pos = uv_pos[uv_mask != 0.0]

    uv_pos = uv_pos.reshape(img_size, img_size, 3)

    # get face_id (triangle_id) per pixel
    face_id = uv_mask[uv_mask != 0].astype(np.int32) - 1

    assert len(face_id) == len(rendered_pos)

    return uv_pos, uv_mask, face_id


def save_obj(data_path, name):
    smpl_data = torch.load( data_path + '/' + name)
    
    frame_num = smpl_data['body_pose'].shape[0]
    print('frame_num', frame_num)
    start_pose = 0

    smpl_model = smplx.SMPL(model_path ='../assets/smpl_files/smpl',  batch_size = 1)

    norm_obj_dir = os.path.join(data_path, 'norm_obj')
    os.makedirs(norm_obj_dir, exist_ok=True)

    for pose_idx in range(start_pose, frame_num + start_pose):
        image_key = str(pose_idx).zfill(8)

        cano_smpl = smpl_model.forward(betas=smpl_data['beta'],
                                global_orient=smpl_data['body_pose'][pose_idx, :3][None],
                                transl = smpl_data['trans'][pose_idx, :][None],
                                # global_orient=cpose_param[:, :3],
                                body_pose=smpl_data['body_pose'][pose_idx, 3:][None],
                                )
        norm_vertices = cano_smpl.vertices.detach().cpu().numpy().squeeze()
        mesh = trimesh.Trimesh(norm_vertices, smpl_model.faces, process=False)
        mesh.export('%s/%s.obj' % (norm_obj_dir, str(image_key)))



def save_npz(data_path, res=128):
    from posmap_generator.lib.renderer.mesh import load_obj_mesh
    verts, faces, uvs, faces_uvs = load_obj_mesh(uv_template_fn, with_texture=True)
    start_obj_num = 0

    norm_obj_dir = os.path.join(data_path, 'norm_obj')
    inp_map_dir = os.path.join(data_path, 'inp_map')
    os.makedirs(inp_map_dir, exist_ok=True)
    # os.makedirs(query_map_dir, exist_ok=True)

    norm_obj_length = len(os.listdir(norm_obj_dir))
    result = {}
    for indx in range(start_obj_num, start_obj_num+norm_obj_length):
        image_key = str(indx).zfill(8)
        body_mesh = trimesh.load('%s/%s.obj'%(norm_obj_dir, image_key), process=False)

        if res==128:
            posmap128, _, _ = render_posmap(body_mesh.vertices, body_mesh.faces, uvs, faces_uvs, img_size=128)
            result['posmap128'] = posmap128   
        elif res == 256:
        
            posmap256, _, _ = render_posmap(body_mesh.vertices, body_mesh.faces, uvs, faces_uvs, img_size=256)
            result['posmap256'] = posmap256

        else:
            posmap512, _, _ = render_posmap(body_mesh.vertices, body_mesh.faces, uvs, faces_uvs, img_size=512)
            result['posmap512'] = posmap512

        save_fn = join(inp_map_dir, 'inp_posemap_%s_%s.npz'% (str(res), image_key))
        np.savez(save_fn, **result)


if __name__ == '__main__':
    smpl_parm_path = 'path to you data folder'
    parms_name = 'smpl_parms.pth'

    save_obj(smpl_parm_path, parms_name)

#    # save step by step
    #print('saving pose_map 128 ...')
    #save_npz(smpl_parm_path, 128)

