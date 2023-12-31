# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
'''
Utily functions for the inference
'''
import torch
import numpy as np
import os
import PIL.Image
from training.utils.utils_3d import save_obj, savemeshtes2
import imageio
import cv2
from tqdm import tqdm
from training.networks_get3d import GeneratorDMTETMesh,DMTETSynthesisNetwork


def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    gw = _N // gh
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    assert C in [1, 3]
    if not fname is None:
        if C == 1:
            PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
        if C == 3:
            PIL.Image.fromarray(img, 'RGB').save(fname)
    return img


def save_3d_shape(mesh_v_list, mesh_f_list, root, idx):
    n_mesh = len(mesh_f_list)
    mesh_dir = os.path.join(root, 'mesh_pred')
    os.makedirs(mesh_dir, exist_ok=True)
    for i_mesh in range(n_mesh):
        mesh_v = mesh_v_list[i_mesh]
        mesh_f = mesh_f_list[i_mesh]
        mesh_name = os.path.join(mesh_dir, '%07d_%02d.obj' % (idx, i_mesh))
        save_obj(mesh_v, mesh_f, mesh_name)


def gen_swap(ws_geo_list, ws_tex_list, camera, generator:GeneratorDMTETMesh, save_path, gen_mesh=False, ):
    '''
    With two list of latent code, generate a matrix of results, N_geo x N_tex
    :param ws_geo_list: the list of geometry latent code
    :param ws_tex_list: the list of texture latent code
    :param camera:  camera to render the generated mesh
    :param generator: GET3D_Generator
    :param save_path: path to save results
    :param gen_mesh: whether we generate textured mesh
    :return:
    '''
    img_list = []
    os.makedirs(save_path, exist_ok=True)
    with torch.no_grad():
        for i_geo, ws_geo in enumerate(ws_geo_list):
            for i_tex, ws_tex in enumerate(ws_tex_list):
                img, mask, sdf, deformation, v_deformed, mesh_v, mesh_f, gen_camera, img_wo_light, mask_pyramid, tex_hard_mask, \
                sdf_reg_loss, render_return_value = generator.synthesis.generate(
                    ws_tex.unsqueeze(dim=0), update_emas=None, camera=camera,
                    update_geo=None, ws_geo=ws_geo.unsqueeze(dim=0),
                )
                img_list.append(img[:, :3].data.cpu().numpy())
                if gen_mesh:
                    generated_mesh = generator.synthesis.extract_3d_shape(ws_tex.unsqueeze(dim=0), ws_geo.unsqueeze(dim=0))
                    for mesh_v, mesh_f, all_uvs, all_mesh_tex_idx, tex_map in zip(*generated_mesh):
                        savemeshtes2(
                            mesh_v.data.cpu().numpy(),
                            all_uvs.data.cpu().numpy(),
                            mesh_f.data.cpu().numpy(),
                            all_mesh_tex_idx.data.cpu().numpy(),
                            os.path.join(save_path, '%02d_%02d.obj' % (i_geo, i_tex))
                        )
                        lo, hi = (-1, 1)
                        img = np.asarray(tex_map.permute(1, 2, 0).data.cpu().numpy(), dtype=np.float32)
                        img = (img - lo) * (255 / (hi - lo))
                        img = img.clip(0, 255)
                        mask = np.sum(img.astype(np.float16), axis=-1, keepdims=True)
                        mask = (mask <= 3.0).astype(np.float16)
                        kernel = np.ones((3, 3), 'uint8')
                        dilate_img = cv2.dilate(img, kernel, iterations=1)
                        img = img * (1 - mask) + dilate_img * mask
                        img = img.clip(0, 255).astype(np.uint8)
                        PIL.Image.fromarray(np.ascontiguousarray(img[::-1, :, :]), 'RGB').save(
                            os.path.join(save_path, '%02d_%02d.png' % (i_geo, i_tex)))
    img_list = np.concatenate(img_list, axis=0)
    img = save_image_grid(img_list, os.path.join(save_path, 'inter_img.jpg'), drange=[-1, 1], grid_size=[ws_tex_list.shape[0], ws_geo_list.shape[0]])
    return img


def save_visualization_for_interpolation(
        generator:GeneratorDMTETMesh, num_sam=10, c_to_compute_w_avg=None, save_dir=None, gen_mesh=False):
    '''
    Interpolate between two latent code and generate a swap between them
    :param generator: GET3D generator
    :param num_sam: number of samples we hope to generate
    :param c_to_compute_w_avg: None is default
    :param save_dir: path to save
    :param gen_mesh: whether we want to generate 3D textured mesh
    :return:
    '''
    with torch.no_grad():
        generator.update_w_avg(c_to_compute_w_avg)
        geo_codes = torch.randn(num_sam, generator.z_dim, device=generator.device)
        tex_codes = torch.randn(num_sam, generator.z_dim, device=generator.device)
        ws_geo = generator.mapping_geo(geo_codes, None, truncation_psi=0.7)
        ws_tex = generator.mapping(tex_codes, None, truncation_psi=0.7)
        print(f'Ws Geo Latent Code:{ws_geo.shape}')
        print(f'Ws Tex Latent Code:{ws_tex.shape}')
        camera_list = [generator.synthesis.generate_rotate_camera_list(n_batch=num_sam)[4]]

        select_geo_codes = np.arange(4)  # You can change to other selected shapes
        select_tex_codes = np.arange(4)
        for i in range(len(select_geo_codes) - 1):
            ws_geo_a = ws_geo[select_geo_codes[i]].unsqueeze(dim=0)
            ws_geo_b = ws_geo[select_geo_codes[i + 1]].unsqueeze(dim=0)
            ws_tex_a = ws_tex[select_tex_codes[i]].unsqueeze(dim=0)
            ws_tex_b = ws_tex[select_tex_codes[i + 1]].unsqueeze(dim=0)
            new_ws_geo = []
            new_ws_tex = []
            n_interpolate = 3
            for _i in range(n_interpolate):
                w = float(_i + 1) / n_interpolate
                w = 1 - w
                new_ws_geo.append(ws_geo_a * w + ws_geo_b * (1 - w))
                new_ws_tex.append(ws_tex_a * w + ws_tex_b * (1 - w))
            new_ws_tex = torch.cat(new_ws_tex, dim=0)
            new_ws_geo = torch.cat(new_ws_geo, dim=0)
            # print(f'New WS Geo Latent Code:{new_ws_geo.shape}')
            # print(f'New WS Tex Latent Code:{new_ws_tex.shape}')
            save_path = os.path.join(save_dir, 'interpolate_%02d' % (i))
            os.makedirs(save_path, exist_ok=True)
            gen_swap(
                new_ws_geo, new_ws_tex, camera_list[0], generator,
                save_path=save_path, gen_mesh=gen_mesh
            )


def save_visualization(
        G_ema, grid_z, grid_c, run_dir, cur_nimg, grid_size, cur_tick,
        image_snapshot_ticks=50,
        save_gif_name=None,
        save_all=True,
        grid_tex_z=None,
):
    '''
    Save visualization during training
    :param G_ema: GET3D generator
    :param grid_z: latent code for geometry latent code
    :param grid_c: None
    :param run_dir: path to save images
    :param cur_nimg: current k images during training
    :param grid_size: size of the image
    :param cur_tick: current kicks for training
    :param image_snapshot_ticks: current snapshot ticks
    :param save_gif_name: the name to save if we want to export gif
    :param save_all:  whether we want to save gif or not
    :param grid_tex_z: the latent code for texture geenration
    :return:
    '''
    with torch.no_grad():
        G_ema.update_w_avg()
        camera_list = G_ema.synthesis.generate_rotate_camera_list(n_batch=grid_z[0].shape[0])
        camera_img_list = []
        if not save_all:
            camera_list = [camera_list[4]]  # we only save one camera for this
        if grid_tex_z is None:
            grid_tex_z = grid_z
        for i_camera, camera in enumerate(camera_list):
            images_list = []
            mesh_v_list = []
            mesh_f_list = []
            for z, geo_z, c in zip(grid_tex_z, grid_z, grid_c):
                img, mask, sdf, deformation, v_deformed, mesh_v, mesh_f, gen_camera, img_wo_light, tex_hard_mask = G_ema.generate_3d(
                    z=z, geo_z=geo_z, c=c, noise_mode='const',
                    generate_no_light=True, truncation_psi=0.7, camera=camera)
                rgb_img = img[:, :3]
                save_img = torch.cat([rgb_img, mask.permute(0, 3, 1, 2).expand(-1, 3, -1, -1)], dim=-1).detach()
                images_list.append(save_img.cpu().numpy())
                mesh_v_list.extend([v.data.cpu().numpy() for v in mesh_v])
                mesh_f_list.extend([f.data.cpu().numpy() for f in mesh_f])
            images = np.concatenate(images_list, axis=0)
            if save_gif_name is None:
                save_file_name = 'fakes'
            else:
                save_file_name = 'fakes_%s' % (save_gif_name.split('.')[0])
            if save_all:
                img = save_image_grid(
                    images, None,
                    drange=[-1, 1], grid_size=grid_size)
            else:
                img = save_image_grid(
                    images, os.path.join(
                        run_dir,
                        f'{save_file_name}_{cur_nimg // 1000:06d}_{i_camera:02d}.png'),
                    drange=[-1, 1], grid_size=grid_size)
            camera_img_list.append(img)
        if save_gif_name is None:
            save_gif_name = f'fakes_{cur_nimg // 1000:06d}.gif'
        if save_all:
            imageio.mimsave(os.path.join(run_dir, save_gif_name), camera_img_list)
        n_shape = 10  # we only save 10 shapes to check performance
        if cur_tick % min((image_snapshot_ticks * 20), 100) == 0:
            save_3d_shape(mesh_v_list[:n_shape], mesh_f_list[:n_shape], run_dir, cur_nimg // 100)


def save_textured_mesh_for_inference(
        G_ema, grid_z, grid_c, run_dir, save_mesh_dir=None,
        c_to_compute_w_avg=None, grid_tex_z=None, use_style_mixing=False):
    '''
    Generate texture mesh for generation
    :param G_ema: GET3D generator
    :param grid_z: a grid of latent code for geometry
    :param grid_c: None
    :param run_dir: save path
    :param save_mesh_dir: path to save generated mesh
    :param c_to_compute_w_avg: None
    :param grid_tex_z: latent code for texture
    :param use_style_mixing: whether we use style mixing or not
    :return:
    '''
    with torch.no_grad():
        G_ema.update_w_avg(c_to_compute_w_avg)
        save_mesh_idx = 0
        mesh_dir = os.path.join(run_dir, save_mesh_dir)
        os.makedirs(mesh_dir, exist_ok=True)
        for idx in range(len(grid_z)):
            geo_z = grid_z[idx]
            if grid_tex_z is None:
                tex_z = grid_z[idx]
            else:
                tex_z = grid_tex_z[idx]
            generated_mesh = G_ema.generate_3d_mesh(
                geo_z=geo_z, tex_z=tex_z, c=None, truncation_psi=0.7,
                use_style_mixing=use_style_mixing)
            for mesh_v, mesh_f, all_uvs, all_mesh_tex_idx, tex_map in zip(*generated_mesh):
                savemeshtes2(
                    mesh_v.data.cpu().numpy(),
                    all_uvs.data.cpu().numpy(),
                    mesh_f.data.cpu().numpy(),
                    all_mesh_tex_idx.data.cpu().numpy(),
                    os.path.join(mesh_dir, '%07d.obj' % (save_mesh_idx))
                )
                lo, hi = (-1, 1)
                img = np.asarray(tex_map.permute(1, 2, 0).data.cpu().numpy(), dtype=np.float32)
                img = (img - lo) * (255 / (hi - lo))
                img = img.clip(0, 255)
                mask = np.sum(img.astype(np.float16), axis=-1, keepdims=True)
                mask = (mask <= 3.0).astype(np.float16)
                kernel = np.ones((3, 3), 'uint8')
                dilate_img = cv2.dilate(img, kernel, iterations=1)
                img = img * (1 - mask) + dilate_img * mask
                img = img.clip(0, 255).astype(np.uint8)
                PIL.Image.fromarray(np.ascontiguousarray(img[::-1, :, :]), 'RGB').save(
                    os.path.join(mesh_dir, '%07d.png' % (save_mesh_idx)))
                save_mesh_idx += 1


def save_geo_for_inference(G_ema, run_dir):
    '''
    Generate the 3D objs (without texture) for generation
    :param G_ema: GET3D Generation
    :param run_dir: save path
    :return:
    '''
    import kaolin as kal
    def normalize_and_sample_points(mesh_v, mesh_f, kal, n_sample, normalized_scale=1.0):
        vertices = mesh_v.cuda()
        scale = (vertices.max(dim=0)[0] - vertices.min(dim=0)[0]).max()
        mesh_v1 = vertices / scale * normalized_scale
        mesh_f1 = mesh_f.cuda()
        points, _ = kal.ops.mesh.sample_points(mesh_v1.unsqueeze(dim=0), mesh_f1, n_sample)
        return points

    with torch.no_grad():
        use_style_mixing = True
        truncation_phi = 1.0
        mesh_dir = os.path.join(run_dir, 'gen_geo_for_eval_phi_%.2f' % (truncation_phi))
        surface_point_dir = os.path.join(run_dir, 'gen_geo_surface_points_for_eval_phi_%.2f' % (truncation_phi))
        os.makedirs(mesh_dir, exist_ok=True)
        os.makedirs(surface_point_dir, exist_ok=True)
        n_gen = 1500 * 5  # We generate 5x of test set here
        i_mesh = 0
        for i_gen in tqdm(range(n_gen)):
            geo_z = torch.randn(1, G_ema.z_dim, device=G_ema.device)
            generated_mesh = G_ema.generate_3d_mesh(
                geo_z=geo_z, tex_z=None, c=None, truncation_psi=truncation_phi,
                with_texture=False, use_style_mixing=use_style_mixing)
            for mesh_v, mesh_f in zip(*generated_mesh):
                if mesh_v.shape[0] == 0: continue
                save_obj(mesh_v.data.cpu().numpy(), mesh_f.data.cpu().numpy(), os.path.join(mesh_dir, '%07d.obj' % (i_mesh)))
                points = normalize_and_sample_points(mesh_v, mesh_f, kal, n_sample=2048, normalized_scale=1.0)
                np.savez(os.path.join(surface_point_dir, '%07d.npz' % (i_mesh)), pcd=points.data.cpu().numpy())
                i_mesh += 1

def generate_a_model(
    G_ema, geo_zs, cs, run_dir, cur_nimg, cur_tick,
    # image_snapshot_ticks=50,
    tex_zs=None
):
    '''
    Save visualization during training
    :param G_ema: GET3D generator
    :param geo_zs: list of latent code for geometry latent code
    :param cs: None
    :param run_dir: path to save images
    :param cur_nimg: current k images during training
    :param cur_tick: current kicks for training
    :param image_snapshot_ticks: current snapshot ticks
    :param tex_zs: list of latent code for texture geenration
    :return: a image
    '''
    with torch.no_grad():
        G_ema.update_w_avg()
        camera_list = G_ema.synthesis.generate_rotate_camera_list(n_batch=1)
        camera_img_list:list[torch.Tensor] = []
        camera_list = [camera_list[4]]  # we only save one camera for this
        if tex_zs is None:
            tex_zs = geo_zs
        for i_camera, camera in enumerate(camera_list):
            images_list = []
            mesh_v_list = []
            mesh_f_list = []
            print(f'Generate Model at Camera {i_camera}')
            for tex_z, geo_z, c in zip(tex_zs, geo_zs, cs):
                img, mask, sdf, deformation, v_deformed, mesh_v, mesh_f, gen_camera, img_wo_light, tex_hard_mask = G_ema.generate_3d(
                    z=tex_z, geo_z=geo_z, c=c, noise_mode='const',
                    generate_no_light=True, truncation_psi=0.7, camera=camera)
                rgb_img = img[:, :3]
                save_img = torch.cat([rgb_img, mask.permute(0, 3, 1, 2).expand(-1, 3, -1, -1)], dim=-1).detach()
                images_list.append(save_img.cpu().numpy())
                mesh_v_list.extend([v.data.cpu().numpy() for v in mesh_v])
                mesh_f_list.extend([f.data.cpu().numpy() for f in mesh_f])
            camera_img_list.append(img)

        # n_shape = 10  # we only save 10 shapes to check performance
        # save_3d_shape(mesh_v_list[:n_shape], mesh_f_list[:n_shape], run_dir, cur_nimg // 100)

        return camera_img_list[0][:, :3]

def generate_model_interpolation(
        generator:GeneratorDMTETMesh, geo_zs:torch.Tensor, tex_zs:torch.Tensor, c_to_compute_w_avg=None, save_dir=None, gen_mesh=False, geo_interpolate=True, tex_interpolate=False):
    '''
    Interpolate between two latent code and generate a swap between them
    :param generator: GET3D generator
    :param num_sam: number of samples we hope to generate
    :param c_to_compute_w_avg: None is default
    :param save_dir: path to save
    :param gen_mesh: whether we want to generate 3D textured mesh
    :return: Sequence of interpolate transition images
    '''
    with torch.no_grad():
        generator.update_w_avg(c_to_compute_w_avg)
        geo_codes = geo_zs
        tex_codes = tex_zs
        ws_geo = generator.mapping_geo(geo_codes, None, truncation_psi=0.7)
        ws_tex = generator.mapping(tex_codes, None, truncation_psi=0.7)
        camera_list = [generator.synthesis.generate_rotate_camera_list(n_batch=1)[4]]

        ws_geo_a = ws_geo[0].unsqueeze(dim=0)
        ws_geo_b = ws_geo[1].unsqueeze(dim=0)
        ws_tex_a = ws_tex[0].unsqueeze(dim=0)
        ws_tex_b = ws_tex[1].unsqueeze(dim=0)
        new_ws_geo = []
        new_ws_tex = []
        n_interpolate = 8
        for _i in range(n_interpolate):
            w = float(_i + 1) / n_interpolate
            w = 1 - w
            new_ws_geo.append(ws_geo_a * w + ws_geo_b * (1 - w))
            new_ws_tex.append(ws_tex_a * w + ws_tex_b * (1 - w))
        new_ws_tex = torch.cat(new_ws_tex, dim=0)
        new_ws_geo = torch.cat(new_ws_geo, dim=0)
        save_path = os.path.join(save_dir, 'interpolation')
        os.makedirs(save_path, exist_ok=True)

        # print(new_ws_geo.shape)

        imgs = interpolate_seq(
            n_interpolate,
            new_ws_geo, new_ws_tex, camera_list[0], generator,
            save_path=save_path, gen_mesh=gen_mesh
        )
        return imgs


def interpolate_seq(frame_count:int,ws_geo_list, ws_tex_list, camera, generator:GeneratorDMTETMesh, save_path:str, gen_mesh=False):
    '''
    With two list of latent code, generate interpolation between two latent code in list
    :param ws_geo_list: the list of geometry latent code
    :param ws_tex_list: the list of texture latent code
    :param camera:  camera to render the generated mesh
    :param generator: GET3D_Generator
    :param save_path: path to save results
    :param gen_mesh: whether we generate textured mesh
    :return:
    '''
    imgs = []
    os.makedirs(save_path, exist_ok=True)
    with torch.no_grad():
        for i in range(frame_count):
            img, mask, sdf, deformation, v_deformed, mesh_v, mesh_f, gen_camera, img_wo_light, mask_pyramid, tex_hard_mask, \
                sdf_reg_loss, render_return_value = generator.synthesis.generate(
                    ws_tex_list[i].unsqueeze(dim=0), update_emas=None, camera=camera,
                    update_geo=None, ws_geo=ws_geo_list[i].unsqueeze(dim=0),
                )
            imgs.append(img[:, :3])
            if gen_mesh:
                selection = -1 # which mapped latent codes to save its generation result
                generated_mesh = generator.synthesis.extract_3d_shape(ws_tex_list[selection].unsqueeze(dim=0), ws_geo_list[selection].unsqueeze(dim=0))
                for mesh_v, mesh_f, all_uvs, all_mesh_tex_idx, tex_map in zip(*generated_mesh):
                    savemeshtes2(
                        mesh_v.data.cpu().numpy(),
                        all_uvs.data.cpu().numpy(),
                        mesh_f.data.cpu().numpy(),
                        all_mesh_tex_idx.data.cpu().numpy(),
                        os.path.join(save_path, 'interpolation_mesh.obj')
                    )
                    lo, hi = (-1, 1)
                    img = np.asarray(tex_map.permute(1, 2, 0).data.cpu().numpy(), dtype=np.float32)
                    img = (img - lo) * (255 / (hi - lo))
                    img = img.clip(0, 255)
                    mask = np.sum(img.astype(np.float16), axis=-1, keepdims=True)
                    mask = (mask <= 3.0).astype(np.float16)
                    kernel = np.ones((3, 3), 'uint8')
                    dilate_img = cv2.dilate(img, kernel, iterations=1)
                    img = img * (1 - mask) + dilate_img * mask
                    img = img.clip(0, 255).astype(np.uint8)
                    PIL.Image.fromarray(np.ascontiguousarray(img[::-1, :, :]), 'RGB').save(
                        os.path.join(save_path, 'interpolation_texture.png'))
    imgs = torch.cat(imgs)
    return imgs


def save_model(generator:GeneratorDMTETMesh, geo_z:torch.Tensor, tex_z:torch.Tensor,save_path:str):
    ws_geo = generator.mapping_geo(geo_z[-1].unsqueeze(0), None, truncation_psi=0.7)
    ws_tex = generator.mapping(tex_z[-1].unsqueeze(0), None, truncation_psi=0.7)

    import random
    hash = random.getrandbits(128)
    output_filepaths = [
        os.path.join(save_path, f'interpolation_mesh_{hash}.obj'),
        os.path.join(save_path, f'interpolation_texture_{hash}.png'),
        ]

    generated_mesh = generator.synthesis.extract_3d_shape(ws_tex,ws_geo)
    for mesh_v, mesh_f, all_uvs, all_mesh_tex_idx, tex_map in zip(*generated_mesh):
        savemeshtes2(
            mesh_v.data.cpu().numpy(),
            all_uvs.data.cpu().numpy(),
            mesh_f.data.cpu().numpy(),
            all_mesh_tex_idx.data.cpu().numpy(),
            output_filepaths[0]
        )
        lo, hi = (-1, 1)
        img = np.asarray(tex_map.permute(1, 2, 0).data.cpu().numpy(), dtype=np.float32)
        img = (img - lo) * (255 / (hi - lo))
        img = img.clip(0, 255)
        mask = np.sum(img.astype(np.float16), axis=-1, keepdims=True)
        mask = (mask <= 3.0).astype(np.float16)
        kernel = np.ones((3, 3), 'uint8')
        dilate_img = cv2.dilate(img, kernel, iterations=1)
        img = img * (1 - mask) + dilate_img * mask
        img = img.clip(0, 255).astype(np.uint8)
        PIL.Image.fromarray(np.ascontiguousarray(img[::-1, :, :]), 'RGB').save(output_filepaths[1])
    return output_filepaths