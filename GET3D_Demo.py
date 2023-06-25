import gradio as gr
import json
import torch
import numpy as np
import tempfile
import copy
from torch_utils import training_stats
from torch_utils import custom_ops
from training.inference_utils import generate_a_model,generate_model_interpolation,save_model
import dnnlib

# Load Plugin
from torch_utils.ops import upfirdn2d
from torch_utils.ops import bias_act
from torch_utils.ops import filtered_lrelu
upfirdn2d._init()
bias_act._init()
filtered_lrelu._init()

# Custom Inference Method
def inference(
    run_dir='.',  # Output directory.
    training_set_kwargs={},  # Options for training set.
    G_kwargs={},  # Options for generator network.
    rank=0,  # Rank of the current process in [0, num_gpus[.
    resume_pretrain=None,
    inference_mode='generate',
    **dummy_kawargs
):
    device = torch.device('cuda', rank)
    torch.backends.cudnn.enabled = True

    common_kwargs = dict(
        c_dim=0, img_resolution=training_set_kwargs['resolution'] if 'resolution' in training_set_kwargs else 1024, img_channels=3)
    G_kwargs['device'] = device

    G:torch.nn.Module = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(
        device)  # subclass of torch.nn.Module
    G_ema = copy.deepcopy(G).eval()  # deepcopy can make sure they are correct.
    if resume_pretrain is not None and (rank == 0):
        print('==> resume from pretrained path %s' % (resume_pretrain))
        model_state_dict = torch.load(resume_pretrain, map_location=device)
        G.load_state_dict(model_state_dict['G'], strict=True)
        G_ema.load_state_dict(model_state_dict['G_ema'], strict=True)
        # D.load_state_dict(model_state_dict['D'], strict=True)

    if inference_mode == 'generate':
        print('==> generate a model')
        n_shape = 1
        print(f"Geo Seed: {dummy_kawargs['geo_seed']}, Tex Seed: {dummy_kawargs['tex_seed']}")
        geo_seed_gen = torch.cuda.manual_seed(dummy_kawargs['geo_seed'])
        geo_zs_a:list[torch.Tensor] = torch.randn([n_shape, G.z_dim], generator=geo_seed_gen, device=device).split(1)  # random code for geometry
        tex_seed_a_gen = torch.cuda.manual_seed(dummy_kawargs['tex_seed'])
        tex_zs_a:list[torch.Tensor] = torch.randn([n_shape, G.z_dim], generator=tex_seed_a_gen, device=device).split(1)  # random code for texture
        cs:list[torch.Tensor] = torch.ones(n_shape, device=device).split(1)
        imgs = generate_a_model(G_ema, geo_zs_a, cs, run_dir, 0, 0, tex_zs_a)
    elif inference_mode == 'interpolation' or inference_mode == 'save_interpolation':
        n_shape = 1
        geo_seed_a_gen = torch.cuda.manual_seed(dummy_kawargs['geo_seed_a'])
        geo_zs_a:list[torch.Tensor] = torch.randn([n_shape, G.z_dim], generator=geo_seed_a_gen, device=device).split(1)  # random code for geometry
        tex_seed_a_gen = torch.cuda.manual_seed(dummy_kawargs['tex_seed_a'])
        tex_zs_a:list[torch.Tensor] = torch.randn([n_shape, G.z_dim], generator=tex_seed_a_gen, device=device).split(1)  # random code for texture
        geo_seed_b_gen = torch.cuda.manual_seed(dummy_kawargs['geo_seed_b'])
        geo_zs_b:list[torch.Tensor] = torch.randn([n_shape, G.z_dim], generator=geo_seed_b_gen, device=device).split(1)  # random code for geometry
        tex_seed_b_gen = torch.cuda.manual_seed(dummy_kawargs['tex_seed_b'])
        tex_zs_b:list[torch.Tensor] = torch.randn([n_shape, G.z_dim], generator=tex_seed_b_gen, device=device).split(1)  # random code for texture
        cs:list[torch.Tensor] = torch.ones(n_shape, device=device).split(1)

        geo_zs = torch.cat((*geo_zs_a,*geo_zs_b))
        tex_zs = torch.cat((*tex_zs_a,*tex_zs_b))

        # Interpolate conditioning
        geo_zs = geo_zs if dummy_kawargs['interpo_geo'] is True else geo_zs[0,:].unsqueeze(0).repeat(2,1)
        tex_zs = tex_zs if dummy_kawargs['interpo_tex'] is True else tex_zs[0,:].unsqueeze(0).repeat(2,1)

        if inference_mode == 'save_interpolation':
            print('==> generate and save interpolated model')
            return save_model(G_ema,geo_zs,tex_zs,run_dir)

        print('==> generate model interpolation')
        imgs = generate_model_interpolation( # return images and new latent codes
            G_ema,geo_zs,tex_zs,
            save_dir=run_dir
            )
        
    else:
        print('Noting to generate')
        return

    return images_correction(imgs)

def images_correction(imgs:torch.Tensor):
    imgs = imgs.permute(0,2,3,1) # transpose image to correct shape

    # Image Correction
    lo, hi = [-1, 1]
    imgs:np.ndarray = np.asarray(imgs.cpu(), dtype=np.float32)
    imgs = (imgs - lo) * (255 / (hi - lo))
    imgs = np.rint(imgs).clip(0, 255).astype(np.uint8)
    return imgs

def generate_model(geo_seed,tex_seed=10):
    return inference(rank=rank, geo_seed=int(geo_seed), tex_seed=int(tex_seed), **c)[0]

def generate_interpolation(imgs,geo_seed_a,tex_seed_a,geo_seed_b,tex_seed_b,interpo_geo,interpo_tex):
    imgs = [np.zeros((1,1,1,3),np.uint8)] # Initialize with a black image

    imgs = inference(
        rank=rank,
        geo_seed_a=int(geo_seed_a), tex_seed_a=int(tex_seed_a), geo_seed_b=int(geo_seed_b), tex_seed_b=int(tex_seed_b),
        inference_mode='interpolation',
        interpo_geo = bool(interpo_geo),
        interpo_tex = bool(interpo_tex),
        **c
        )

    # update animate images
    imgs = np.split(imgs,imgs.shape[0])
    
    return imgs,imgs[0][0]

def generate_and_save_model(geo_seed_a,tex_seed_a,geo_seed_b,tex_seed_b,interpo_geo,interpo_tex):
    filepaths = inference(
        rank=rank,
        geo_seed_a=int(geo_seed_a), tex_seed_a=int(tex_seed_a), geo_seed_b=int(geo_seed_b), tex_seed_b=int(tex_seed_b),
        inference_mode='save_interpolation',
        interpo_geo = bool(interpo_geo),
        interpo_tex = bool(interpo_tex),
        **c
        )
    return filepaths

def retrive_a_img(imgs:list[np.ndarray]):
    if len(imgs) > 1:
        return imgs,imgs.pop(0)[0]
    return imgs,imgs[0][0]


# Load Dictionary
rank = 0
c = dnnlib.EasyDict()
with open('options/training_options.json') as f:
    c.update(json.load(f))

# Launch processes.
print('Launching processes...')
torch.multiprocessing.set_start_method('spawn', force=True)
with tempfile.TemporaryDirectory() as temp_dir:
    print(f'Temporary Directory: {temp_dir}')

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if c.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # Gradio Web UI
    with gr.Blocks(title='GET3D Interpolation Demo') as demo:
        result_imgs_var = gr.State([np.zeros((1,1,1,3),np.uint8)]) # list of image tensors
        # result_save_model_var = gr.State([]) # [ OBJ filepath , Texture filepath ]
        with gr.Column(min_width=100):
            gr.Label('GET3D 3D Textured Shape Interpolation Demo',container=False)
            with gr.Row():
                with gr.Box():
                    geo_a_seed_init = 0
                    tex_a_seed_init = 0
                    image_a = gr.Image(generate_model(geo_a_seed_init,tex_a_seed_init),label='Model A',height=256,width=2048)
                    geo_a_seed = gr.Slider(value=geo_a_seed_init,maximum=100,step=1,label='Geometry Variant')
                    tex_a_seed = gr.Slider(value=tex_a_seed_init,maximum=100,step=1,label='Texture Variant')
                    geo_a_seed.release(generate_model,[geo_a_seed,tex_a_seed],image_a)
                    tex_a_seed.release(generate_model,[geo_a_seed,tex_a_seed],image_a)
                with gr.Box():
                    geo_b_seed_init = 50
                    tex_b_seed_init = 10
                    image_b = gr.Image(generate_model(geo_b_seed_init),label='Model B',height=256,width=2048)
                    geo_b_seed = gr.Slider(value=geo_b_seed_init,maximum=100,step=1,label='Geometry Seed')
                    tex_b_seed = gr.Slider(value=tex_b_seed_init,maximum=100,step=1,label='Texture Seed',visible=False) # hide texture seed for fun
                    geo_b_seed.release(generate_model,[geo_b_seed,tex_b_seed],image_b)
                    tex_b_seed.release(generate_model,[geo_b_seed,tex_b_seed],image_b)

            with gr.Box():
                with gr.Column():
                    image_result = gr.Image(label='Interpolation',height=384)
                    toggle_interpo_shape = gr.Checkbox(True,label='Interpolate Shape')
                    toggle_interpo_texture = gr.Checkbox(False,label='Interpolate Texture')
                    btn_interpolate = gr.Button('Interpolate')
                    dep = image_result.change(retrive_a_img,result_imgs_var,[result_imgs_var,image_result],every=0.2)
                    btn_interpolate.click(
                        generate_interpolation,
                        [result_imgs_var,geo_a_seed,tex_a_seed,geo_b_seed,tex_b_seed,toggle_interpo_shape,toggle_interpo_texture],
                        [result_imgs_var,image_result],
                        cancels=[dep]
                        )
                    with gr.Row():
                        btn_save_interpo_model = gr.Button('Save Model')
                        saved_file_download = gr.File(label='Saved Model')
                        btn_save_interpo_model.click(generate_and_save_model,[geo_a_seed,tex_a_seed,geo_b_seed,tex_b_seed,toggle_interpo_shape,toggle_interpo_texture],saved_file_download)

        gr.Markdown(
            """
            - GET3d-WebUI - https://github.com/GaussianGuaicai/GET3D-WebUI
            - GET3D - https://github.com/nv-tlabs/GET3D
            """
            )
demo.queue()
demo.launch(server_name="0.0.0.0",server_port=7870,debug=True,show_error=True)