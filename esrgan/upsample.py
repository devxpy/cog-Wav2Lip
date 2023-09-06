import cv2
import glob
import os
import sys
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
import numpy as np
import torch
from gfpgan import GFPGANer
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact
from basicsr.utils import imwrite, img2tensor, tensor2img
from torchvision.transforms.functional import normalize
from basicsr.utils.registry import ARCH_REGISTRY

import warnings
warnings.filterwarnings("ignore")

def load_sr(model_path, device, face):
    if not face=='codeformer':
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4) #alter to match dims as needed
        netscale = 4
        model_path = os.path.normpath(model_path)
        if not os.path.exists(model_path):
            model_path = load_file_from_url(
                url='https://github.com/anothermartz/Easy-Wav2Lip/releases/download/Prerequesits/4x_BigFace_v3_Clear.pth',
                model_dir='weights', progress=True, file_name=None)
        upsampler = RealESRGANer(
            scale=netscale,
            model_path=model_path,
            dni_weight=None,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True,
            gpu_id=0)
        if face==None:
            run_params=upsampler
        else:
            gfp = GFPGANer(
                model_path='https://github.com/anothermartz/Easy-Wav2Lip/releases/download/Prerequesits/GFPGANv1.4.pth',
                upscale=1,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=None)
            run_params=gfp
    else:
        run_params = ARCH_REGISTRY.get('CodeFormer')(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                                              connect_list=['32', '64', '128', '256']).to(device)
        ckpt_path = load_file_from_url(url='https://github.com/anothermartz/Easy-Wav2Lip/releases/download/Prerequesits/codeformer.pth',
                                       model_dir='weights/CodeFormer', progress=True, file_name=None)
        checkpoint = torch.load(ckpt_path)['params_ema']
        run_params.load_state_dict(checkpoint)
        run_params.eval()
    return run_params


def upscale(image, face, properties):
    try:
        if face==1:             ## GFP-GAN
            _, _, output = properties.enhance(image, has_aligned=False, only_center_face=False, paste_back=True)
        elif face==2:           ## CODEFORMER
            net = properties[0]
            device = properties[1]
            w = properties[2]
            image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
            cropped_face_t = img2tensor(image / 255., bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(device)
            try:
                with torch.no_grad():
                    cropped_face_t = net(cropped_face_t, w=w, adain=True)[0]
                    restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))
                del cropped_face_t
                torch.cuda.empty_cache()
            except Exception as error:
                print(f'\tFailed inference for CodeFormer: {error}')
                restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))
            output = restored_face.astype('uint8')
        elif face==0:           ## ESRGAN
            img = image.astype(np.float32) / 255.
            output, _ = properties.enhance(image, outscale=4)
    except RuntimeError as error:
        print('Error', error)
        print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
    return output
