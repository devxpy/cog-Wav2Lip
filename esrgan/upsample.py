import os
import warnings
from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer

warnings.filterwarnings("ignore")

def load_sr(model_path, device, face):
  run_params = GFPGANer(
      model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
      upscale=1,
      arch='clean',
      channel_multiplier=2,
      bg_upsampler=None)

  '''run_params = GFPGANer(
      model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth',
      upscale=1,
      arch='RestoreFormer',
      channel_multiplier=2,
      bg_upsampler=None)'''
  
  return run_params


def upscale(image, face, properties):
      _, _, output = properties.enhance(image, has_aligned=False, only_center_face=False, paste_back=True)
      return output
