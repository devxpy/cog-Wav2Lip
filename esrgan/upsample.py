import sys
import os
import warnings
from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer

warnings.filterwarnings("ignore")

def load_sr(sr_model):
  if sr_model == 'gfpgan':
    run_params = GFPGANer(
      model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
      upscale=1,
      arch='clean',
      channel_multiplier=2,
      bg_upsampler=None)
  elif sr_model == 'RestoreFormer':
    run_params = GFPGANer(
      model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth',
      upscale=1,
      arch='RestoreFormer',
      channel_multiplier=2,
      bg_upsampler=None)
  else:
    sys.exit('load_sr model not supported - must be gfpgan or RestoreFormer')    
  
  return run_params


def upscale(image, properties):
      _, _, output = properties.enhance(image, has_aligned=False, only_center_face=False, paste_back=True)
      return output
