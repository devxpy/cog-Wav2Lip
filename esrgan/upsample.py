import os
import warnings
from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer

warnings.filterwarnings("ignore")

def load_sr(model_path, device, face):
  model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4) #alter to match dims as needed
  model_path = os.path.normpath(model_path)
  run_params=gfp = GFPGANer(
      model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
      upscale=1,
      arch='clean',
      channel_multiplier=2,
      bg_upsampler=None)
  
  return run_params


def upscale(image, face, properties):
      _, _, output = properties.enhance(image, has_aligned=False, only_center_face=True, paste_back=True)
      return output
