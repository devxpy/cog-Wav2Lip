import subprocess
import json
from models import Wav2Lip

def get_video_details(filename):
  cmd = ['ffprobe', '-v', 'error', '-show_format', '-show_streams', '-of', 'json', filename]
  result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  info = json.loads(result.stdout)

  # Get video stream
  video_stream = next(stream for stream in info['streams'] if stream['codec_type'] == 'video')

  # Get resolution
  width = int(video_stream['width'])
  height = int(video_stream['height'])
  resolution = width*height

  # Get fps
  fps = eval(video_stream['avg_frame_rate'])

  # Get length
  length = float(info['format']['duration'])

  return width, height, fps, length

def show_video(file_path):
  """Function to display video in Colab"""
  mp4 = open(file_path,'rb').read()
  data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
  display(HTML("""
  <video controls width=600>
      <source src="%s" type="video/mp4">
  </video>
  """ % data_url))

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)

    if hours > 0:
        return f'{hours}h {minutes}m {seconds}s'
    elif minutes > 0:
        return f'{minutes}m {seconds}s'
    else:
        return f'{seconds}s'

def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_model(path):
    model = Wav2Lip()
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()

def get_input_length(filename):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return float(result.stdout)

def is_url(string):
    url_regex = re.compile(r'^(https?|ftp)://[^\s/$.?#].[^\s]*$')
    return bool(url_regex.match(string))
