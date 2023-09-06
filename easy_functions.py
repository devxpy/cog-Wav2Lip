import subprocess
import json

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
