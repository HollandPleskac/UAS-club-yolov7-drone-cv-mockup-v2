# UAS-club-yolov7-drone-cv-mockup-v2

Some instructions to get this setup

1. create conda environment with python=3.9
   * install pytorch with conda like it says to do on the official installation (use your cuda version)
   * conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
2. Follow this: https://www.youtube.com/watch?v=-QWxJ0j9EY8&t=378s
   * skip the step where he creates the requirements_gpu step
   * just run pip install -r requirements.txt instead
3. You will run into an error that is the exact same as this stack overflow post! Follow the check marked answer and the code will work.
   * https://stackoverflow.com/questions/74372636/indices-should-be-either-on-cpu-or-on-the-same-device-as-the-indexed-tensor
  

Great Resource for yolov7 with camera: https://www.youtube.com/watch?v=XzUMigbYRUI

Great Resource for yolov7: https://stackabuse.com/real-time-pose-estimation-from-video-in-python-with-yolov7/
