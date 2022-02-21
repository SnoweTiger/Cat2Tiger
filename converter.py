

import torch

# import torch.nn as nn
# import torch.nn.functional as F

import matplotlib.pyplot as plt

import os



from utils import *
from model import Generator

import argparse


# import argparse
# parser = argparse.ArgumentParser()
# # parser.add_argument('-F', '--filename', help='input filename (video or image)')
# parser.add_argument('-O', '--output', help='output format video or gif (avalible only for video)')
# # parser.add_argument('-S', '--start', default=0, help='start frame (avalible only for video)')
# # parser.add_argument('-D', '--duration', help='duration in frame (avalible only for video)')
# args = parser.parse_args()
# print(args.filename)

parser = argparse.ArgumentParser()
parser.add_argument('-O', '--output', help='output format video or gif (avalible only for video)')
output_format = str(parser.parse_args().output)
# print(str(output_format))

WEIGHT_PATH = 'weight'
load_epoch = 28
device = 'cuda' if torch.cuda.is_available() else 'cpu'

files = os.listdir('.')
filename_image = 'image.jpg'
filename_video = 'video.mp4'

if filename_image in files:
    print('Found image. Start convert to tiger.')
    convert_image(device, filename_image, 'image_fake.jpg', load_epoch,  WEIGHT_PATH)
    print('Fake tiger image wrote to image_fake.jpg.')

elif filename_video in files:
    print('Found video. Start convert to tiger.')
    if output_format.lower() == 'gif': out_format = 'gif'
    else: out_format = 'video'
    # convert_video(device, filename_video, 'video_fake.gif', load_epoch, WEIGHT_PATH, out = out_format)
    convert_video(device, 'video.mp4', load_epoch, WEIGHT_PATH, out = out_format)
    print('Fake tiger video wrote to video_fake.')

else:
    print('No support file found.')
