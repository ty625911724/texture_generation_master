import argparse
import tensorflow as tf
import numpy as np
import os
from PIL import Image

from utils import build_content_loss
from utils import build_style_loss
from utils import buid_cc_loss_lays
from vgg_model import build_vgg19

parser = argparse.ArgumentParser()
parser.add_argument('--iteration', type=int, default=10000, help='the number of iterations')
parser.add_argument('--cc_loss', action='store_true', help='use the cc_loss')
parser.add_argument('--content', action='store_true', help='use the content image')
parser.add_argument('--content_path', type=str, default='./images/noisy.jpg', help='The path of the content image')
parser.add_argument('--style_path', type=str, default='./images/brick.jpg', help='The path of the content image')
opt = parser.parse_args()

IMAGE_W = 512 
IMAGE_H = 512 
CONTENT_IMG =  opt.content_path
STYLE_IMG = opt.style_path
if opt.cc_loss:
  OUTOUT_DIR = './results_cc'
else:
  OUTOUT_DIR = './results_style'
OUTPUT_IMG = 'results.png'
VGG_MODEL = 'imagenet-vgg-verydeep-19.mat'
INI_NOISE_RATIO = 0.7
if opt.content and opt.cc_loss:
  STYLE_STRENGTH = 0.05
  cc_STRENGTH = 2
elif opt.content and not opt.cc_loss:
  STYLE_STRENGTH = 0.5
  cc_STRENGTH = 20
else:
  STYLE_STRENGTH = 500
  cc_STRENGTH = 20000

CONTENT_LAYERS =[('conv4_2',1.)]
STYLE_LAYERS=[('conv1_1',1.),('conv2_1',1.),('conv3_1',1.),('conv4_1',1.),('conv5_1',1.)]

MEAN_VALUES = np.array([123, 117, 104]).reshape((1,1,1,3))

def read_image(path):
  image = Image.open(path)
  image = np.array(image.resize((IMAGE_H,IMAGE_W)))
  image = image[np.newaxis,:,:,:] 
  image = image - MEAN_VALUES
  return image

def write_image(path, image):
  image = image + MEAN_VALUES
  image = image[0]
  image = np.clip(image, 0, 255).astype('uint8')
  image = Image.fromarray(image)
  image.save(path)

def main():
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  net = build_vgg19(VGG_MODEL,IMAGE_W,IMAGE_H)

  sess = tf.Session(config=config)
  sess.run(tf.initialize_all_variables())
  noise_img = np.random.uniform(-20, 20, (1, IMAGE_H, IMAGE_W, 3)).astype('float32')
  content_img = read_image(CONTENT_IMG)
  style_img = read_image(STYLE_IMG)
  
  if opt.content:
    sess.run([net['input'].assign(content_img)])
    cost_content = sum(map(lambda l,: l[1]*build_content_loss(sess.run(net[l[0]]) ,  net[l[0]])
      , CONTENT_LAYERS))

  sess.run([net['input'].assign(style_img)])
  cost_style = sum(map(lambda l: l[1]*build_style_loss(sess.run(net[l[0]]) ,  net[l[0]])
    , STYLE_LAYERS))

  if opt.cc_loss:
    cost_cc = buid_cc_loss_lays(STYLE_LAYERS,net,sess)
    cost_total = STYLE_STRENGTH * cost_style + cc_STRENGTH * cost_cc
  else:
    cost_total = STYLE_STRENGTH * cost_style

  if opt.content:
    cost_total = cost_total + cost_content

  #optimizer = tf.train.AdamOptimizer(2.0)
  optimizer = tf.train.AdamOptimizer(1.0)

  train = optimizer.minimize(cost_total)
  sess.run(tf.initialize_all_variables())
  
  if opt.content:
    sess.run(net['input'].assign(INI_NOISE_RATIO* noise_img + (1.-INI_NOISE_RATIO) * content_img))
  else:
    sess.run(net['input'].assign(content_img))

  if not os.path.exists(OUTOUT_DIR):
      os.mkdir(OUTOUT_DIR)

  for i in range(opt.iteration):
    sess.run(train)
    if i%100 ==0:
      result_img = sess.run(net['input'])
      if opt.content:
        print('cost_total:',sess.run(cost_total),' cost_content:',sess.run(cost_content),' style_content:'
          ,STYLE_STRENGTH *sess.run(cost_style))
      elif opt.cc_loss:
        cost_cc_value = sess.run(cost_cc)
        cost_style_value = sess.run(cost_style)
        cost_total_value = STYLE_STRENGTH * cost_style_value + cc_STRENGTH * cost_cc_value
        print('cost_cc_value:',cost_cc_value,' cost_style_value:',cost_style_value, ' cost_total_value:',cost_total_value)
      else:
        cost_style_value = sess.run(cost_style)
        cost_total_value = STYLE_STRENGTH * cost_style_value
        print('cost_total_value:',cost_total_value)
      write_image(os.path.join(OUTOUT_DIR,'%s.png'%(str(i).zfill(4))),result_img)
  write_image(os.path.join(OUTOUT_DIR,OUTPUT_IMG),result_img)

if __name__ == '__main__':
  main()