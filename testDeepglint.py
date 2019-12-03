from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function

import argparse
import json
from functools import partial
import traceback

import imlib as im
import numpy as np
import tensorflow as tf
import pylib
import tflib as tl
import os

import models
import mydata


# ===========================================================
#                           param                           =
# ===========================================================

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', type=str, default='128', help='experiment_name')
parser.add_argument('--gpu', type=str, default='all', help='gpu')
parser.add_argument('--dataroot', type=str, default='/home/iim/Data/deepglint_112x112', help='data root path')
parser.add_argument('--attr_path', type=str, default='deepglint_attr_list_2.txt', help='attribute list file path')
parser.add_argument('--des_dir', type=str, default='/home/iim/Data/deepglint_emotion', help='destination dirictory path')

# for multipe attributes
parser.add_argument('--test_atts', nargs='+', default=None)
parser.add_argument('--test_ints', nargs='+', default=None, help='leave to None for all 1')
# for single attributes
parser.add_argument('--test_int', type=float, default=1.0, help='test_int')

args_ = parser.parse_args()
with open('./output/%s/setting.txt' % args_.experiment_name) as f:
    args = json.load(f)

# model
atts = args['atts']
n_att = len(atts)
img_size = args['img_size']
shortcut_layers = args['shortcut_layers']
inject_layers = args['inject_layers']
enc_dim = args['enc_dim']
dec_dim = args['dec_dim']
dis_dim = args['dis_dim']
dis_fc_dim = args['dis_fc_dim']
enc_layers = args['enc_layers']
dec_layers = args['dec_layers']
dis_layers = args['dis_layers']

label = args['label']
use_stu = args['use_stu']
stu_dim = args['stu_dim']
stu_layers = args['stu_layers']
stu_inject_layers = args['stu_inject_layers']
stu_kernel_size = args['stu_kernel_size']
stu_norm = args['stu_norm']
stu_state = args['stu_state']
multi_inputs = args['multi_inputs']
rec_loss_weight = args['rec_loss_weight']
one_more_conv = args['one_more_conv']

dataroot = args_.dataroot
attr_path = args_.attr_path
des_dir = args_.des_dir
gpu = args_.gpu
if gpu != 'all':
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu


##########testing############
# multipe attribute
test_atts = args_.test_atts
test_ints = args_.test_ints
if test_atts is not None and test_ints is None:
    test_ints = [1 for i in range(len(test_atts))]
# single attribute
test_int = args_.test_int

thres_int = args['thres_int']
use_cropped_img = args['use_cropped_img']
experiment_name = args_.experiment_name


# ============================================================================================
#                                       data                                                 =
# ============================================================================================

att_dict = {'Mouth_Slightly_Open': 0, 'Smiling': 1}
img_names = np.loadtxt(attr_path, skiprows=2, usecols=[0], dtype=np.str)
img_paths = [name for name in img_names]
att_id = [att_dict[att] + 1 for att in atts]
labels = np.loadtxt(attr_path, skiprows=2, usecols=att_id, dtype=np.int64)

'''
if isinstance(labels, tuple):
    dataset = tf.data.Dataset.from_tensor_slices((img_paths,) + tuple(labels))
else:
    dataset = tf.data.Dataset.from_tensor_slices((img_paths, labels))
'''

img_num = len(img_names)


# =============================================================================================
#                                       graphs                                                =
# =============================================================================================

sess = tl.session()
te_data = mydata.Test(dataroot, attr_path, atts, img_size, 1, part='test', sess=sess, crop=not use_cropped_img, im_no=None)
# models
Genc = partial(models.Genc, dim=enc_dim, n_layers=enc_layers, multi_inputs=multi_inputs)
Gdec = partial(models.Gdec, dim=dec_dim, n_layers=dec_layers, shortcut_layers=shortcut_layers,
            inject_layers=inject_layers, one_more_conv=one_more_conv)
Gstu = partial(models.Gstu, dim=stu_dim, n_layers=stu_layers, inject_layers=stu_inject_layers,
            kernel_size=stu_kernel_size, norm=stu_norm, pass_state=stu_state)

# inputs
xa_sample = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 3])
_b_sample = tf.placeholder(tf.float32, shape=[None, n_att])
raw_b_sample = tf.placeholder(tf.float32, shape=[None, n_att])

# sample
test_label = _b_sample - raw_b_sample if label == 'diff' else _b_sample
if use_stu:
    x_sample = Gdec(Gstu(Genc(xa_sample, is_training=False),
                        test_label, is_training=False), test_label, is_training=False)
else:
    x_sample = Gdec(Genc(xa_sample, is_training=False), test_label, is_training=False)


# ===========================================================================================
#                                      test                                                 =
# ===========================================================================================

# initialization
ckpt_dir = './output/%s/checkpoints' % experiment_name
tl.load_checkpoint(ckpt_dir, sess)

'''
# test
multi_atts = test_atts is not None

for i in range(img_num):
    img_name = img_paths[i]
    label = labels[i]
    img_path = os.path.join(dataroot, img_name)
    img = tf.read_file(img_path)
    img = tf.image.decode_png(img, 3)
    img = tf.image.resize_images(img, [img_size, img_size], tf.image.ResizeMethod.BICUBIC)
    img = tf.clip_by_value(img, 0, 255) / 127.5 - 1
    label = (label + 1) // 2


    print(img)

    xa_sample_ipt = img[np.newaxis, :, :, :]
    
    #xa_sample_ipt = xa_sample_ipt.eval(session=sess)
    print(xa_sample_ipt.shape)
    print(xa_sample_ipt)
    a_sample_ipt = label[np.newaxis, :]
    print(a_sample_ipt.shape)
    b_sample_ipt_list = [a_sample_ipt.copy() for _ in range(1)]
    print(test_atts)
    if label[0] == 1:
        multi_atts = ['Mouth_Slightly_Open']
    else:
        multi_atts = ['Smiling']

    for a in multi_atts:
        i = atts.index(a)
        b_sample_ipt_list[-1][:, i] = 1 - b_sample_ipt_list[-1][:, i]

    x_sample_opt_list = [xa_sample_ipt, np.full((1, img_size, img_size // 10, 3), -1.0)]
    raw_a_sample_ipt = a_sample_ipt.copy()
    raw_a_sample_ipt = (raw_a_sample_ipt * 2 - 1) * thres_int
    for i, b_sample_ipt in enumerate(b_sample_ipt_list):
        _b_sample_ipt = (b_sample_ipt * 2 - 1) * thres_int
        for t_att, t_int in zip(test_atts, test_ints):
            _b_sample_ipt[..., atts.index(t_att)] = _b_sample_ipt[..., atts.index(t_att)] * t_int
        if i > 0:
            _b_sample_ipt[..., i - 1] = _b_sample_ipt[..., i - 1] * test_int

        x_sample_opt_list.append(sess.run(x_sample, feed_dict={xa_sample: xa_sample_ipt,
                                                                   _b_sample: _b_sample_ipt,
                                                                   raw_b_sample: raw_a_sample_ipt}))
    sample = np.concatenate(x_sample_opt_list, 2)

    save_folder = 'sample_testing_multi'
    save_dir = './output/%s/%s' % (experiment_name, save_folder)
    pylib.mkdir(save_dir)
    im.imwrite(sample.squeeze(0), os.path.join(save_dir, img_name))
'''
# test
try:
    multi_atts = test_atts is not None
    for idx, batch in enumerate(te_data):
        if idx % 100 == 0:
            print(idx)
        img_name = img_paths[idx]
        xa_sample_ipt = batch[0]
        a_sample_ipt = batch[1]
        b_sample_ipt_list = [a_sample_ipt.copy() for _ in range(1)]

        if multi_atts: # test_multiple_attributes
            for a in test_atts:
                i = atts.index(a)
                b_sample_ipt_list[-1][:, i] = 1 - b_sample_ipt_list[-1][:, i]
                b_sample_ipt_list[-1] = mydata.Test.check_attribute_conflict(b_sample_ipt_list[-1], atts[i], atts)
        else: # test_single_attributes
            for i in range(len(atts)):
                tmp = np.array(a_sample_ipt, copy=True)
                tmp[:, i] = 1 - tmp[:, i]   # inverse attribute
                tmp = mydata.Test.check_attribute_conflict(tmp, atts[i], atts)
                b_sample_ipt_list.append(tmp)


        x_sample_opt_list = [xa_sample_ipt, np.full((1, img_size, img_size // 10, 3), -1.0)]
        raw_a_sample_ipt = a_sample_ipt.copy()
        raw_a_sample_ipt = (raw_a_sample_ipt * 2 - 1) * thres_int
        for i, b_sample_ipt in enumerate(b_sample_ipt_list):
            _b_sample_ipt = (b_sample_ipt * 2 - 1) * thres_int
            if multi_atts: # i must be 0
                for t_att, t_int in zip(test_atts, test_ints):
                    _b_sample_ipt[..., atts.index(t_att)] = _b_sample_ipt[..., atts.index(t_att)] * t_int
            if i > 0:   # i == 0 is for reconstruction
                _b_sample_ipt[..., i - 1] = _b_sample_ipt[..., i - 1] * test_int
            
            output = (sess.run(x_sample, feed_dict={xa_sample: xa_sample_ipt,
                                                                   _b_sample: _b_sample_ipt,
                                                                   raw_b_sample: raw_a_sample_ipt}))

        save_path = os.path.join(des_dir, img_name)
        img_real_name = 'emo_' + save_path.split('/')[-1]
        real_save_dir = os.path.join(des_dir, save_path.split('/')[-2])
        pylib.mkdir(real_save_dir)
        real_save_path = os.path.join(real_save_dir, img_real_name)


        im.imwrite(output.squeeze(0), real_save_path)


except:
    traceback.print_exc()
finally:
    sess.close()
