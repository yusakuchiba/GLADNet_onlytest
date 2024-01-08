from __future__ import print_function
import os
import argparse
from glob import glob

from PIL import Image
import tensorflow as tf

from model import lowlight_enhance
from utils import *

parser = argparse.ArgumentParser(description='')

parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--gpu_idx', dest='gpu_idx', default="0", help='GPU idx')
parser.add_argument('--gpu_mem', dest='gpu_mem', type=float, default=0.8, help="0 to 1, gpu memory usage")
parser.add_argument('--phase', dest='phase', default='train', help='train or test')

parser.add_argument('--epoch', dest='epoch', type=int, default=50, help='number of total epoches')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=8, help='number of samples in one batch')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=384, help='patch size')
parser.add_argument('--eval_every_epoch', dest='eval_every_epoch', default=1, help='evaluating and saving checkpoints every #  epoch')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint', help='directory for checkpoints')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='directory for evaluating outputs')

parser.add_argument('--save_dir', dest='save_dir', default='./test_results', help='directory for testing outputs')
parser.add_argument('--test_dir', dest='test_dir', default='./data/test/low', help='directory for testing inputs')

args = parser.parse_args()
#save_dir = './test_results'

def lowlight_train(lowlight_enhance):
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)

    train_low_data = []
    train_high_data = []

    train_low_data_names = glob('/mnt/hdd/wangwenjing/FGtraining/low/*.png')#./data/train/low/*.png') 
    train_low_data_names.sort()
    train_high_data_names = glob('/mnt/hdd/wangwenjing/FGtraining/normal/*.png')#./data/train/normal/*.png') 
    train_high_data_names.sort()
    assert len(train_low_data_names) == len(train_high_data_names)
    print('[*] Number of training data: %d' % len(train_low_data_names))

    for idx in range(len(train_low_data_names)):
        if (idx + 1) % 1000 == 0:
            print(idx + 1)
        low_im = load_images(train_low_data_names[idx])
        train_low_data.append(low_im)
        high_im = load_images(train_high_data_names[idx])
        train_high_data.append(high_im)

    eval_low_data = []
    eval_high_data = []

    eval_low_data_name = glob('./data/eval/low/*.*')

    for idx in range(len(eval_low_data_name)):
        eval_low_im = load_images(eval_low_data_name[idx])
        eval_low_data.append(eval_low_im)

    lowlight_enhance.train(train_low_data, train_high_data, eval_low_data, batch_size=args.batch_size, patch_size=args.patch_size, epoch=args.epoch, sample_dir=args.sample_dir, ckpt_dir=args.ckpt_dir, eval_every_epoch=args.eval_every_epoch)


def lowlight_test(lowlight_enhance):
    data_type = ["test","train","val"]

    if args.test_dir == None:
        print("[!] please provide --test_dir")
        exit(0)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)


    for word in sorted(os.listdir(args.test_dir)):
        #print(len(os.listdir(os.path.join(args.test_dir,word))))
        word_dir = os.path.join(args.test_dir,word)
        #print(word_dir)

        if not os.path.exists(os.path.join(args.save_dir,word)):
            os.makedirs(os.path.join(args.save_dir,word))


        for dt in data_type:

            dt_dir = os.path.join(word_dir,dt)
            #print(dt_dir)
            if not os.path.exists(os.path.join(args.save_dir,word,dt)):
                os.makedirs(os.path.join(args.save_dir,word,dt))

            for frame in sorted(os.listdir(dt_dir)):
                frame_dir = os.path.join(dt_dir,frame)
                print(frame_dir)
                if not os.path.exists(os.path.join(args.save_dir,word,dt,frame)):
                    os.makedirs(os.path.join(args.save_dir,word,dt,frame))
                #print(f"frame:{frame}")
                #test_low_data_name = glob(os.path.join(frame) + '/*.*')
                test_low_data_name = glob(frame_dir + '\*.*')
                #print(f"testdir:{test_low_data_name}")
                test_low_data = []
                test_high_data = []
                save_dir = os.path.join(args.save_dir,word,dt,frame)
                for idx in range(len(test_low_data_name)):
                    #print(os.path.join("E:\split",*(test_low_data_name[idx].split("\\")[2:])))
                    if not os.path.exists(os.path.join("E:\glad",*(test_low_data_name[idx].split("\\")[2:]))):
                        print(os.path.join("E:\glad",*(test_low_data_name[idx].split("\\")[2:])))
                        test_low_im = load_images(test_low_data_name[idx])
                        test_low_data.append(test_low_im)
                        print("append:",test_low_data_name[idx])
                    
                    

                #print(f"savedir:{save_dir}")
                lowlight_enhance.test(test_low_data, test_high_data, test_low_data_name, save_dir)


def main(_):
    if args.use_gpu:
        print("[*] GPU\n")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=args.gpu_mem)
        with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) as sess:
            model = lowlight_enhance(sess)
            if args.phase == 'train':
                lowlight_train(model)
            elif args.phase == 'test':
                lowlight_test(model)
            else:
                print('[!] Unknown phase')
                exit(0)
    else:
        print("[*] CPU\n")
        with tf.compat.v1.Session() as sess:
            model = lowlight_enhance(sess)
            if args.phase == 'train':
                lowlight_train(model)
            elif args.phase == 'test':
                lowlight_test(model)
            else:
                print('[!] Unknown phase')
                exit(0)

if __name__ == '__main__':
    tf.compat.v1.app.run()
