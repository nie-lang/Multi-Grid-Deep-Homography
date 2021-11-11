import tensorflow as tf
import os
import numpy as np
import cv2 as cv

from models import H_estimator
from utils import DataLoader, load, save
import constant
import skimage


os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = constant.GPU
test_folder = constant.TEST_FOLDER
snapshot_dir =  constant.SNAPSHOT_DIR + '/model.ckpt-500000'
batch_size = constant.TEST_BATCH_SIZE

height, width = 512, 512



# define dataset
with tf.name_scope('dataset'):
    ##########testing###############
    
    test_inputs_clips_tensor = tf.placeholder(shape=[batch_size, height, width, 3 * 2], dtype=tf.float32)
    test_inputs = test_inputs_clips_tensor
    print('test inputs = {}'.format(test_inputs))
    



# depth is not needed in the inference process, 
#we assign "test_depth" arbitrary values such as an all-one map
test_depth = tf.ones_like(test_inputs[...,0:1])
print("test_depth.shape")
print(test_depth.shape)
with tf.variable_scope('generator', reuse=None):
    print('testing = {}'.format(tf.get_variable_scope().name))
    test_warp2_depth, test_mesh, test_warp2_H1, test_warp2_H2, test_warp2_H3, test_one_warp_H1, test_one_warp_H2, test_one_warp_H3 = H_estimator(test_inputs, test_inputs, test_depth)
    


config = tf.ConfigProto()
config.gpu_options.allow_growth = True      
with tf.Session(config=config) as sess:
    # dataset
    data_loader = DataLoader(test_folder, height, width)

    # initialize weights
    sess.run(tf.global_variables_initializer())
    print('Init global successfully!')

    # tf saver
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=None)

    restore_var = [v for v in tf.global_variables()]
    loader = tf.train.Saver(var_list=restore_var)

    def inference_func(ckpt):
        print("============")
        print(ckpt)
        load(loader, sess, ckpt)
        print("============")
        length = 1106
        psnr_list = []
        ssim_list = []

        for i in range(0, length):
            input_clip = np.expand_dims(data_loader.get_data_clips(i), axis=0)
            
            #Attention: both inputs and outpus are the types of numpy, that is :(preH, warp_gt) and (input_clip,h_clip)
            _, mesh, warp_H1, warp_H2, warp_H3, warp_one_H1, warp_one_H2, warp_one_H3 = sess.run([test_warp2_depth, test_mesh, test_warp2_H1, test_warp2_H2, test_warp2_H3, test_one_warp_H1, test_one_warp_H2, test_one_warp_H3], 
                    feed_dict={test_inputs_clips_tensor: input_clip})
            
            # warp  = warp_H3
            final_warp = (warp_H3+1) * 127.5    
            final_warp = final_warp[0] 
            # warp_one  = warp_one_H3
            final_warp_one = warp_one_H3[0]
            # input1
            input1 = (input_clip[...,0:3]+1) * 127.5    
            input1 = input1[0]
            
            # calculate psnr/ssim
            psnr = skimage.measure.compare_psnr(input1*final_warp_one, final_warp*final_warp_one, 255)
            ssim = skimage.measure.compare_ssim(input1*final_warp_one, final_warp*final_warp_one, data_range=255, multichannel=True)
            
            # image fusion
            img1 = input1
            img2 = final_warp*final_warp_one
            fusion = np.zeros((512,512,3), np.uint8)
            fusion[...,0] = img2[...,0] 
            fusion[...,1] = img1[...,1]*0.5 +  img2[...,1]*0.5
            fusion[...,2] = img1[...,2]
            path = "../fusion/" + str(i+1).zfill(6) + ".jpg"
            cv.imwrite(path, fusion)
            
            
            print('i = {} / {}, psnr = {:.6f}'.format( i+1, length, psnr))
            
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            
            
        print("===================Results Analysis==================")   
        print("psnr")
        psnr_list.sort(reverse = True)
        psnr_list_30 = psnr_list[0 : 331]
        psnr_list_60 = psnr_list[331: 663]
        psnr_list_100 = psnr_list[663: -1]
        print("top 30%", np.mean(psnr_list_30))
        print("top 30~60%", np.mean(psnr_list_60))
        print("top 60~100%", np.mean(psnr_list_100))
        print('average psnr:', np.mean(psnr_list))
        
        ssim_list.sort(reverse = True)
        ssim_list_30 = ssim_list[0 : 331]
        ssim_list_60 = ssim_list[331: 663]
        ssim_list_100 = ssim_list[663: -1]
        print("top 30%", np.mean(ssim_list_30))
        print("top 30~60%", np.mean(ssim_list_60))
        print("top 60~100%", np.mean(ssim_list_100))
        print('average ssim:', np.mean(ssim_list))
        
    inference_func(snapshot_dir)

    






