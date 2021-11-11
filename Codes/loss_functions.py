import tensorflow as tf
import numpy as np

import constant
grid_w = constant.GRID_W
grid_h = constant.GRID_H


def intensity_loss(gen_frames, gt_frames, l_num):
    """
    Calculates the sum of lp losses between the predicted and ground truth frames.

    @param gen_frames: The predicted frames at each scale.
    @param gt_frames: The ground truth frames at each scale
    @param l_num: 1 or 2 for l1 and l2 loss, respectively).

    @return: The lp loss.
    """
    return tf.reduce_mean(tf.abs((gen_frames - gt_frames) ** l_num))



def depth_consistency_loss3(warp2_depth, mesh):
    shape = warp2_depth.get_shape().as_list()
    
    # assign average depth value to each grid
    depth_patches = tf.extract_image_patches(warp2_depth, [1,shape[1]/grid_h,shape[2]/grid_w,1], [1,shape[1]/grid_h,shape[2]/grid_w,1], [1,1,1,1], padding='VALID')
    depth_map = tf.reduce_mean(depth_patches, axis=3)
    depth_map = tf.reshape(depth_map, [shape[0], grid_h, grid_w])
    
    ones = tf.ones_like(depth_map, dtype=tf.float32)
    zeros = tf.zeros_like(depth_map, dtype=tf.float32)
    
    ##############################
    # compute horizontal edges
    w_edges = mesh[:,:,0:grid_w,:] - mesh[:,:,1:grid_w+1,:]
    # compute angles of two successive horizontal edges
    cos_w = tf.reduce_sum(w_edges[:,:,0:grid_w-1,:] * w_edges[:,:,1:grid_w,:],3) / (tf.sqrt(tf.reduce_sum(w_edges[:,:,0:grid_w-1,:]*w_edges[:,:,0:grid_w-1,:],3))*tf.sqrt(tf.reduce_sum(w_edges[:,:,1:grid_w,:]*w_edges[:,:,1:grid_w,:],3)))
    # horizontal angle-preserving error for two successive horizontal edges
    delta_w_angle = 1 - cos_w
    # horizontal angle-preserving error for two successive horizontal grids
    delta_w_angle = delta_w_angle[:,0:grid_h,:] + delta_w_angle[:,1:grid_h+1,:]
    ##############################
    
    ##############################
    # compute vertical edges
    h_edges = mesh[:,0:grid_h,:,:] - mesh[:,1:grid_h+1,:,:]
    # compute angles of two successive vertical edges
    cos_h = tf.reduce_sum(h_edges[:,0:grid_h-1,:,:] * h_edges[:,1:grid_h,:,:],3) / (tf.sqrt(tf.reduce_sum(h_edges[:,0:grid_h-1,:,:]*h_edges[:,0:grid_h-1,:,:],3))*tf.sqrt(tf.reduce_sum(h_edges[:,1:grid_h,:,:]*h_edges[:,1:grid_h,:,:],3)))
    # vertical angle-preserving error for two successive vertical edges
    delta_h_angle = 1 - cos_h
    # vertical angle-preserving error for two successive vertical grids
    delta_h_angle = delta_h_angle[:,:,0:grid_w] + delta_h_angle[:,:,1:grid_w+1]
    ##############################
    
    
    error = []
    # define the number of depth levels
    depth_num = 32
    for i in range(depth_num):
        # compute the start/end depth value for i-th depth level
        start_depth = i*(1./depth_num)
        end_depth = (i+1)*(1./depth_num)
        # get the 0-1 mask for the i-th depth level
        depth_mask = tf.where(tf.logical_and((depth_map>start_depth) , (depth_map<=end_depth)), ones, zeros)
        
        # successive depth grid on the horizontal dimension
        depth_diff_w = (1-tf.abs(depth_mask[:,:,0:grid_w-1] - depth_mask[:,:,1:grid_w])) * depth_mask[:,:,0:grid_w-1]
        error_w = depth_diff_w * delta_w_angle
        # successive depth grid on the vertical dimension
        depth_diff_h = (1-tf.abs(depth_mask[:,0:grid_h-1,:] - depth_mask[:,1:grid_h,:])) * depth_mask[:,0:grid_h-1,:]
        error_h = depth_diff_h * delta_h_angle

        error.append(tf.reduce_mean(error_w) + tf.reduce_mean(error_h))

    
    loss = tf.stack(error, 0)
    loss = tf.reduce_sum(loss)
    
    
    return loss



