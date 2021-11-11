import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from tensorDLT import solve_DLT
from tf_spatial_transform import transform
from tensorflow.contrib.layers import conv2d
import tf_spatial_transform_local

import constant
grid_w = constant.GRID_W
grid_h = constant.GRID_H


#Covert global homo into mesh
def H2Mesh(H2, patch_size):
    batch_size = tf.shape(H2)[0]
    h = patch_size/grid_h
    w = patch_size/grid_w
    ori_pt = []
    for i in range(grid_h + 1):
        for j in range(grid_w + 1):
            ww = j * w
            hh = i * h
            p = tf.constant([ww, hh, 1], shape=[3], dtype=tf.float32)
            ori_pt.append(tf.expand_dims(tf.expand_dims(p, 0),2))
    ori_pt = tf.concat(ori_pt, axis=2)
    ori_pt = tf.tile(ori_pt,[batch_size, 1, 1])
    tar_pt = tf.matmul(H2, ori_pt)
    
    x_s = tf.slice(tar_pt, [0, 0, 0], [-1, 1, -1])
    y_s = tf.slice(tar_pt, [0, 1, 0], [-1, 1, -1])
    z_s = tf.slice(tar_pt, [0, 2, 0], [-1, 1, -1])

    H2_local = tf.concat([x_s/z_s, y_s/z_s], axis=1)
    H2_local = tf.transpose(H2_local, perm=[0, 2, 1])
    H2_local = tf.reshape(H2_local, [batch_size, grid_h+1, grid_w+1, 2])
    
    return H2_local


def H_model(train_inputs_aug, train_inputs, train_depth, patch_size=512.):

    batch_size = tf.shape(train_inputs)[0]
    H1_motion, H2_motion, mesh_motion = build_model(train_inputs_aug) #, feature2_warp_gt, feature3_warp_gt
    
    #scale transformation matrix
    M = np.array([[patch_size / 2.0, 0., patch_size / 2.0],
                  [0., patch_size / 2.0, patch_size / 2.0],
                  [0., 0., 1.]]).astype(np.float32)
    M_tensor = tf.constant(M, tf.float32)
    M_tile = tf.tile(tf.expand_dims(M_tensor, [0]), [batch_size, 1, 1])
    # Inverse of M
    M_inv = np.linalg.inv(M)
    M_tensor_inv = tf.constant(M_inv, tf.float32)
    M_tile_inv = tf.tile(tf.expand_dims(M_tensor_inv, [0]), [batch_size, 1, 1])
    
    ## solve global homo H1/H2
    H1 = solve_DLT(H1_motion, patch_size)
    H2 = solve_DLT(H1_motion+H2_motion, patch_size)
    H1_mat = tf.matmul(tf.matmul(M_tile_inv, H1), M_tile)
    H2_mat = tf.matmul(tf.matmul(M_tile_inv, H2), M_tile)
    
    
    ###prepare for calculating loss###
    ## warp images using H1/H2
    image2_tensor = train_inputs[..., 3:6]
    warp2_H1 = transform(image2_tensor, H1_mat)
    warp2_H2 = transform(image2_tensor, H2_mat)
    ## warp all-one images using H1/H2
    one = tf.ones_like(image2_tensor, dtype=tf.float32)
    one_warp_H1 = transform(one, H1_mat)
    one_warp_H2 = transform(one, H2_mat)
    
    ## initialize the mesh using H2
    ini_mesh = H2Mesh(H2, patch_size)
    ## calculate the final predicted mesh
    mesh = ini_mesh + mesh_motion

    depth = tf.concat([train_depth, train_depth, train_depth], 3)
    ## warp the image/all-one image/ depth map using mesh
    warp2_H3, one_warp_H3, warp2_depth = tf_spatial_transform_local.transformer(image2_tensor, one, depth, mesh)
    warp2_depth = tf.expand_dims(tf.reduce_mean(warp2_depth, 3),3)
    warp2_depth = tf.clip_by_value(warp2_depth,  0, 1)

    
    return warp2_depth, mesh, warp2_H1, warp2_H2, warp2_H3, one_warp_H1, one_warp_H2, one_warp_H3

def _conv_block(x, num_out_layers, kernel_sizes, strides):
    conv1 = conv2d(inputs=x, num_outputs=num_out_layers[0], kernel_size=kernel_sizes[0], activation_fn=tf.nn.relu, scope='conv1')
    conv2 = conv2d(inputs=conv1, num_outputs=num_out_layers[1], kernel_size=kernel_sizes[1], activation_fn=tf.nn.relu, scope='conv2')
    return conv2


def build_model(train_inputs):
      with tf.variable_scope('model'):
        input1 = train_inputs[...,0:3]
        input2 = train_inputs[...,3:6]
        input1 = tf.expand_dims(tf.reduce_mean(input1, axis=3),[3])
        input2 = tf.expand_dims(tf.reduce_mean(input2, axis=3),[3])

        H1_motion, H2_motion, mesh_motion = _vgg(input1, input2)
        return H1_motion, H2_motion, mesh_motion


def feature_extractor(image_tf):
    feature = []
    with tf.variable_scope('conv_block1'): 
      conv1 = _conv_block(image_tf, ([64, 64]), (3, 3), (1, 1))      #512
      maxpool1 = slim.max_pool2d(conv1, 2, stride=2, padding = 'SAME')
    with tf.variable_scope('conv_block2'):
      conv2 = _conv_block(maxpool1, ([64, 64]), (3, 3), (1, 1))      #256
      maxpool2 = slim.max_pool2d(conv2, 2, stride=2, padding = 'SAME')
    with tf.variable_scope('conv_block3'):
      conv3 = _conv_block(maxpool2, ([128, 128]), (3, 3), (1, 1))    #128
      maxpool3 = slim.max_pool2d(conv3, 2, stride=2, padding = 'SAME')
    with tf.variable_scope('conv_block4'):
      conv4 = _conv_block(maxpool3, ([128, 128]), (3, 3), (1, 1))    #64
      conv1_r64 = tf.image.resize_images(conv1, [64, 64], method=0)
      conv2_r64 = tf.image.resize_images(conv2, [64, 64], method=0)
      conv3_r64 = tf.image.resize_images(conv3, [64, 64], method=0)
      feature.append(tf.concat([conv4, conv1_r64, conv2_r64, conv3_r64], 3))
      maxpool4 = slim.max_pool2d(conv4, 2, stride=2, padding = 'SAME')
    with tf.variable_scope('conv_block5'):
      conv5 = _conv_block(maxpool4, ([256, 256]), (3, 3), (1, 1))    #32
      conv1_r32 = tf.image.resize_images(conv1, [32, 32], method=0)
      conv2_r32 = tf.image.resize_images(conv2, [32, 32], method=0)
      conv3_r32 = tf.image.resize_images(conv3, [32, 32], method=0)
      conv4_r32 = tf.image.resize_images(conv4, [32, 32], method=0)
      feature.append(tf.concat([conv5, conv1_r32, conv2_r32, conv3_r32, conv4_r32], 3))
      maxpool5 = slim.max_pool2d(conv5, 2, stride=2, padding = 'SAME')
    with tf.variable_scope('conv_block6'):                                      
      conv6 = _conv_block(maxpool5, ([256, 256]), (3, 3), (1, 1))    #16
      conv1_r16 = tf.image.resize_images(conv1, [16, 16], method=0)
      conv2_r16 = tf.image.resize_images(conv2, [16, 16], method=0)
      conv3_r16 = tf.image.resize_images(conv3, [16, 16], method=0)
      conv4_r16 = tf.image.resize_images(conv4, [16, 16], method=0)
      conv5_r16 = tf.image.resize_images(conv5, [16, 16], method=0)
      feature.append(tf.concat([conv6, conv1_r16, conv2_r16, conv3_r16, conv4_r16, conv5_r16], 3))
    
    return feature


# contextual correlation layer
def CCL(c1, warp):
    shape = warp.get_shape().as_list()
    kernel = 3
    stride = 1
    rate = 1
    if shape[1] == 16:
      rate = 1
      stride = 1
    elif shape[1] == 32:
      rate = 2
      stride = 1
    else:
      rate = 3
      stride = 1
    
    # extract patches as convolutional filters
    patches = tf.extract_image_patches(warp, [1,kernel,kernel,1], [1,stride,stride,1], [1,rate,rate,1], padding='SAME')
    patches = tf.reshape(patches, [shape[0], -1, kernel, kernel, shape[3]])
    matching_filters = tf.transpose(patches, [0, 2, 3, 4, 1])
    print(matching_filters.shape)
    
    # using convolution to match
    match_vol = []
    for i in range(shape[0]):
      single_match = tf.nn.atrous_conv2d(tf.expand_dims(c1[i], [0]), matching_filters[i], rate=rate, padding='SAME')
      match_vol.append(single_match)
    
    match_vol = tf.concat(match_vol, axis=0)
    channels = int(match_vol.shape[3])

    print("channels")
    print(channels)
    
    # scale softmax
    softmax_scale = 10
    match_vol = tf.nn.softmax(match_vol*softmax_scale,3)
    
    # convert the correlation volume to feature flow
    h_one = tf.linspace(0., shape[1]-1., int(match_vol.shape[1]))
    w_one = tf.linspace(0., shape[2]-1., int(match_vol.shape[2]))
    h_one = tf.matmul(tf.expand_dims(h_one, 1), tf.ones(shape=tf.stack([1, shape[2]])))
    w_one = tf.matmul(tf.ones(shape=tf.stack([shape[2], 1])), tf.transpose(tf.expand_dims(w_one, 1), [1, 0]))
    h_one = tf.tile(tf.expand_dims(tf.expand_dims(h_one, 0),3), [shape[0],1,1,channels])
    w_one = tf.tile(tf.expand_dims(tf.expand_dims(w_one, 0),3), [shape[0],1,1,channels])
    
    i_one = tf.expand_dims(tf.linspace(0., channels-1., channels),0)
    i_one = tf.expand_dims(i_one,0)
    i_one = tf.expand_dims(i_one,0)
    i_one = tf.tile(i_one, [shape[0], shape[1], shape[2], 1])
 
    flow_w = match_vol*(i_one%shape[2] - w_one)
    flow_h = match_vol*(i_one//shape[2] - h_one)
    flow_w = tf.expand_dims(tf.reduce_sum(flow_w,3),3)
    flow_h = tf.expand_dims(tf.reduce_sum(flow_h,3),3)
    
    flow = tf.concat([flow_w, flow_h], 3)
    print("flow.shape")
    print(flow.shape)

    return flow
    
def regression_H4pt_Net1(correlation):
    conv1 = conv2d(inputs=correlation, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu)
    conv1 = conv2d(inputs=conv1, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu)
    
    maxpool1 = slim.max_pool2d(conv1, 2, stride=2, padding = 'SAME')
    conv2 = conv2d(inputs=maxpool1, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu)
    conv2 = conv2d(inputs=conv2, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu)
    
    maxpool2 = slim.max_pool2d(conv2, 2, stride=2, padding = 'SAME')
    conv3 = conv2d(inputs=maxpool2, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu)
    conv3 = conv2d(inputs=conv3, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu)
    
    fc1 = conv2d(inputs=conv3, num_outputs=128, kernel_size=4, activation_fn=tf.nn.relu, padding="VALID")
    fc2 = conv2d(inputs=fc1, num_outputs=128, kernel_size=1, activation_fn=tf.nn.relu)
    fc3 = conv2d(inputs=fc2, num_outputs=8, kernel_size=1, activation_fn=None)
    H1_motion = tf.expand_dims(tf.squeeze(tf.squeeze(fc3,1),1), [2])
    
    return H1_motion
    
def regression_H4pt_Net2(correlation):
    conv1 = conv2d(inputs=correlation, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu)
    conv1 = conv2d(inputs=conv1, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu)
    
    maxpool1 = slim.max_pool2d(conv1, 2, stride=2, padding = 'SAME')
    conv2 = conv2d(inputs=maxpool1, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu)
    conv2 = conv2d(inputs=conv2, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu)
    
    maxpool2 = slim.max_pool2d(conv2, 2, stride=2, padding = 'SAME')
    conv3 = conv2d(inputs=maxpool2, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu)
    conv3 = conv2d(inputs=conv3, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu)
    
    maxpool3 = slim.max_pool2d(conv3, 2, stride=2, padding = 'SAME')
    conv4 = conv2d(inputs=maxpool3, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu)
    conv4 = conv2d(inputs=conv4, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu)
    
    fc1 = conv2d(inputs=conv4, num_outputs=128, kernel_size=4, activation_fn=tf.nn.relu, padding="VALID")
    fc2 = conv2d(inputs=fc1, num_outputs=128, kernel_size=1, activation_fn=tf.nn.relu)
    fc3 = conv2d(inputs=fc2, num_outputs=8, kernel_size=1, activation_fn=None)
    H2_motion = tf.expand_dims(tf.squeeze(tf.squeeze(fc3,1),1), [2])
    
    return H2_motion
    
def regression_H4pt_Net3(correlation):
    conv1 = conv2d(inputs=correlation, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu)
    conv1 = conv2d(inputs=conv1, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu)
    
    maxpool1 = slim.max_pool2d(conv1, 2, stride=2, padding = 'SAME')
    conv2 = conv2d(inputs=maxpool1, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu)
    conv2 = conv2d(inputs=conv2, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu)
    
    maxpool2 = slim.max_pool2d(conv2, 2, stride=2, padding = 'SAME')
    conv3 = conv2d(inputs=maxpool2, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu)
    conv3 = conv2d(inputs=conv3, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu)
    
    maxpool3 = slim.max_pool2d(conv3, 2, stride=2, padding = 'SAME')
    conv4 = conv2d(inputs=maxpool3, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu)
    conv4 = conv2d(inputs=conv4, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu)
    
    maxpool4 = slim.max_pool2d(conv4, 2, stride=2, padding = 'SAME')
    conv5 = conv2d(inputs=maxpool4, num_outputs=256, kernel_size=3, activation_fn=tf.nn.relu)
    conv5 = conv2d(inputs=conv5, num_outputs=256, kernel_size=3, activation_fn=tf.nn.relu)
    
    fc1 = conv2d(inputs=conv5, num_outputs=2048, kernel_size=4, activation_fn=tf.nn.relu, padding="VALID")
    fc2 = conv2d(inputs=fc1, num_outputs=1024, kernel_size=1, activation_fn=tf.nn.relu)
    fc3 = conv2d(inputs=fc2, num_outputs=(grid_w+1)*(grid_h+1)*2, kernel_size=1, activation_fn=None)
    #net3_f = tf.expand_dims(tf.squeeze(tf.squeeze(fc3,1),1), [2])
    mesh_motion = tf.reshape(fc3, (-1, grid_h+1, grid_w+1, 2))
    
    return mesh_motion



def _vgg(input1, input2):
    batch_size = tf.shape(input1)[0]
    
    ## feature extractors with shared weights
    with tf.variable_scope('feature_extract', reuse = None): 
      feature1 = feature_extractor(input1)
    with tf.variable_scope('feature_extract', reuse = True): 
      feature2 = feature_extractor(input2)
      
    # the 1st layer of the pyramid
    featureflow_1 = CCL(tf.nn.l2_normalize(feature1[-1],axis=3), tf.nn.l2_normalize(feature2[-1],axis=3))  
    H1_motion = regression_H4pt_Net1(featureflow_1)
    
    # warp the feature map
    patch_size = 32.
    H1 = solve_DLT(H1_motion/16., patch_size)
    M = np.array([[patch_size / 2.0, 0., patch_size / 2.0],
                  [0., patch_size / 2.0, patch_size / 2.0],
                  [0., 0., 1.]]).astype(np.float32)
    M_tensor = tf.constant(M, tf.float32)
    M_tile = tf.tile(tf.expand_dims(M_tensor, [0]), [batch_size, 1, 1])
    M_inv = np.linalg.inv(M)
    M_tensor_inv = tf.constant(M_inv, tf.float32)
    M_tile_inv = tf.tile(tf.expand_dims(M_tensor_inv, [0]), [batch_size, 1, 1])
    H1 = tf.matmul(tf.matmul(M_tile_inv, H1), M_tile)
    feature2_warp = transform(tf.nn.l2_normalize(feature2[-2],axis=3), H1)
    
    # the 2nd layer of the pyramid
    featureflow_2 = CCL(tf.nn.l2_normalize(feature1[-2],axis=3), feature2_warp)  
    H2_motion = regression_H4pt_Net2(featureflow_2)
    
    # warp the feature map
    patch_size = 64.
    H2 = solve_DLT((H1_motion+H2_motion)/8., patch_size)
    M = np.array([[patch_size / 2.0, 0., patch_size / 2.0],
                  [0., patch_size / 2.0, patch_size / 2.0],
                  [0., 0., 1.]]).astype(np.float32)
    M_tensor = tf.constant(M, tf.float32)
    M_tile = tf.tile(tf.expand_dims(M_tensor, [0]), [batch_size, 1, 1])
    M_inv = np.linalg.inv(M)
    M_tensor_inv = tf.constant(M_inv, tf.float32)
    M_tile_inv = tf.tile(tf.expand_dims(M_tensor_inv, [0]), [batch_size, 1, 1])
    H2 = tf.matmul(tf.matmul(M_tile_inv, H2), M_tile)
    feature3_warp = transform(tf.nn.l2_normalize(feature2[-3],axis=3), H2)
    
    # the 3rd layer of the pyramid
    featureflow_3 = CCL(tf.nn.l2_normalize(feature1[-3],axis=3), feature3_warp)     
    mesh_motion = regression_H4pt_Net3(featureflow_3)
    
    
    return H1_motion, H2_motion, mesh_motion