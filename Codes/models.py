import tensorflow as tf
import H_model



def H_estimator(train_inputs_aug, train_inputs, train_depth):
    return H_model.H_model(train_inputs_aug, train_inputs, train_depth)


def disjoint_augment_image_pair(train_inputs, min_val=-1, max_val=1):
    img1 = train_inputs[...,0:3]
    img2 = train_inputs[...,3:6]
    
    # Randomly shift brightness
    random_brightness = tf.random_uniform([], 0.7, 1.3)
    img1_aug = img1 * random_brightness
    random_brightness = tf.random_uniform([], 0.7, 1.3)
    img2_aug = img2 * random_brightness
    
    
    # Randomly shift color
    random_colors = tf.random_uniform([3], 0.7, 1.3)
    white = tf.ones([tf.shape(img1)[0], tf.shape(img1)[1], tf.shape(img1)[2]])
    color_image = tf.stack([white * random_colors[i] for i in range(3)], axis=3)
    img1_aug  *= color_image

    random_colors = tf.random_uniform([3], 0.7, 1.3)
    #white = tf.ones([tf.shape(img1)[0], tf.shape(img1)[1], tf.shape(img1)[2]])
    color_image = tf.stack([white * random_colors[i] for i in range(3)], axis=3)
    img2_aug  *= color_image
    
    

    # Saturate
    img1_aug  = tf.clip_by_value(img1_aug,  min_val, max_val)
    img2_aug  = tf.clip_by_value(img2_aug, min_val, max_val)
    
    train_inputs = tf.concat([img1_aug, img2_aug], axis = 3)

    return train_inputs
    
