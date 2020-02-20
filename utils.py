import tensorflow as tf
import numpy as np

def build_content_loss(p, x):
    M = p.shape[1]*p.shape[2]
    N = p.shape[3]
    loss = (1./(2* N**0.5 * M**0.5 )) * tf.reduce_sum(tf.pow((x - p),2))  
    return loss


def gram_matrix(x, area, depth):
    x1 = tf.reshape(x,(area,depth))
    g = tf.matmul(tf.transpose(x1), x1)
    return g

def gram_matrix_val(x, area, depth):
    x1 = x.reshape(area,depth)
    g = np.dot(x1.T, x1)
    return g

def build_style_loss(a, x):
    M = a.shape[1]*a.shape[2]
    N = a.shape[3]
    A = gram_matrix_val(a, M, N )
    #A,a is style image
    G = gram_matrix(x, M, N )
    loss = (1./(4 * N**2 * M**2)) * tf.reduce_sum(tf.pow((G - A),2))
    return loss

def gram_xcc(x,pixel_shift = 1):
    shape = tf.shape(x)
    x_positive_shift = tf.gather(x,tf.range(0, shape[1]-pixel_shift),axis = 1)
    x_negative_shift = tf.gather(x,tf.range(pixel_shift, shape[1]),axis = 1)
    shape = tf.shape(x_positive_shift)
    psi1 = tf.reshape(x_positive_shift, (shape[1] * shape[2], shape[3]))
    psi2 = tf.reshape(x_negative_shift, (shape[1] * shape[2], shape[3]))
    g = tf.matmul(tf.transpose(psi1), psi2)
    #calculate the gram loss of x_cc 
    return g

def gram_xcc_val(x,pixel_shift = 1):
    shape = x.shape
    x_positive_shift = x[:,0:shape[1]-pixel_shift,:,:]
    x_negative_shift = x[:,pixel_shift:shape[1],:,:]
    shape = x_positive_shift.shape
    psi1 = x_positive_shift.reshape(shape[1] * shape[2], shape[3])
    psi2 = x_negative_shift.reshape(shape[1] * shape[2], shape[3])
    g = np.dot(psi1.T, psi2)
    #calculate the gram loss of x_cc of style image
    return g

def gram_ycc(x,pixel_shift = 1):
    shape = tf.shape(x)
    x_positive_shift = tf.gather(x,tf.range(0, shape[2]-pixel_shift),axis = 2)
    x_negative_shift = tf.gather(x,tf.range(pixel_shift, shape[2]),axis = 2)
    shape = tf.shape(x_positive_shift)
    psi1 = tf.reshape(x_positive_shift, (shape[1] * shape[2], shape[3]))
    psi2 = tf.reshape(x_negative_shift, (shape[1] * shape[2], shape[3]))
    g = tf.matmul(tf.transpose(psi1), psi2)
    return g

def gram_ycc_val(x,pixel_shift = 1):
    shape = x.shape
    x_positive_shift = x[:,:,0:shape[2]-pixel_shift,:]
    x_negative_shift = x[:,:,pixel_shift:shape[2],:]
    shape = x_positive_shift.shape
    psi1 = x_positive_shift.reshape(shape[1] * shape[2], shape[3])
    psi2 = x_negative_shift.reshape(shape[1] * shape[2], shape[3])
    g = np.dot(psi1.T, psi2)
    return g

def build_cc_loss(a, x, pixel_shift):
    M = a.shape[1]*(a.shape[2] - pixel_shift)
    N = a.shape[3]
    x_A = gram_xcc_val(a,pixel_shift)
    x_G = gram_xcc(x,pixel_shift)
    x_loss = (1./(4 * N**2 * M**2)) * tf.reduce_sum(tf.pow((x_G - x_A),2))
    y_A = gram_ycc_val(a,pixel_shift)
    #A,a为风格图片这样子
    y_G = gram_ycc(x,pixel_shift)
    y_loss = (1./(4 * N**2 * M**2)) * tf.reduce_sum(tf.pow((y_G - y_A),2))
    s_loss = tf.multiply(0.5, tf.add(x_loss, y_loss))
    return s_loss

def buid_cc_loss_lays(STYLE_LAYERS, net, sess):
    cc_lost_lay = []
    for l in STYLE_LAYERS:
        if(l[0] == 'conv1_1'):
            #a2 = l[1]*build_cc_loss(sess.run(net[l[0]]) ,  net[l[0]],pixel_shift = 2)
            a4 = l[1]*build_cc_loss(sess.run(net[l[0]]) ,  net[l[0]],pixel_shift = 4)
            a8 = l[1]*build_cc_loss(sess.run(net[l[0]]) ,  net[l[0]],pixel_shift = 8)
            a16 = l[1]*build_cc_loss(sess.run(net[l[0]]) ,  net[l[0]],pixel_shift = 16)
            a32 = l[1]*build_cc_loss(sess.run(net[l[0]]) ,  net[l[0]],pixel_shift = 32)
            a64 = l[1]*build_cc_loss(sess.run(net[l[0]]) ,  net[l[0]],pixel_shift = 64)
            #a = tf.multiply(tf.add_n([a2,a4,a8,a16,a32]),0.2)
            a = tf.multiply(tf.add_n([a4,a8,a16,a32,a64]),0.2)
            cc_lost_lay.append(a)
        if(l[0] == 'conv2_1'):
            #a2 = l[1]*build_cc_loss(sess.run(net[l[0]]) ,  net[l[0]],pixel_shift = 2)
            a4 = l[1]*build_cc_loss(sess.run(net[l[0]]) ,  net[l[0]],pixel_shift = 4)
            a8 = l[1]*build_cc_loss(sess.run(net[l[0]]) ,  net[l[0]],pixel_shift = 8)
            a16 = l[1]*build_cc_loss(sess.run(net[l[0]]) ,  net[l[0]],pixel_shift = 16)
            a32 = l[1]*build_cc_loss(sess.run(net[l[0]]) ,  net[l[0]],pixel_shift = 32)
            #a = tf.multiply(tf.add_n([a2,a4,a8,a16]),0.25)
            a = tf.multiply(tf.add_n([a4,a8,a16,a32]),0.25)
            cc_lost_lay.append(a)
        if(l[0] == 'conv3_1'):
            #a2 = l[1]*build_cc_loss(sess.run(net[l[0]]) ,  net[l[0]],pixel_shift = 2)
            a4 = l[1]*build_cc_loss(sess.run(net[l[0]]) ,  net[l[0]],pixel_shift = 4)
            a8 = l[1]*build_cc_loss(sess.run(net[l[0]]) ,  net[l[0]],pixel_shift = 8)
            a16 = l[1]*build_cc_loss(sess.run(net[l[0]]) ,  net[l[0]],pixel_shift = 16)
            #a = tf.multiply(tf.add_n([a2,a4,a8]),0.33)
            a = tf.multiply(tf.add_n([a4,a8,a16]),0.33)
            cc_lost_lay.append(a)
        if(l[0] == 'conv4_1'):
            #a2 = l[1]*build_cc_loss(sess.run(net[l[0]]) ,  net[l[0]],pixel_shift = 2)
            a4 = l[1]*build_cc_loss(sess.run(net[l[0]]) ,  net[l[0]],pixel_shift = 4)
            a8 = l[1]*build_cc_loss(sess.run(net[l[0]]) ,  net[l[0]],pixel_shift = 8)
            #a = tf.multiply(tf.add_n([a2,a4]),0.5)
            a = tf.multiply(tf.add_n([a4,a8]),0.5)
            cc_lost_lay.append(a)
            #对于每一层，要分别计算cc_loss，每一层的loss保存在cc_lost_lay中
    cost_cc = sum(cc_lost_lay) /4.0
    return cost_cc