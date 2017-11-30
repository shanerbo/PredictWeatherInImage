import tensorflow as tf
import numpy as np
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib import slim

inputs = np.ones([50,321,321,3],dtype=np.float32)
with slim.arg_scope(resnet_v2.resnet_arg_scope()):
    net = resnet_v2.resnet_v2_50(inputs=inputs, num_classes=None,
                                is_training=False,
                                global_pool=False,
                                output_stride=16)
init_fn = slim.assign_from_checkpoint_fn("resnet_v2_50.ckpt", slim.get_model_variables('resnet_v2_50'))
init = tf.global_variables_initializer()
with tf.Session() as sess:
    init_fn(sess)
    sess.run(init)
    N = sess.run(net)
    print (type(N))
    print ('\n')
