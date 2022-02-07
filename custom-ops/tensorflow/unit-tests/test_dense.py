from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras import initializers
from pim_tf.pim_layers import PimDense
import pim_tf as tf_pim_ops
import time
import timeit

tf.debugging.set_log_device_placement(True)
tf.keras.backend.set_floatx('float16')

eval_time = []

class DenseTest(tf.test.TestCase):
    def testDense_1DRandom_layer(self):
        in_batch = 1
        in_size = 32
        out_size = 32

        a = tf.random.uniform(shape=(in_batch, in_size), dtype=tf.float16)
        b = tf.random.uniform(shape=(in_size, out_size), dtype=tf.float16)
        dense = tf.keras.layers.Dense(out_size, use_bias=True,  bias_initializer='glorot_uniform', dtype=tf.float16)
        pim_dense = PimDense(out_size, use_bias=True,  bias_initializer='glorot_uniform', dtype=tf.float16)

        #dummy run to init weights
        golden = dense(a)
        result = pim_dense(a)
        weights = dense.get_weights()
        pim_dense.set_weights(weights)

        with self.test_session():
            golden = dense(a)
            weights = dense.get_weights()
            #print('Input shape',a.shape)
            #print('Weight shape',weights[0].shape)
            #print('Golden shape',golden.shape)
            #print('bias' , weights[1])
            result = pim_dense(a)
            #print('Result',result,golden)
            self.assertAllClose(result, golden, atol=0.01)


    def testDense_2DRandom_layer(self):
        eval_time.clear()
        in_batch = 1
        in_size = 8
        out_size = 4

        a = tf.ones(shape=(in_batch, in_size*2 , in_size), dtype=tf.float16)
        dense = tf.keras.layers.Dense(out_size, use_bias=True,  bias_initializer='glorot_uniform', dtype=tf.float16)
        pim_dense = PimDense(out_size, use_bias=True, dtype=tf.float16)

        #dummy run to init weights
        golden = dense(a)
        result = pim_dense(a)
        weights = dense.get_weights()
        pim_dense.set_weights(weights)

        with self.test_session():
            golden = dense(a)
            #print('Input shape',a.shape)
            #print('Weight shape',weights[0])
            #print('Golden shape',golden.shape)
            result = pim_dense(a)
            eval_time.append(timeit.timeit(lambda : pim_dense(a), number = 2))
            print(eval_time)
            #print('Result',result)
            #print('Golden',golden)
            self.assertAllClose(result, golden, atol=0.01)


if __name__ == "__main__":
    tf_pim_ops.pim_init()
    tf.test.main()
    tf_pim_ops.pim_deinit()
