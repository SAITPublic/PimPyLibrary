from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pim_api
import tensorflow as tf
import pim_tf as tf_pim_ops


tf.debugging.set_log_device_placement(True)

class PimMulTest(tf.test.TestCase):
    def test_mul_constant(self):
      with tf.device('/GPU:0'):
        input0 = tf.constant([1]*32, dtype=tf.float16)
        input1 = tf.constant([2]*32, dtype=tf.float16)
        mul = tf.constant([1], dtype=tf.int32)
        result = None
        with self.test_session():
            result = tf_pim_ops.pim_eltwise(input0, input1, mul)
            self.assertAllEqual(result, [2]*32)
            #print(result)

    def test_mul_random(self):
      with tf.device('/GPU:0'):
        input0 = tf.random.uniform(shape=[1,1024], dtype=tf.float16)
        input1 = tf.random.uniform(shape=[1,1024], dtype=tf.float16)
        mul = tf.constant([1], dtype=tf.int32)
        result = None
        with self.test_session():
            result = tf_pim_ops.pim_eltwise(input0, input1, mul)
            golden = tf.math.multiply(input0,input1)
            self.assertAllClose(result, golden,atol=0.01)
            #print(result)

if __name__ == '__main__':
    #pim_api.PimInitialize(pim_api.RT_TYPE_HIP, pim_api.PIM_FP16)
    tf_pim_ops.pim_init()
    tf.test.main()
    tf_pim_ops.pim_deinit()

