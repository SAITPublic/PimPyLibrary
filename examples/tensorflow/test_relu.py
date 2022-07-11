from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pim_tf as tf_pim_ops

tf.debugging.set_log_device_placement(True)


class PimReluTest(tf.test.TestCase):
    def test1(self):
        with self.test_session():
            inp = tf.constant([-3., 5., -13., -4., 9., 0.], dtype=tf.float16)
            result = tf_pim_ops.pim_relu(inp)
            self.assertAllEqual(result, [0., 5., 0., 0., 9., 0.])

    def test2(self):
        with self.test_session():
            inp = tf.constant(
                [[-5., -1., 0.], [2., -1., 0.]], dtype=tf.float16)
            result = tf_pim_ops.pim_relu(inp)
            self.assertAllEqual(result, [[0., 0., 0.], [2., 0., 0.]])

    def test3(self):
        with self.test_session():
            inp = tf.random.uniform(shape=[1,(128 * 1024)],minval=-10.0,maxval=10.0, dtype=tf.float16)
            result = tf_pim_ops.pim_relu(inp)
            golden = tf.nn.relu(inp)
            self.assertAllEqual(result, golden)


if __name__ == '__main__':
    tf_pim_ops.pim_init()
    tf.test.main()
    tf_pim_ops.pim_deinit()
