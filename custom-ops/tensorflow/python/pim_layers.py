import tensorflow as tf
from tensorflow.keras import initializers
from pim_tf.pim_tf_ops import pim_dense

class PimDense(tf.keras.layers.Dense):
    def __init__(self,
               units,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs
              ):
        super(PimDense, self).__init__( units,
               activation,
               use_bias,
               kernel_initializer,
               bias_initializer,
               kernel_regularizer,
               bias_regularizer,
               activity_regularizer,
               kernel_constraint,
               bias_constraint,
               **kwargs)

    def build(self, input_shape):
        ret =  super(PimDense, self).build(input_shape)
        weights = super(PimDense, self).get_weights()
        self.kernel = weights[0]
        if len(weights) > 1:
            self.bias = weights[1]
            self.has_bias = tf.constant([1])
        else:
            self.has_bias = tf.constant([0])
        return ret

    def compute_output_shape(self, input_shape):
        return super(PimDense, self).compute_output_shape(input_shape)

    def get_config(self):
        return super(PimDense, self).get_config()

    def call(self, input):
        use_pim_bias = tf.constant([0])
        x = pim_dense(input, self.kernel, self.bias, use_pim_bias)
        print('Kernel' , self.kernel)
        if self.has_bias:
           return x + self.bias
        return x 

    def set_weights(self,weights):
        self.kernel = weights[0]
        if len(weights) > 1:
            self.bias = weights[1]
