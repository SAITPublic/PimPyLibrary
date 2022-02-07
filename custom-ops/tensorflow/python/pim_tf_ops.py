from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

import site
path = site.getusersitepackages()
pim_tf_ops  = load_library.load_op_library((path+'/libpim_tf.so'))
pim_relu    = pim_tf_ops.pim_relu
pim_dense   = pim_tf_ops.pim_dense
pim_eltwise = pim_tf_ops.pim_eltwise
pim_init    = pim_tf_ops.pim_init
pim_deinit  = pim_tf_ops.pim_deinit
