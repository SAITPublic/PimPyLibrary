project(CUST_OP)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(PYTHON_BIN_PATH "python3")
set(ROCM_PATH $ENV{ROCM_PATH})

execute_process(COMMAND python3 -c
                "import tensorflow as tf; print(' '.join(tf.sysconfig.get_link_flags()), end='')"
                 OUTPUT_VARIABLE TF_LFLAGS)
execute_process(COMMAND python3 -c
                "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()), end='')"
                OUTPUT_VARIABLE TF_CFLAGS)

set(CMAKE_CXX_COMPILER "${ROCM_PATH}/bin/hipcc")


set(PIM_RELU_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cc/kernels/pim_relu.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/cc/ops/pim_relu_ops.cc)

set(PIM_ELT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cc/kernels/pim_eltwise.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/cc/ops/pim_eltwise_ops.cc)

set(PIM_DENSE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cc/kernels/pim_dense.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/cc/ops/pim_dense_ops.cc)

set(PIM_INIT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cc/kernels/pim_init.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/cc/ops/pim_init_ops.cc)

set(PIM_DEINIT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/cc/kernels/pim_deinit.cc
	    ${CMAKE_CURRENT_SOURCE_DIR}/cc/ops/pim_deinit_ops.cc)

include_directories("${ROCM_PATH}/include")
include_directories(/usr/local/lib/python3.6/dist-packages/tensorflow/include)
link_directories("${ROCM_PATH}/hip/lib" "${ROCM_PATH}/hsa/lib" "${ROCM_PATH}/libhsakmt/lib" "${ROCM_PATH}/lib" "${PROJECT_SOURCE_DIR}/external_libs")
add_library(pim_tf SHARED ${PIM_RELU_SOURCES} ${PIM_ELT_SOURCES} ${PIM_DENSE_SOURCES} ${PIM_INIT_SOURCES} ${PIM_DEINIT_SOURCES})
target_link_libraries(pim_tf PimRuntime ${TF_LFLAGS})

#install ( TARGETS pim_tf LIBRARY DESTINATION  ${ROCM_PATH}/lib)
#install ( DIRECTORY python DESTINATION ${ROCM_PATH}/lib/tf_pim_ops )
#install ( FILES ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py  DESTINATION ${ROCM_PATH}/lib/tf_pim_ops)
