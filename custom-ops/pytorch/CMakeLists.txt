project(PY_CUST_OP)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(PYTHON_BIN_PATH "python")
set(ROCM_PATH $ENV{ROCM_PATH})

install ( FILES ${CMAKE_CURRENT_SOURCE_DIR}/pim_eltwise.py  DESTINATION ${ROCM_PATH}/lib/py_pim_ops)
install ( FILES ${CMAKE_CURRENT_SOURCE_DIR}/pim_dense.py  DESTINATION ${ROCM_PATH}/lib/py_pim_ops)
install ( FILES ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py  DESTINATION ${ROCM_PATH}/lib/py_pim_ops)
