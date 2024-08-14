find_package(ONEMKL)
if(NOT ONEMKL_FOUND)
  message(FATAL_ERROR "Can NOT find ONEMKL cmake helpers module!")
endif()

set(TORCH_XPU_OPS_ONEMKL_LIBRARIES ${ONEMKL_LIBRARIES})

list(INSERT TORCH_XPU_OPS_ONEMKL_LIBRARIES 0 "-Wl,--no-as-needed")
list(APPEND TORCH_XPU_OPS_ONEMKL_LIBRARIES "-Wl,--as-needed")
