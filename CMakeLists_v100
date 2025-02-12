cmake_minimum_required(VERSION 3.18)
project(ans_test CUDA CXX)
find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS})
enable_language(CXX)
enable_language(CUDA)


set(CMAKE_CUDA_ARCHITECTURES 70)

# Set V100 architecture
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_70")


project(ans_test CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# Core library files needed
add_library(ans_core STATIC
  GpuANSUtils.cuh
  GpuANSCodec.h
  GpuANSEncode.cuh
  GpuANSDecode.cuh
  GpuANSEncode.cu
  GpuANSDecode.cu
  BatchProvider.cuh
  BatchPrefixSum.cuh
  StackDeviceMemory.h
  DeviceDefs.cuh
  DeviceUtils.h
  DeviceUtils.cpp
  StaticUtils.h
  StackDeviceMemory.cpp

 GpuFloatCodec.h
   GpuFloatUtils.cuh
     GpuFloatCompress.cuh
       GpuFloatDecompress.cuh
         GpuFloatInfo.cuh
	   GpuFloatCompress.cu
	     GpuFloatDecompress.cu
	       GpuFloatInfo.cu

  )

# BFloat16 test executable
add_executable(bfloat16_test bfloat16_test.cpp)
target_link_libraries(bfloat16_test PRIVATE ans_core CUDA::cuda_driver)

# Float16 test executable
add_executable(float16_test float16_test.cpp)
target_link_libraries(float16_test PRIVATE ans_core CUDA::cuda_driver)

# Float32 test executable
add_executable(float32_test float32_test.cpp)
target_link_libraries(float32_test PRIVATE ans_core CUDA::cuda_driver)
