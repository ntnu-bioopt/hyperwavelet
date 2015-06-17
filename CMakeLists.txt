cmake_minimum_required(VERSION 2.6)
project(wavelet-bin C CXX)


set(BLAS_LIBRARIES mkl_intel_lp64 mkl_intel_thread mkl_core iomp5)

#pthreads
find_package(Threads)

set(CMAKE_BUILD_TYPE Release)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

find_package(CUDA)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_11 -Xptxas -v)
cuda_compile(CUDA_O src/gpuwavelet.cu)
cuda_add_executable(wavelet-bin ${CUDA_O} src/wavelets.cpp src/main.cpp src/readimage.cpp)
target_link_libraries(wavelet-bin ${BLAS_LIBRARIES} cublas)

add_library(hyperwavelet src/wavelets.cpp ${CUDA_O})

#install
install (TARGETS wavelet-bin DESTINATION bin)
install (TARGETS hyperwavelet DESTINATION lib)
install (FILES include/wavelets.h DESTINATION hyperwavelet/include)
