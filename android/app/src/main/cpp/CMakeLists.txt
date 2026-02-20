cmake_minimum_required(VERSION 3.22.1)

project("mgpu")

# MGPU source directories
set(MGPU_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../..")
set(MGPU_SRC "${MGPU_ROOT}/src")

# Include MGPU engine
include_directories(
    ${MGPU_ROOT}
    ${MGPU_SRC}
    ${MGPU_SRC}/engine
    ${MGPU_SRC}/models
)

# Find OpenCL
find_library(opencl-lib OpenCL)
find_library(log-lib log)

# MGPU engine sources
file(GLOB_RECURSE MGPU_ENGINE_SOURCES
    "${MGPU_SRC}/engine/*.cpp"
    "${MGPU_SRC}/models/*.cpp"
)

# Add MGPU engine
add_library(mgpu_engine STATIC ${MGPU_ENGINE_SOURCES})

# JNI wrapper sources
add_library(mgpu SHARED
    jni_wrapper.cpp
)

# Link libraries
target_link_libraries(mgpu
    mgpu_engine
    ${opencl-lib}
    ${log-lib}
)

# Compiler flags
target_compile_options(mgpu PRIVATE
    -fvisibility=hidden
    -ffunction-sections
    -fdata-sections
)

target_link_options(mgpu PRIVATE
    -Wl,--gc-sections
)
