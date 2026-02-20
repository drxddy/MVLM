cmake_minimum_required(VERSION 3.22.1)

project("mgpu")

# MGPU source directories
set(MGPU_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../..")
set(MGPU_SRC "${MGPU_ROOT}/src")
set(THIRD_PARTY "${MGPU_ROOT}/third_party")

# OpenCL headers location
set(OPENCL_INCLUDE_DIR "${THIRD_PARTY}/OpenCL-Headers" CACHE PATH "OpenCL include directory")

# Include MGPU engine
include_directories(
    ${MGPU_ROOT}
    ${MGPU_SRC}
    ${MGPU_SRC}/engine
    ${MGPU_SRC}/models
    ${OPENCL_INCLUDE_DIR}
)

# Find log library
find_library(log-lib log)

# MGPU engine sources - use explicit list to avoid glob issues
set(MGPU_ENGINE_SOURCES
    "${MGPU_SRC}/engine/device.cpp"
    "${MGPU_SRC}/engine/compute.cpp"
    "${MGPU_SRC}/engine/memory.cpp"
    "${MGPU_SRC}/engine/pipeline.cpp"
    "${MGPU_SRC}/engine/profiler.cpp"
    "${MGPU_SRC}/models/gguf_loader.cpp"
    "${MGPU_SRC}/models/tokenizer.cpp"
    "${MGPU_SRC}/models/moondream2.cpp"
)

# Add MGPU engine
add_library(mgpu_engine STATIC ${MGPU_ENGINE_SOURCES})

# OpenCL: try to find pre-built ICD loader from MGPU build, fallback to system
set(OPENCL_ICD_PATH "${MGPU_ROOT}/third_party/OpenCL-ICD-Loader/build/libOpenCL.so")
if(EXISTS ${OPENCL_ICD_PATH})
    message(Using pre-built OpenCL ICD: ${OPENCL_ICD_PATH})
    add_library(OpenCL SHARED IMPORTED)
    set_target_properties(OpenCL PROPERTIES IMPORTED_LOCATION ${OPENCL_ICD_PATH})
    target_link_libraries(mgpu_engine PUBLIC OpenCL log dl)
else()
    # Fallback: use system OpenCL (only works on devices with OpenCL)
    find_library(OPENCL_LIB NAMES OpenCL)
    if(OPENCL_LIB)
        message(Using system OpenCL: ${OPENCL_LIB})
        target_link_libraries(mgpu_engine PUBLIC ${OPENCL_LIB} log dl)
    else()
        message(WARNING "OpenCL not found - linking with dl for runtime loading")
        target_link_libraries(mgpu_engine PUBLIC log dl)
    endif()
endif()

# JNI wrapper sources
add_library(mgpu SHARED
    jni_wrapper.cpp
)

# Link libraries
target_link_libraries(mgpu
    mgpu_engine
    log
    android
    dl
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
