cmake_minimum_required(VERSION 3.18)
project(MGPU LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(MGPU_ANDROID "Build for Android/Adreno targets" OFF)

# --- Compile flags ---
add_compile_options(-Wall -Wextra -O2)

# --- Find OpenCL ---
if(MGPU_ANDROID)
    if(NOT OPENCL_INCLUDE_DIR)
        set(OPENCL_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third_party/OpenCL-Headers")
    endif()
    if(NOT OPENCL_LIB)
        set(OPENCL_LIB "${CMAKE_SOURCE_DIR}/third_party/OpenCL-ICD-Loader/build/libOpenCL.so")
    endif()
    add_library(OpenCL SHARED IMPORTED)
    set_target_properties(OpenCL PROPERTIES
        IMPORTED_LOCATION "${OPENCL_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${OPENCL_INCLUDE_DIR}"
    )
else()
    find_package(OpenCL REQUIRED)
endif()

# --- mgpu_engine (static library) ---
file(GLOB ENGINE_SOURCES src/engine/*.cpp src/models/*.cpp)
add_library(mgpu_engine STATIC ${ENGINE_SOURCES})
target_include_directories(mgpu_engine PUBLIC src)
if(MGPU_ANDROID)
    target_link_libraries(mgpu_engine PUBLIC OpenCL log)
    target_compile_definitions(mgpu_engine PUBLIC MGPU_ANDROID=1)
else()
    target_link_libraries(mgpu_engine PUBLIC OpenCL::OpenCL)
endif()

# --- mgpu_device_info (Phase 0 device info dump) ---
add_executable(mgpu_device_info src/app/device_info.cpp)
target_link_libraries(mgpu_device_info PRIVATE mgpu_engine)

# --- mgpu_cli ---
add_executable(mgpu_cli src/app/main.cpp)
target_link_libraries(mgpu_cli PRIVATE mgpu_engine)

# --- mgpu_bench ---
add_executable(mgpu_bench benchmarks/gemm_bench.cpp)
target_link_libraries(mgpu_bench PRIVATE mgpu_engine)

# --- Install OpenCL kernel files ---
file(GLOB KERNEL_FILES src/kernels/*.cl)
install(FILES ${KERNEL_FILES} DESTINATION share/mgpu/kernels)
