cmake_minimum_required(VERSION 3.18)
project(MGPU LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Android-specific settings
if(MGPU_ANDROID)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_LIBRARIES "-latomic")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -latomic")

    # Enable FP16 and subgroup extensions
    add_compile_definitions(
        MGPU_ANDROID=1
        __opencl_c_fp16=1
    )
endif()

option(MGPU_ANDROID "Build for Android/Adreno targets" OFF)

# --- Compile flags ---
if(NOT MGPU_ANDROID)
    add_compile_options(-Wall -Wextra -O2)
endif()

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

# --- Tests ---
option(MGPU_BUILD_TESTS "Build unit tests" ON)

if(MGPU_BUILD_TESTS)
    # Try to find GTest
    find_package(GTest QUIET)

    if(NOT GTest_FOUND)
        # Use FetchContent to get GTest
        include(FetchContent)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG v1.14.0
        )
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()

    # Unit tests (no GPU required)
    add_executable(test_gguf tests/test_gguf.cpp tests/test_utils.h)
    target_link_libraries(test_gguf PRIVATE mgpu_engine GTest::gtest GTest::gtest_main)
    target_include_directories(test_gguf PRIVATE tests)

    add_executable(test_tokenizer tests/test_tokenizer.cpp tests/test_utils.h)
    target_link_libraries(test_tokenizer PRIVATE mgpu_engine GTest::gtest GTest::gtest_main)
    target_include_directories(test_tokenizer PRIVATE tests)

    # Integration tests (require GPU)
    add_executable(test_device tests/test_device.cpp)
    target_link_libraries(test_device PRIVATE mgpu_engine GTest::gtest GTest::gtest_main)

    # Add all tests to CTest
    enable_testing()
    add_test(NAME GGUFTest COMMAND test_gguf)
    add_test(NAME TokenizerTest COMMAND test_tokenizer)
    add_test(NAME DeviceTest COMMAND test_device)
endif()
