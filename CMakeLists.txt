cmake_minimum_required(VERSION 3.18)
project(cuda_linalg LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

file(GLOB_RECURSE LINALG_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/linalg/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/linalg/*.cuh"
)

set(MAIN_SOURCE main.cu)

add_executable(${PROJECT_NAME} ${MAIN_SOURCE} ${LINALG_SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/linalg)
