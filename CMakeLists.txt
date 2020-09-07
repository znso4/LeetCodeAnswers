cmake_minimum_required(VERSION 3.8)

# use vcpkg
# set(CMAKE_TOOLCHAIN_FILE D:/sdk/vcpkg/scripts/buildsystems/vcpkg.cmake)

project(LeetCode)

set(CMAKE_CXX_STANDARD 17)

# use OpenCV
# set(OpenCV_DIR D:/sdk/opencv)
# set(OpenCV_ARCH "x64")
# set(OpenCV_RUNTIME vc16)
# set(OpenCV_STATIC OFF)
# find_package(OpenCV REQUIRED)

# add source files
aux_source_directory(./src SOURCES)

# add exe
add_executable(main ${SOURCES})

target_precompile_headers(main PUBLIC src/pch.h)
