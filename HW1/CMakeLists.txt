cmake_minimum_required(VERSION 3.10)
project(OpenCVProject)
# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# 添加 UTF-8 编码支持（仅适用于 MSVC）
if (MSVC)
  add_compile_options(/utf-8)
endif()
if (MSVC)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
endif()
# 设置 OpenCV 的路径
set(OpenCV_DIR "your path") # 修改为实际路径
find_package(OpenCV REQUIRED)
# 定义静态链接宏
add_definitions(-DOPENCV_STATIC)
# 包含头文件
include_directories(${OpenCV_INCLUDE_DIRS})
# 添加可执行文件
add_executable(main main.cpp)
# 链接 OpenCV 静态库
target_link_libraries(main ${OpenCV_LIBS})
# 复制资源文件
file(COPY resources/TRY.png DESTINATION ${CMAKE_BINARY_DIR}/Debug)