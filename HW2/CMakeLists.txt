cmake_minimum_required(VERSION 3.10)
project(HW2)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MSVC 特定配置
if (MSVC)
  add_compile_options(/utf-8)
  # 静态链接运行时库 (/MT, /MTd)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MTd")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MT")
endif()

# 设置 OpenCV 路径 (沿用 HW1 的配置)
set(OpenCV_DIR "your path") # 修改为实际路径
find_package(OpenCV REQUIRED)

# 定义静态链接宏
add_definitions(-DOPENCV_STATIC)

# 包含头文件
include_directories(${OpenCV_INCLUDE_DIRS})

# 添加可执行文件
add_executable(contour_line_extraction main.cpp)

# 链接 OpenCV 静态库
target_link_libraries(contour_line_extraction ${OpenCV_LIBS})

# 复制图片文件夹到构建目录 (确保 exe 能找到图片)
file(COPY picture DESTINATION ${CMAKE_BINARY_DIR})