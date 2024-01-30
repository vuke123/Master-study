set(CMAKE_CXX_STANDARD 17)

cmake_minimum_required(VERSION 3.10)
project(object_detection)

find_package(OpenCV REQUIRED)

add_executable(object_detection objectDetection.cpp)

target_compile_features(object_detection PRIVATE cxx_std_17)

target_include_directories(object_detection PRIVATE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(object_detection PRIVATE ${OpenCV_LIBS})

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/haarcascade_eye_tree_eyeglasses.xml DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/haarcascade_frontalface_alt.xml DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/face.jpg DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
