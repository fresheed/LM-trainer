cmake_minimum_required (VERSION 2.6) 
 
set (PROJECT LMTrainer) 
set (PROJECT_DIR ${CMAKE_CURRENT_LIST_DIR})
message ("CURRENT DIR: ${PROJECT_DIR}")
project (${PROJECT}) 
 
#set (HEADERS fdls.h)
 
#set (SOURCES csdfs.cpp) 

file (GLOB_RECURSE all_headers "src/*.h")
message("My headers: " ${all_headers})
file (GLOB_RECURSE all_sources "src/*.cpp")
message("My sources: " ${all_sources})

# FANN processing
include_directories (lib/fann/headers)
#add_subdirectory (lib/fann/compiled/ bin/doublefann)
find_library (lib/fann/compiled/libdoublefann.so.2.2.0 doublefann)

# Eigen processing
include_directories (lib/eigen/Eigen)

set (EXECUTABLE_OUTPUT_PATH "${PROJECT_DIR}/bin")

set(MY_CXX_FLAGS  "-Wall -std=c++11 -fmessage-length=0 -v")

add_executable (${PROJECT} ${all_headers} ${all_sources})


target_link_libraries (${PROJECT} LINK_PUBLIC doublefann)