cmake_minimum_required (VERSION 2.6) 
 
set (PROJECT LMTrainer) 
set (PROJECT_DIR ${CMAKE_CURRENT_LIST_DIR})
message ("CURRENT DIR: ${PROJECT_DIR}")
project (${PROJECT}) 
 
#set (HEADERS fdls.h)
 
#set (SOURCES csdfs.cpp) 

file (GLOB_RECURSE all_headers "${PROJECT_DIR}/src/*.h")
message("My headers: " ${all_headers})
file (GLOB_RECURSE all_sources "${PROJECT_DIR}/src/*.cpp")
message("My sources: " ${all_sources})

# FANN processing
include_directories (${PROJECT_DIR}/lib/fann/headers)
#add_subdirectory (lib/fann/compiled/ bin/doublefann)
#find_library ("${PROJECT_DIR}/lib/fann/compiled/libdoublefann.so" doublefann)
# Your-external "mylib"
add_library( doublefann SHARED IMPORTED )
# You can define two import-locations: one for debug and one for release.
set_target_properties( doublefann PROPERTIES IMPORTED_LOCATION ${PROJECT_DIR}/lib/fann/compiled/libdoublefann.so ) 

# Eigen processing
include_directories (lib/eigen/Eigen)

set (EXECUTABLE_OUTPUT_PATH "${PROJECT_DIR}/bin")

set(MY_CXX_FLAGS  "-Wall -std=c++11 -fmessage-length=0 -v -g ")
set(CMAKE_CXX_FLAGS "-O0 ${MY_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fno-reorder-blocks -fno-schedule-insns -fno-inline")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

add_executable (${PROJECT} ${all_headers} ${all_sources})

target_link_libraries (${PROJECT} doublefann)