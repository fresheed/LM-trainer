cmake_minimum_required (VERSION 2.6) 
 
set (PROJECT LMTrainer) 
set (PROJECT_DIR ${CMAKE_CURRENT_LIST_DIR})
message ("CURRENT DIR: ${PROJECT_DIR}")
project (${PROJECT}) 
 
file (GLOB_RECURSE all_headers "${PROJECT_DIR}/src/*.h")
message("My headers: " ${all_headers})
file (GLOB_RECURSE all_sources "${PROJECT_DIR}/src/*.cpp")
message("My sources: " ${all_sources})

# FANN processing
include_directories (${PROJECT_DIR}/lib/fann/headers)
file (GLOB_RECURSE fann_sources "${PROJECT_DIR}/lib/fann/sources/*.c")
message("Fann sources: " ${fann_sources})
file (GLOB_RECURSE fann_headers "${PROJECT_DIR}/lib/fann/headers/*.h")
message("Fann headers: " ${fann_headers})

# Eigen processing
include_directories (lib/eigen/Eigen)

set (EXECUTABLE_OUTPUT_PATH "${PROJECT_DIR}/bin")
if (UNIX)
  set(MY_CXX_FLAGS  "-Wall -std=c++11 -fmessage-length=0 -v -g ")
  set(CMAKE_CXX_FLAGS "-O0 ${MY_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fno-reorder-blocks -fno-schedule-insns -fno-inline")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
elseif (WIN32)
  message ("windows build; no params at moment ...")
endif ()

add_executable (${PROJECT} ${all_headers} ${all_sources} ${fann_sources})