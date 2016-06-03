set ("${PROJECT}_VERSION_MAJ" 0)
set ("${PROJECT}_VERSION_MIN" 1)
set ("${PROJECT}_VERSION_A" 1)
set ("${PROJECT}_VERSION_B" 1)
set ("${PROJECT}_VERSION" ${${PROJECT}_VERSION_MAJ}0${${PROJECT}_VERSION_MIN}0${${PROJECT}_VERSION_A}0${${PROJECT}_VERSION_B})

message(STATUS ${${PROJECT}_VERSION})

# основной файл программы
set (MAIN_SOURCES
    fann_cpp_test.cpp
    )

# непубличные пары исходников
set (PRIVATE_CLASSES
        ann_wrappers/FannAnnWrapper
#        ann_wrappers/CustomAnnWrapper

	data_wrappers/FannDataWrapper
	
	testers/FannTester

	trainer/LMTrainer

	matrix_wrappers/EigenBackend
 #       ann_wrappers/AnnWrapper
    )

# файлы только заголовочные, у которых нет пары-исходника
SET (HEADERS_ONLY
        types
        common
        ann_wrappers/AnnWrapper
	matrix_wrappers/Mtx
	data_wrappers/DataWrapper
	matrix_wrappers/MatrixBackend
    )

# публичные исходники
set (PUBLIC_CLASSES)

# используемые в программе библиотеки
set (ADDITIONAL_LIBRARIES
  doublefann
    )

set (PUBLIC_HEADERS)
set (SOURCES)

foreach (class ${PRIVATE_CLASSES})
    LIST (APPEND SOURCES ${class}.cpp)
    LIST (APPEND HEADERS ${class}.hpp)
endforeach ()

foreach (class ${HEADERS_ONLY})
    LIST (APPEND HEADERS ${class}.hpp)
endforeach ()
    
foreach (class ${PUBLIC_CLASSES})
    LIST (APPEND SOURCES ${class}.cpp)
    LIST (APPEND HEADERS ../include/${PROJECT}/${class}.hpp)
    LIST (APPEND PUBLIC_HEADERS ../include/${PROJECT}/${class}.hpp)
endforeach ()

add_executable (${PROJECT} ${MAIN_SOURCES} ${SOURCES})

target_link_libraries (${PROJECT} ${ADDITIONAL_LIBRARIES})

set_target_properties(${PROJECT} PROPERTIES VERSION "${${PROJECT}_VERSION}" SOVERSION "0")

INSTALL (
    TARGETS
    ${PROJECT}
    DESTINATION
    lib${LIB_SUFFIX}
)

INSTALL (
    FILES
    ${PUBLIC_HEADERS}
    DESTINATION
    include/${PROJECT}
)


# eigen setup
SET( EIGEN3_INCLUDE_DIR "eigen/cmake" )
IF( NOT EIGEN3_INCLUDE_DIR )
    MESSAGE( FATAL_ERROR "Please point the environment variable EIGEN3_INCLUDE_DIR to the include directory of your Eigen3 installation.")
ENDIF()
INCLUDE_DIRECTORIES ( "${EIGEN3_INCLUDE_DIR}" )