cmake_minimum_required(VERSION 3.11)

project(VkfftLib)

if (MSVC)

	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

add_library(${PROJECT_NAME} OBJECT VkFFTImp/VkFFTimp.cpp
		benchmark_scripts/vkFFT_scripts/src/utils_VkFFT.cpp)


find_package(Vulkan REQUIRED)
# target_include_directories(${PROJECT_NAME} PUBLIC ${Vulkan_INCLUDE_DIRS} glslang-main/glslang/Include)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)  

# note that it is not CMAKE_INSTALL_PREFIX we are checking here
if(DEFINED CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    message(
        STATUS
        "CMAKE_INSTALL_PREFIX is not set\n"
        "Default value: ${CMAKE_INSTALL_PREFIX}\n"
        "Will set it to ${CMAKE_SOURCE_DIR}/install"
    )
    set(CMAKE_INSTALL_PREFIX
        "${CMAKE_SOURCE_DIR}/install"
        CACHE PATH "Where the library will be installed to" FORCE
    )
else()
    message(
        STATUS
        "CMAKE_INSTALL_PREFIX was already set\n"
        "Current value: ${CMAKE_INSTALL_PREFIX}"
    )
endif()

include(GNUInstallDirs)

add_library(half INTERFACE)

target_include_directories(half
INTERFACE
    # where the library itself will look for its internal headers
    ${CMAKE_CURRENT_SOURCE_DIR}/half_lib/
	INTERFACE
    # where top-level project will look for the library's public headers
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/half_lib/
    # where external projects will look for the library's public headers
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

set(half_headers ${CMAKE_CURRENT_SOURCE_DIR}/half_lib/half.hpp)
set_target_properties(half PROPERTIES PUBLIC_HEADER "${half_headers}")

# install the target and create export-set
install(TARGETS half
    EXPORT "engineTargets"
    # these get default values from GNUInstallDirs, no need to set them
    #RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} # bin
    #LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} # lib
    #ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} # lib
    # except for public headers, as we want them to be inside a library folder
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/half # include/SomeLibrary
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} # include
)

add_library(VkFFT INTERFACE)

target_link_libraries(${PROJECT_NAME} PUBLIC VkFFT half)

target_include_directories(${PROJECT_NAME}
PUBLIC
    # where the library itself will look for its internal headers
    ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_scripts/vkFFT_scripts/include/ ${CMAKE_CURRENT_SOURCE_DIR}/vkFFT/ ${Vulkan_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/half_lib/ ${Vulkan_INCLUDE_DIRS}
PUBLIC
    # where top-level project will look for the library's public headers
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/VkFFTImp/> glslang-main/glslang/Include
    # where external projects will look for the library's public headers
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

file(GLOB_RECURSE public_headers
"${CMAKE_CURRENT_SOURCE_DIR}/vkFFT/*.h"
"${CMAKE_CURRENT_SOURCE_DIR}/VkFFTImp/*.h"
"${CMAKE_CURRENT_SOURCE_DIR}/benchmark_scripts/vkFFT_scripts/include/utils_VkFFT.h"
)

set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${public_headers}")

# install the target and create export-set
install(TARGETS ${PROJECT_NAME}
    EXPORT "engineTargets"
    # these get default values from GNUInstallDirs, no need to set them
    #RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} # bin
    #LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} # lib
    #ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} # lib
    # except for public headers, as we want them to be inside a library folder
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME} # include/SomeLibrary
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} # include
)