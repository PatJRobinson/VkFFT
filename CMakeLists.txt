cmake_minimum_required(VERSION 3.11)

project(VkfftLib)

if (MSVC)

	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

add_library(${PROJECT_NAME} OBJECT VkFFTImp/VkFFTimp.cpp
		benchmark_scripts/vkFFT_scripts/src/utils_VkFFT.cpp)


find_package(Vulkan REQUIRED)
target_include_directories(${PROJECT_NAME} PUBLIC ${Vulkan_INCLUDE_DIRS} glslang-main/glslang/Include)


target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)  

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/vkFFT/)
add_library(VkFFT INTERFACE)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/half_lib/)
add_library(half INTERFACE)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_scripts/vkFFT_scripts/include/ ${CMAKE_CURRENT_SOURCE_DIR}/VkFFTImp/)
target_include_directories(${PROJECT_NAME} PUBLIC ${Vulkan_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} PUBLIC VkFFT half)
