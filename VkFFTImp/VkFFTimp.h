#ifndef VKFFT_IMP_H
#define VKFFT_IMP_H

#include <vector>
#include <memory>
#include <string.h>
#include <chrono>
#include <thread>
#include <iostream>
#include <algorithm>
#define __STDC_FORMAT_MACROS
#include <inttypes.h>

#include <vulkan/vulkan.h>
#include "half.hpp"
#include "benchmark_scripts/vkFFT_scripts/include/user_benchmark_VkFFT.h"

namespace Calyo
{

	struct VkFFTData
	{
		VkGPU vkGPU_;
		VkBuffer * buffer_;
		VkDeviceMemory bufferDeviceMemory_;

		unsigned int num_batches_;
		unsigned int fft_size_;

		VkFFTApplication app_;

	};



	// struct VkCalyoMgt
	// {

	// 	VkFFTData fft_data_;
	// 	Calyo::VkCalyoStatus status_;
	// 	Calyo::VkCalyoModules initial_module_;
	// };

	// int upload_data_to_GPU(VkCalyoMgt * p_data, VkFFTData * P_fft_data, void * input_buffer, uint64_t buffer_size);

	// int download_data_from_GPU(VkCalyoMgt * p_data, VkFFTData * P_fft_data, void * output_buffer, uint64_t buffer_size);

	int initVkFFT(VkFFTData * p_data, VkFFTUserSystemParameters* userParams, bool should_allocate_buffer = true);

	// int initVkFFT(VkCalyoMgt * p_calyoGPU, VkFFTData * p_data, VkFFTUserSystemParameters* userParams, bool should_allocate_buffer = true);

	VkFFTResult run(VkFFTData * p_data, float * buffer_input, float * output_buffer);

	int fft(VkFFTData * p_data);

	int ifft(VkFFTData * p_data);

	// clean up
	void destroytVkFFT(VkFFTData* p_data);
	// example main
	int test_vkfft();
}
#endif