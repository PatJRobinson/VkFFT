#include "VkFFTimp.h"

namespace Calyo
{

	// int upload_data_to_GPU(VkCalyoMgt * p_data, VkFFTData * p_fft_data, void * input_buffer, uint64_t buffer_size)
	// {
	// 	VkFFTResult resFFT;
	// 	VkBuffer * buf;
	// 	// get buffer
	// 	// switch(p_data->get_initial_module())
	// 	// {
	// 	// 	case Calyo::FFT:
	// 	// 		buf = &p_fft_data->buffer_;
	// 	// 		break;
	// 	// 	case Calyo::IFFT:
	// 	// 		buf = &p_fft_data->buffer_;
	// 	// 		break;
	// 	// 	default:
	// 	// 		printf("Error, line %d: invalid module ID\n", __LINE__);
	// 	// 		return 1;
	// 	// }

	// 	buf = p_data->get_first_buffer_ptr();

	// 	//Sample buffer transfer tool. Uses staging buffer (if needed) of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
	// 	resFFT = transferDataFromCPU(&p_fft_data->vkGPU_, input_buffer, buf, buffer_size);
	// 	if (resFFT != VKFFT_SUCCESS) return 1;

	// 	return 0;
	// }

	// int download_data_from_GPU(VkCalyoMgt * p_data, VkFFTData * p_fft_data, void * output_buffer, uint64_t buffer_size)
	// {
	// 	VkFFTResult resFFT;
	// 	VkBuffer * buf;
	// 	// get buffer
	// 	// switch(p_data->get_status())
	// 	// {
	// 	// 	case Calyo::FFT_SUCCESS:
	// 	// 		buf = &p_fft_data->buffer_;
	// 	// 		break;
	// 	// 	case Calyo::IFFT_SUCCESS:
	// 	// 		buf = &p_fft_data->buffer_;
	// 	// 		break;
	// 	// 	case Calyo::FAILURE:
	// 	// 		printf("Error, line %d: invalid data\n", __LINE__);
	// 	// 		return 1;
	// 	// 	default:
	// 	// 		printf("Error, line %d: invalid status message\n", __LINE__);
	// 	// 		return 1;
	// 	// }

	// 	buf = p_data->get_first_buffer_ptr();

	// 	resFFT = transferDataToCPU(&p_fft_data->vkGPU_, output_buffer, buf, buffer_size);
	// 	if (resFFT != VKFFT_SUCCESS) return 1;

	// 	return 0;
	// }


	// int initVkFFT(VkCalyoMgt * p_calyoGPU, VkFFTData * p_data, VkFFTUserSystemParameters* userParams, bool should_allocate_buffer)
	// {
	// 	p_data->vkGPU_.enableValidationLayers = 0;

	// 	//Sample Vulkan project GPU initialization.
	// 	VkFFTResult resFFT = VKFFT_SUCCESS;

	// 	VkResult res = VK_SUCCESS;
	// 	//create instance - a connection between the application and the Vulkan library 
	// 	p_data->vkGPU_.instance = p_calyoGPU->get_VkInstance();
	// 	//set up the debugging messenger 
	// 	p_data->vkGPU_.debugMessenger = *p_calyoGPU->get_debug_messenger();
	// 	//check if there are GPUs that support Vulkan and select one
	// 	p_data->vkGPU_.physicalDevice = p_calyoGPU->get_physical_device();
	// 	//create logical device representation
	// 	p_data->vkGPU_.device = p_calyoGPU->get_device();
	// 	//create fence for synchronization 
	// 	p_data->vkGPU_.fence = p_calyoGPU->get_fence();
	// 	//create a place, command buffer memory is allocated from
	// 	p_data->vkGPU_.commandPool = p_calyoGPU->get_command_pool();

	// 	p_data->vkGPU_.queue = p_calyoGPU->get_queue();

	// 	// // initialise calyo gpu object
	// 	// if (!p_calyoGPU->is_initialised_)
	// 	// {
	// 	// 	p_calyoGPU->physical_device_ = p_data->vkGPU_.physicalDevice;
	// 	// 	p_calyoGPU->logical_device_ = p_data->vkGPU_.device;
	// 	// 	p_calyoGPU->is_initialised_ = true;
	// 	// }

	// 	vkGetPhysicalDeviceProperties(p_data->vkGPU_.physicalDevice, &p_data->vkGPU_.physicalDeviceProperties);
	// 	vkGetPhysicalDeviceMemoryProperties(p_data->vkGPU_.physicalDevice, &p_data->vkGPU_.physicalDeviceMemoryProperties);

	// 	glslang_initialize_process();//compiler can be initialized before VkFFT
	// 	uint64_t isCompilerInitialized = 1;

	// 	// compile shaders
	// 	// set up descriptors
	// 	//Configuration + FFT application .
	// 	VkFFTConfiguration configuration = {};
	// 	p_data->app_ = {};
	// 	//FFT + iFFT sample code.
	// 	//Setting up FFT configuration for forward and inverse FFT.
	// 	configuration.FFTdim = 1; //FFT dimension, 1D, 2D or 3D (default 1).

	// 	configuration.size[0] = p_data->fft_size_;
	// 	configuration.numberBatches = p_data->num_batches_;//(uint64_t)((64 * 32 * (uint64_t)pow(2, 16)) / configuration.size[0]);
	// 	if (configuration.numberBatches < 1) configuration.numberBatches = 1;

	// 	configuration.saveApplicationToString = 1;

	// 	//After this, configuration file contains pointers to Vulkan objects needed to work with the GPU: VkDevice* device - created device, [uint64_t *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] - allocated GPU memory FFT is performed on. [uint64_t *kernelSize, VkBuffer *kernel, VkDeviceMemory* kernelDeviceMemory] - allocated GPU memory, where kernel for convolution is stored.

	// 	configuration.device = &p_data->vkGPU_.device;

	// 	configuration.queue = &p_data->vkGPU_.queue; //to allocate memory for LUT, we have to pass a queue, &p_data->vkGPU_.fence, commandPool and physicalDevice pointers 
	// 	configuration.fence = &p_data->vkGPU_.fence;
	// 	configuration.commandPool = &p_data->vkGPU_.commandPool;
	// 	configuration.physicalDevice = &p_data->vkGPU_.physicalDevice;
	// 	configuration.isCompilerInitialized = isCompilerInitialized;//compiler can be initialized before VkFFT plan creation. if not, VkFFT will create and destroy one after initialization

	// 	//Allocate buffer for the input data.
	// 	uint64_t bufferSize = (uint64_t)sizeof(float) * 2 * configuration.size[0] * configuration.numberBatches;

	// 	if (should_allocate_buffer)
	// 	{
	// 		resFFT = allocateBuffer(&p_data->vkGPU_, p_data->buffer_, &p_data->bufferDeviceMemory_, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, bufferSize);
	// 		if (resFFT != VKFFT_SUCCESS) return resFFT;
	// 	}

	// 	configuration.buffer = p_data->buffer_;


	// 	configuration.bufferSize = &bufferSize;

	// 	if (configuration.loadApplicationFromString) {
	// 		FILE* kernelCache;
	// 		uint64_t str_len;
	// 		char fname[500];
	// 		int VkFFT_version = VkFFTGetVersion();
	// 		sprintf(fname, "VkFFT_binary");
	// 		kernelCache = fopen(fname, "rb");
	// 		if (!kernelCache) return VKFFT_ERROR_EMPTY_FILE;
	// 		fseek(kernelCache, 0, SEEK_END);
	// 		str_len = ftell(kernelCache);
	// 		fseek(kernelCache, 0, SEEK_SET);
	// 		configuration.loadApplicationString = malloc(str_len);
	// 		fread(configuration.loadApplicationString, str_len, 1, kernelCache);
	// 		fclose(kernelCache);
	// 	}
	// 	//Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
	// 	resFFT = initializeVkFFT(&p_data->app_, configuration);
	// 	if (resFFT != VKFFT_SUCCESS) return resFFT;

	// 	if (configuration.loadApplicationFromString)
	// 		free(configuration.loadApplicationString);

	// 	if (configuration.saveApplicationToString) {
	// 		FILE* kernelCache;
	// 		char fname[500];
	// 		int VkFFT_version = VkFFTGetVersion();
	// 		sprintf(fname, "VkFFT_binary");
	// 		kernelCache = fopen(fname, "wb");
	// 		fwrite(p_data->app_.saveApplicationString, p_data->app_.applicationStringSize, 1, kernelCache);
	// 		fclose(kernelCache);
	// 	}

	// 	return 0;
	// }
	
	int initVkFFT(VkFFTData * p_data, VkFFTUserSystemParameters* userParams, bool should_allocate_buffer)
	{
		//Sample Vulkan project GPU initialization.
		VkFFTResult resFFT = VKFFT_SUCCESS;

		VkResult res = VK_SUCCESS;

		vkGetPhysicalDeviceProperties(p_data->vkGPU_.physicalDevice, &p_data->vkGPU_.physicalDeviceProperties);
		vkGetPhysicalDeviceMemoryProperties(p_data->vkGPU_.physicalDevice, &p_data->vkGPU_.physicalDeviceMemoryProperties);

		glslang_initialize_process();//compiler can be initialized before VkFFT
		uint64_t isCompilerInitialized = 1;

		// compile shaders
		// set up descriptors
		//Configuration + FFT application .
		VkFFTConfiguration configuration = {};
		p_data->app_ = {};
		//FFT + iFFT sample code.
		//Setting up FFT configuration for forward and inverse FFT.
		configuration.FFTdim = 1; //FFT dimension, 1D, 2D or 3D (default 1).

		configuration.size[0] = p_data->fft_size_;
		configuration.numberBatches = p_data->num_batches_;//(uint64_t)((64 * 32 * (uint64_t)pow(2, 16)) / configuration.size[0]);
		if (configuration.numberBatches < 1) configuration.numberBatches = 1;

		configuration.saveApplicationToString = 1;

		//After this, configuration file contains pointers to Vulkan objects needed to work with the GPU: VkDevice* device - created device, [uint64_t *bufferSize, VkBuffer *buffer, VkDeviceMemory* bufferDeviceMemory] - allocated GPU memory FFT is performed on. [uint64_t *kernelSize, VkBuffer *kernel, VkDeviceMemory* kernelDeviceMemory] - allocated GPU memory, where kernel for convolution is stored.

		configuration.device = &p_data->vkGPU_.device;

		configuration.queue = &p_data->vkGPU_.queue; //to allocate memory for LUT, we have to pass a queue, &p_data->vkGPU_.fence, commandPool and physicalDevice pointers 
		configuration.fence = &p_data->vkGPU_.fence;
		configuration.commandPool = &p_data->vkGPU_.commandPool;
		configuration.physicalDevice = &p_data->vkGPU_.physicalDevice;
		configuration.isCompilerInitialized = isCompilerInitialized;//compiler can be initialized before VkFFT plan creation. if not, VkFFT will create and destroy one after initialization

		//Allocate buffer for the input data.
		uint64_t bufferSize = (uint64_t)sizeof(float) * 2 * configuration.size[0] * configuration.numberBatches;

		if (should_allocate_buffer)
		{
			resFFT = allocateBuffer(&p_data->vkGPU_, p_data->buffer_, &p_data->bufferDeviceMemory_, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, bufferSize);
			if (resFFT != VKFFT_SUCCESS) return resFFT;
		}

		configuration.buffer = p_data->buffer_;


		configuration.bufferSize = &bufferSize;

		if (configuration.loadApplicationFromString) {
			FILE* kernelCache;
			uint64_t str_len;
			char fname[500];
			int VkFFT_version = VkFFTGetVersion();
			sprintf(fname, "VkFFT_binary");
			kernelCache = fopen(fname, "rb");
			if (!kernelCache) return VKFFT_ERROR_EMPTY_FILE;
			fseek(kernelCache, 0, SEEK_END);
			str_len = ftell(kernelCache);
			fseek(kernelCache, 0, SEEK_SET);
			configuration.loadApplicationString = malloc(str_len);
			fread(configuration.loadApplicationString, str_len, 1, kernelCache);
			fclose(kernelCache);
		}
		//Initialize applications. This function loads shaders, creates pipeline and configures FFT based on configuration file. No buffer allocations inside VkFFT library.  
		resFFT = initializeVkFFT(&p_data->app_, configuration);
		if (resFFT != VKFFT_SUCCESS) return resFFT;

		if (configuration.loadApplicationFromString)
			free(configuration.loadApplicationString);

		// if (configuration.saveApplicationToString) {
		// 	FILE* kernelCache;
		// 	char fname[500];
		// 	int VkFFT_version = VkFFTGetVersion();
		// 	sprintf(fname, "VkFFT_binary");
		// 	kernelCache = fopen(fname, "wb");
		// 	fwrite(p_data->app_.saveApplicationString, p_data->app_.applicationStringSize, 1, kernelCache);
		// 	fclose(kernelCache);
		// }

		return 0;
	}

	VkFFTResult run(VkFFTData * p_data, float * buffer_input, float * output_buffer)
	{
		VkFFTResult resFFT;
		uint64_t bufferSize = (uint64_t)sizeof(float) * 2 * p_data->fft_size_ * p_data->num_batches_;

		for (uint64_t i = 0; i < bufferSize; i++) {
			buffer_input[i] = i;//(float)(2 * ((float)rand()) / RAND_MAX - 1.0);
		}

		//Sample buffer transfer tool. Uses staging buffer (if needed) of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
		resFFT = transferDataFromCPU(&p_data->vkGPU_, buffer_input, &p_data->buffer_, bufferSize);
		if (resFFT != VKFFT_SUCCESS) return resFFT;

		VkFFTLaunchParams launchParams = {};
		resFFT = performVulkanFFT(&p_data->vkGPU_, &p_data->app_, &launchParams, 1, 1);
		if (resFFT != VKFFT_SUCCESS) return resFFT;

		// resCALYO = apply_bandpass_filter(buffer, filter)

		resFFT = performVulkanFFT(&p_data->vkGPU_, &p_data->app_, &launchParams, -1, 1);
		if (resFFT != VKFFT_SUCCESS) return resFFT;

		resFFT = transferDataToCPU(&p_data->vkGPU_, output_buffer, &p_data->buffer_, bufferSize);
		if (resFFT != VKFFT_SUCCESS) return resFFT;

		std::vector<float> res ((float*)output_buffer, (float*)output_buffer + bufferSize/sizeof(float));

		for (uint64_t i = 0; i < res.size(); i++) {
			res[i] /= powf(2, 14);
		}

		return resFFT;
	}

	int fft(VkFFTData * p_data)
	{
		VkFFTResult resFFT;

		VkFFTLaunchParams launchParams = {};
		resFFT = performVulkanFFT(&p_data->vkGPU_, &p_data->app_, &launchParams, 1, 1);
		if (resFFT != VKFFT_SUCCESS) 
			return 1;

		return 0;
	}

	int ifft(VkFFTData * p_data)
	{
		VkFFTResult resFFT;

		VkFFTLaunchParams launchParams = {};
		resFFT = performVulkanFFT(&p_data->vkGPU_, &p_data->app_, &launchParams, -1, 1);
		if (resFFT != VKFFT_SUCCESS) 
			return 1;

		return 0;
	}


	// clean up
	void destroytVkFFT(VkFFTData* p_data) {

		vkDestroyFence(p_data->vkGPU_.device, p_data->vkGPU_.fence, NULL);
		vkDestroyCommandPool(p_data->vkGPU_.device, p_data->vkGPU_.commandPool, NULL);
		vkDestroyDevice(p_data->vkGPU_.device, NULL);
		DestroyDebugUtilsMessengerEXT(&p_data->vkGPU_, NULL);
		vkDestroyInstance(p_data->vkGPU_.instance, NULL);
		glslang_finalize_process();//destroy compiler after use

	}

	// example main
	int test_vkfft()
	{
		// VkCalyoMgt data = {};
		// data.fft_data_.buffer_ = {};
		// data.fft_data_.bufferDeviceMemory_ = {};

		// data.fft_data_.fft_size_ = 16384;
		// data.fft_data_.num_batches_ = 32;

		// initVkFFT(&data.fft_data_, 0);

		// float* buffer_input = (float*)malloc((uint64_t)4 * 2 * (uint64_t)pow(2, 27));
		// uint64_t bufferSize = (uint64_t)sizeof(float) * 2 * data.fft_data_.fft_size_ * data.fft_data_.num_batches_;

		// for (uint64_t i = 0; i < bufferSize; i++) {
		// 	buffer_input[i] = i;//(float)(2 * ((float)rand()) / RAND_MAX - 1.0);
		// }

		// std::vector<float> in ((float*)buffer_input, (float*)buffer_input + bufferSize/sizeof(float));

		// // output cpu array		
		// void * output_buffer = (void*)malloc(bufferSize);

		// VkFFTResult resFFT;
		// // if (resFFT != VKFFT_SUCCESS) return resFFT;
		// int i = 0;
		// while (i++ < 5)
		// {
		// 	//Sample buffer transfer tool. Uses staging buffer (if needed) of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
		// 	if (upload_data_to_GPU(&data, buffer_input, bufferSize) != 0)
		// 	{
		// 		printf("Error, line %d: something went wrong uploading data to the GPU\n", __LINE__);
		// 		return 1;
		// 	}

		// 	fft(&data.fft_data_);

		// 	// matched filter (in place)

		// 	ifft(&data.fft_data_);

		// 	// tfm (produce image)

		// 	//Sample buffer transfer tool. Uses staging buffer (if needed) of the same size as destination buffer, which can be reduced if transfer is done sequentially in small buffers.
		// 	if (download_data_from_GPU(&data, output_buffer, bufferSize) != 0)
		// 	{
		// 		printf("Error, line %d: something went wrong downloading data from the GPU\n", __LINE__);
		// 		return 1;
		// 	}

		// 	std::vector<float> res ((float*)output_buffer, (float*)output_buffer + bufferSize/sizeof(float));

		// 	for (uint64_t i = 0; i < res.size(); i++) {
		// 		res[i] /= powf(2, 14);
		// 	}

		// 	;

		// }
		
		// destroytVkFFT(&data.fft_data_);

		printf("Error, line %d: this function is not in use\n", __LINE__);
		return 1;
	}
}