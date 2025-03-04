#include "opencl_utils.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>

std::string getCLErrorString(cl_int error)
{
    switch (error) {
    case CL_SUCCESS:                          return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND:                 return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE:             return "CL_DEVICE_NOT_AVAILABLE";
    case CL_COMPILER_NOT_AVAILABLE:           return "CL_COMPILER_NOT_AVAILABLE";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:    return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES:                 return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY:               return "CL_OUT_OF_HOST_MEMORY";
    case CL_PROFILING_INFO_NOT_AVAILABLE:     return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case CL_MEM_COPY_OVERLAP:                 return "CL_MEM_COPY_OVERLAP";
    case CL_BUILD_PROGRAM_FAILURE:            return "CL_BUILD_PROGRAM_FAILURE";
    case CL_MAP_FAILURE:                      return "CL_MAP_FAILURE";
    case CL_INVALID_VALUE:                    return "CL_INVALID_VALUE";
    case CL_INVALID_DEVICE_TYPE:              return "CL_INVALID_DEVICE_TYPE";
    case CL_INVALID_PLATFORM:                 return "CL_INVALID_PLATFORM";
    case CL_INVALID_DEVICE:                   return "CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT:                  return "CL_INVALID_CONTEXT";
    case CL_INVALID_QUEUE_PROPERTIES:         return "CL_INVALID_QUEUE_PROPERTIES";
    case CL_INVALID_COMMAND_QUEUE:            return "CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_HOST_PTR:                 return "CL_INVALID_HOST_PTR";
    case CL_INVALID_MEM_OBJECT:               return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:  return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case CL_INVALID_IMAGE_SIZE:               return "CL_INVALID_IMAGE_SIZE";
    case CL_INVALID_SAMPLER:                  return "CL_INVALID_SAMPLER";
    case CL_INVALID_BINARY:                   return "CL_INVALID_BINARY";
    case CL_INVALID_BUILD_OPTIONS:            return "CL_INVALID_BUILD_OPTIONS";
    case CL_INVALID_PROGRAM:                  return "CL_INVALID_PROGRAM";
    case CL_INVALID_PROGRAM_EXECUTABLE:       return "CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_KERNEL_NAME:              return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_KERNEL_DEFINITION:        return "CL_INVALID_KERNEL_DEFINITION";
    case CL_INVALID_KERNEL:                   return "CL_INVALID_KERNEL";
    case CL_INVALID_ARG_INDEX:                return "CL_INVALID_ARG_INDEX";
    case CL_INVALID_ARG_VALUE:                return "CL_INVALID_ARG_VALUE";
    case CL_INVALID_ARG_SIZE:                 return "CL_INVALID_ARG_SIZE";
    case CL_INVALID_KERNEL_ARGS:              return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_WORK_DIMENSION:           return "CL_INVALID_WORK_DIMENSION";
    case CL_INVALID_WORK_GROUP_SIZE:          return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_ITEM_SIZE:           return "CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_GLOBAL_OFFSET:            return "CL_INVALID_GLOBAL_OFFSET";
    case CL_INVALID_EVENT_WAIT_LIST:          return "CL_INVALID_EVENT_WAIT_LIST";
    case CL_INVALID_EVENT:                    return "CL_INVALID_EVENT";
    case CL_INVALID_OPERATION:                return "CL_INVALID_OPERATION";
    case CL_INVALID_GL_OBJECT:                return "CL_INVALID_GL_OBJECT";
    case CL_INVALID_BUFFER_SIZE:              return "CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_MIP_LEVEL:                return "CL_INVALID_MIP_LEVEL";
    default:                                  return "Unknown OpenCL error";
    }
}

std::string readFile(const std::string& filePath)
{
    std::ifstream file(filePath, std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filePath);
    }

    std::ostringstream oss;
    oss << file.rdbuf();
    return oss.str();
}

OpenCLContext createOpenCLContext(int platformIndex, int deviceIndex, cl_device_type deviceType)
{
    OpenCLContext clCtx;
    cl_int status = 0;

    // 1) Get all platforms
    cl_uint numPlatforms = 0;
    status = clGetPlatformIDs(0, nullptr, &numPlatforms);
    if (status != CL_SUCCESS || numPlatforms == 0) {
        throw std::runtime_error("Failed to get OpenCL platforms: " + getCLErrorString(status));
    }
    std::vector<cl_platform_id> platforms(numPlatforms);
    status = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    if (status != CL_SUCCESS) {
        throw std::runtime_error("Failed to get platform IDs: " + getCLErrorString(status));
    }

    // Ensure requested platformIndex is valid
    if (platformIndex < 0 || static_cast<cl_uint>(platformIndex) >= numPlatforms) {
        throw std::runtime_error("Invalid platformIndex requested.");
    }
    cl_platform_id chosenPlatform = platforms[platformIndex];

    // 2) Get devices of the chosen type on that platform
    cl_uint numDevices = 0;
    status = clGetDeviceIDs(chosenPlatform, deviceType, 0, nullptr, &numDevices);
    if (status != CL_SUCCESS || numDevices == 0) {
        throw std::runtime_error("Failed to get devices: " + getCLErrorString(status));
    }
    std::vector<cl_device_id> devices(numDevices);
    status = clGetDeviceIDs(chosenPlatform, deviceType, numDevices, devices.data(), nullptr);
    if (status != CL_SUCCESS) {
        throw std::runtime_error("Failed to get device IDs: " + getCLErrorString(status));
    }

    // Ensure requested deviceIndex is valid
    if (deviceIndex < 0 || static_cast<cl_uint>(deviceIndex) >= numDevices) {
        throw std::runtime_error("Invalid deviceIndex requested.");
    }

    clCtx.deviceId = devices[deviceIndex];

    // 3) Create an OpenCL 1.2 context
    clCtx.context = clCreateContext(nullptr, 1, &clCtx.deviceId, nullptr, nullptr, &status);
    if (status != CL_SUCCESS) {
        throw std::runtime_error("Failed to create OpenCL context: " + getCLErrorString(status));
    }

    // 4) Create a command queue (OpenCL 1.2 style)
    clCtx.commandQueue = clCreateCommandQueue(clCtx.context, clCtx.deviceId, 0, &status);
    if (status != CL_SUCCESS) {
        throw std::runtime_error("Failed to create command queue: " + getCLErrorString(status));
    }

    return clCtx;
}

void buildOpenCLProgram(OpenCLContext& clCtx, const std::string& source, const char* buildOptions)
{
    cl_int status = 0;
    const char* sourceStr = source.c_str();
    size_t sourceSize = source.size();

    // Create program from source
    clCtx.program = clCreateProgramWithSource(clCtx.context, 1, &sourceStr, &sourceSize, &status);
    if (status != CL_SUCCESS) {
        throw std::runtime_error("Failed to create program from source: " + getCLErrorString(status));
    }

    // Build (compile) the program
    status = clBuildProgram(clCtx.program, 1, &clCtx.deviceId, buildOptions, nullptr, nullptr);
    if (status != CL_SUCCESS) {
        // Retrieve build log for diagnosis
        size_t logSize = 0;
        clGetProgramBuildInfo(clCtx.program, clCtx.deviceId, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);

        std::vector<char> buildLog(logSize);
        clGetProgramBuildInfo(clCtx.program, clCtx.deviceId, CL_PROGRAM_BUILD_LOG, logSize, buildLog.data(), nullptr);

        std::string logString(buildLog.begin(), buildLog.end());
        std::ostringstream oss;
        oss << "Failed to build program: " << getCLErrorString(status) 
            << "\nBuild log:\n" << logString;
        throw std::runtime_error(oss.str());
    }
}

cl_kernel createKernel(const OpenCLContext& clCtx, const std::string& kernelName)
{
    cl_int status = 0;
    cl_kernel kernel = clCreateKernel(clCtx.program, kernelName.c_str(), &status);
    if (status != CL_SUCCESS) {
        throw std::runtime_error("Failed to create kernel " + kernelName + 
                                 ": " + getCLErrorString(status));
    }
    return kernel;
}

// Optional Info
void printPlatformInfo()
{
    cl_uint numPlatforms = 0;
    cl_int status = clGetPlatformIDs(0, nullptr, &numPlatforms);
    if (status != CL_SUCCESS) {
        std::cerr << "Error getting number of platforms: " << getCLErrorString(status) << std::endl;
        return;
    }

    std::vector<cl_platform_id> platforms(numPlatforms);
    status = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    if (status != CL_SUCCESS) {
        std::cerr << "Error getting platform IDs: " << getCLErrorString(status) << std::endl;
        return;
    }

    for (cl_uint i = 0; i < numPlatforms; ++i) {
        size_t infoLen = 0;
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, nullptr, &infoLen);
        std::vector<char> infoStr(infoLen);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, infoLen, infoStr.data(), nullptr);

        std::cout << "Platform " << i << " Name: " << infoStr.data() << std::endl;
    }
}

void printDeviceInfo(cl_platform_id platform, cl_device_id device)
{
    // Example for device name
    size_t infoLen = 0;
    clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &infoLen);
    std::vector<char> infoStr(infoLen);
    clGetDeviceInfo(device, CL_DEVICE_NAME, infoLen, infoStr.data(), nullptr);

    std::cout << "Device: " << infoStr.data() << " on platform " << platform << std::endl;
}
