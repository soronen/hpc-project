#pragma once

#ifdef __APPLE__
#define CL_TARGET_OPENCL_VERSION 120
#include <OpenCL/opencl.h>
#else
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#endif

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <string>
#include <vector>

struct OpenCLContext
{
    cl_context       context      = nullptr;
    cl_command_queue commandQueue = nullptr;
    cl_program       program      = nullptr;
    cl_device_id     deviceId       = nullptr;
};

// Returns a string describing the given OpenCL error code
std::string getCLErrorString(cl_int error);

// Reads the entire contents of a text file (e.g., *.cl) into a std::string
std::string readFile(const std::string& filePath);

// Create a context and command queue for OpenCL 1.2
OpenCLContext createOpenCLContext(int platformIndex, int deviceIndex, cl_device_type deviceType);

// Build (compile) an OpenCL 1.2 program from source
void buildOpenCLProgram(OpenCLContext& clCtx, const std::string& source, const char* buildOptions = nullptr);

// Create a kernel by name
cl_kernel createKernel(const OpenCLContext& clCtx, const std::string& kernelName);

// (Optional) print platform information
void printPlatformInfo();

// (Optional) print device information
void printDeviceInfo(cl_platform_id platform, cl_device_id device);


std::vector<OpenCLContext> initializeOpenCLDevices();

cl_program buildProgramForDevice(cl_context context, const std::string& source, cl_device_id device);