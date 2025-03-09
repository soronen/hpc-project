#pragma once

#ifdef __APPLE__
#define CL_TARGET_OPENCL_VERSION 120
#include <OpenCL/opencl.h>
#else
#ifdef USE_LUMI
#define CL_TARGET_OPENCL_VERSION 220
#else
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#endif
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
    cl_context context;
    cl_command_queue commandQueue;
    cl_program program;
    cl_device_id deviceId;
};

struct OpenCLBuffers
{
    cl_mem output_image;
    cl_mem colors;
    cl_mem subframes;
    cl_mem instances;
    cl_mem bvh_nodes;
    cl_mem bvh_links;
    cl_mem mesh_indices;
    cl_mem mesh_pos;
    cl_mem mesh_normal;
    cl_mem mesh_albedo;
    cl_mem mesh_material;
};

std::string getCLErrorString(cl_int error);

std::string readFile(const std::string &filePath);

void buildOpenCLProgram(OpenCLContext &clCtx, const std::string &source, const char *buildOptions = nullptr);

cl_kernel createKernel(const OpenCLContext &clCtx, const std::string &kernelName);

void printPlatformInfo();

std::vector<OpenCLContext> initializeOpenCLDevices();

cl_program buildProgramForDevice(cl_context context, const std::string &source, cl_device_id device);

void cleanup_resources(const OpenCLBuffers buffers, const OpenCLContext &context);
