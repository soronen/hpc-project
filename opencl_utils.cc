#include "opencl_utils.hh"
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>

std::string getCLErrorString(cl_int error)
{
    switch (error)
    {
    case CL_SUCCESS:
        return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND:
        return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE:
        return "CL_DEVICE_NOT_AVAILABLE";
    case CL_COMPILER_NOT_AVAILABLE:
        return "CL_COMPILER_NOT_AVAILABLE";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
        return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES:
        return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY:
        return "CL_OUT_OF_HOST_MEMORY";
    case CL_PROFILING_INFO_NOT_AVAILABLE:
        return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case CL_MEM_COPY_OVERLAP:
        return "CL_MEM_COPY_OVERLAP";
    case CL_BUILD_PROGRAM_FAILURE:
        return "CL_BUILD_PROGRAM_FAILURE";
    case CL_MAP_FAILURE:
        return "CL_MAP_FAILURE";
    case CL_INVALID_VALUE:
        return "CL_INVALID_VALUE";
    case CL_INVALID_DEVICE_TYPE:
        return "CL_INVALID_DEVICE_TYPE";
    case CL_INVALID_PLATFORM:
        return "CL_INVALID_PLATFORM";
    case CL_INVALID_DEVICE:
        return "CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT:
        return "CL_INVALID_CONTEXT";
    case CL_INVALID_QUEUE_PROPERTIES:
        return "CL_INVALID_QUEUE_PROPERTIES";
    case CL_INVALID_COMMAND_QUEUE:
        return "CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_HOST_PTR:
        return "CL_INVALID_HOST_PTR";
    case CL_INVALID_MEM_OBJECT:
        return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
        return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case CL_INVALID_IMAGE_SIZE:
        return "CL_INVALID_IMAGE_SIZE";
    case CL_INVALID_SAMPLER:
        return "CL_INVALID_SAMPLER";
    case CL_INVALID_BINARY:
        return "CL_INVALID_BINARY";
    case CL_INVALID_BUILD_OPTIONS:
        return "CL_INVALID_BUILD_OPTIONS";
    case CL_INVALID_PROGRAM:
        return "CL_INVALID_PROGRAM";
    case CL_INVALID_PROGRAM_EXECUTABLE:
        return "CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_KERNEL_NAME:
        return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_KERNEL_DEFINITION:
        return "CL_INVALID_KERNEL_DEFINITION";
    case CL_INVALID_KERNEL:
        return "CL_INVALID_KERNEL";
    case CL_INVALID_ARG_INDEX:
        return "CL_INVALID_ARG_INDEX";
    case CL_INVALID_ARG_VALUE:
        return "CL_INVALID_ARG_VALUE";
    case CL_INVALID_ARG_SIZE:
        return "CL_INVALID_ARG_SIZE";
    case CL_INVALID_KERNEL_ARGS:
        return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_WORK_DIMENSION:
        return "CL_INVALID_WORK_DIMENSION";
    case CL_INVALID_WORK_GROUP_SIZE:
        return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_ITEM_SIZE:
        return "CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_GLOBAL_OFFSET:
        return "CL_INVALID_GLOBAL_OFFSET";
    case CL_INVALID_EVENT_WAIT_LIST:
        return "CL_INVALID_EVENT_WAIT_LIST";
    case CL_INVALID_EVENT:
        return "CL_INVALID_EVENT";
    case CL_INVALID_OPERATION:
        return "CL_INVALID_OPERATION";
    case CL_INVALID_GL_OBJECT:
        return "CL_INVALID_GL_OBJECT";
    case CL_INVALID_BUFFER_SIZE:
        return "CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_MIP_LEVEL:
        return "CL_INVALID_MIP_LEVEL";
    default:
        return "Unknown OpenCL error";
    }
}

std::string readFile(const std::string &filePath)
{
    std::ifstream file(filePath, std::ios::in | std::ios::binary);
    if (!file.is_open())
    {
        throw std::runtime_error("Cannot open file: " + filePath);
    }

    std::ostringstream oss;
    oss << file.rdbuf();
    return oss.str();
}

void buildOpenCLProgram(OpenCLContext &context, const std::string &source, const char *buildOptions)
{
    cl_int status = 0;
    const char *sourceStr = source.c_str();
    size_t sourceSize = source.size();

    context.program = clCreateProgramWithSource(context.context, 1, &sourceStr, &sourceSize, &status);
    if (status != CL_SUCCESS)
    {
        throw std::runtime_error("Failed to create program from source: " + getCLErrorString(status));
    }

    status = clBuildProgram(context.program, 1, &context.deviceId, buildOptions, nullptr, nullptr);
    if (status != CL_SUCCESS)
    {
        size_t logSize = 0;
        clGetProgramBuildInfo(context.program, context.deviceId, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);

        std::vector<char> buildLog(logSize);
        clGetProgramBuildInfo(context.program, context.deviceId, CL_PROGRAM_BUILD_LOG, logSize, buildLog.data(), nullptr);

        std::string logString(buildLog.begin(), buildLog.end());
        std::ostringstream oss;
        oss << "Failed to build program: " << getCLErrorString(status)
            << "\nBuild log:\n"
            << logString;
        throw std::runtime_error(oss.str());
    }
}

cl_kernel createKernel(const OpenCLContext &context, const std::string &kernelName)
{
    cl_int status = 0;
    cl_kernel kernel = clCreateKernel(context.program, kernelName.c_str(), &status);
    if (status != CL_SUCCESS)
    {
        throw std::runtime_error("Failed to create kernel " + kernelName +
                                 ": " + getCLErrorString(status));
    }
    return kernel;
}

void printPlatformInfo()
{
    cl_uint numPlatforms = 0;
    cl_int status = clGetPlatformIDs(0, nullptr, &numPlatforms);
    if (status != CL_SUCCESS)
    {
        std::cerr << "Error getting number of platforms: " << getCLErrorString(status) << std::endl;
        return;
    }

    std::vector<cl_platform_id> platforms(numPlatforms);
    status = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    if (status != CL_SUCCESS)
    {
        std::cerr << "Error getting platform IDs: " << getCLErrorString(status) << std::endl;
        return;
    }

    for (cl_uint i = 0; i < numPlatforms; ++i)
    {
        size_t infoLen = 0;
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, nullptr, &infoLen);
        std::vector<char> infoStr(infoLen);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, infoLen, infoStr.data(), nullptr);

        std::cout << "Platform " << i << " Name: " << infoStr.data() << std::endl;
    }
}

cl_program buildProgramForDevice(cl_context context, const std::string &source, cl_device_id device)
{
    cl_int err;
    const char *source_cstr = source.c_str();
    size_t source_size = source.size();

    cl_program program = clCreateProgramWithSource(context, 1, &source_cstr, &source_size, &err);
    if (err != CL_SUCCESS)
    {
        printf("Error creating program: %d\n", err);
        return NULL;
    }

    err = clBuildProgram(program, 1, &device, "-I .", NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), NULL);
        printf("Error building program: %s\n", log.data());
        clReleaseProgram(program);
        return NULL;
    }

    return program;
}

std::vector<OpenCLContext> initializeOpenCLDevices()
{
    std::vector<OpenCLContext> contexts;
    cl_uint num_platforms;
    cl_int err;

    // Get platforms
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0)
    {
        printf("No OpenCL platforms found\n");
        return contexts;
    }

    std::vector<cl_platform_id> platforms(num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms.data(), NULL);

    // For each platform, get devices
    for (cl_uint p = 0; p < num_platforms; p++)
    {
        cl_uint num_devices;
        err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
        if (err != CL_SUCCESS || num_devices == 0)
            continue;

        std::vector<cl_device_id> devices(num_devices);
        err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, num_devices, devices.data(), NULL);

        // Create context and command queue for each device
        for (cl_uint d = 0; d < num_devices; d++)
        {
            cl_context_properties props[] = {
                CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[p],
                0};

            cl_context context = clCreateContext(props, 1, &devices[d], NULL, NULL, &err);
            if (err != CL_SUCCESS)
                continue;

            cl_command_queue queue = clCreateCommandQueueWithProperties(context, devices[d], nullptr, &err);
            if (err != CL_SUCCESS)
            {
                clReleaseContext(context);
                continue;
            }

            // Create program for this device
            std::string source = readFile("path_tracer.cl");
            cl_program program = buildProgramForDevice(context, source, devices[d]);

            OpenCLContext deviceContext;
            deviceContext.context = context;
            deviceContext.commandQueue = queue;
            deviceContext.deviceId = devices[d];
            deviceContext.program = program;
            contexts.push_back(deviceContext);

            // Log device info
            char device_name[256];
            clGetDeviceInfo(devices[d], CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
            printf("Found GPU device: %s\n", device_name);
        }
    }

    return contexts;
}

void cleanup_resources(const OpenCLBuffers buffers, const OpenCLContext &context)
{
    if (buffers.output_image)
        clReleaseMemObject(buffers.output_image);
    if (buffers.colors)
        clReleaseMemObject(buffers.colors);
    if (buffers.subframes)
        clReleaseMemObject(buffers.subframes);
    if (buffers.instances)
        clReleaseMemObject(buffers.instances);
    if (buffers.bvh_nodes)
        clReleaseMemObject(buffers.bvh_nodes);
    if (buffers.bvh_links)
        clReleaseMemObject(buffers.bvh_links);
    if (buffers.mesh_indices)
        clReleaseMemObject(buffers.mesh_indices);
    if (buffers.mesh_pos)
        clReleaseMemObject(buffers.mesh_pos);
    if (buffers.mesh_normal)
        clReleaseMemObject(buffers.mesh_normal);
    if (buffers.mesh_albedo)
        clReleaseMemObject(buffers.mesh_albedo);
    if (buffers.mesh_material)
        clReleaseMemObject(buffers.mesh_material);
    if (context.commandQueue)
        clReleaseCommandQueue(context.commandQueue);
    if (context.context)
        clReleaseContext(context.context);
    if (context.program)
        clReleaseProgram(context.program);
    fprintf(stderr, "OpenCL resources released\n");
}