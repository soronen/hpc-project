#include "scene.hh"
#include "path_tracer.hh"
#include "bmp.hh"
#include <clocale>
#include <memory>
#include <fstream>

#include <filesystem>
#include <omp.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include "opencl_utils.h"

// Renders the given scene into an image using path tracing.
void baseline_render(const scene &s, uchar4 *image)
{
    for (uint i = 0; i < IMAGE_WIDTH * IMAGE_HEIGHT; ++i)
    {
        uint x = i % IMAGE_WIDTH;
        uint y = i / IMAGE_WIDTH;

        float3 color = {0, 0, 0};

#pragma omp parallel for
        for (uint j = 0; j < SAMPLES_PER_PIXEL; ++j)
        {
            color += path_trace_pixel(
                uint2{x, y},
                j,
                s.subframes.data(),
                s.instances.data(),
                s.bvh_buf.nodes.data(),
                s.bvh_buf.links.data(),
                s.mesh_buf.indices.data(),
                s.mesh_buf.pos.data(),
                s.mesh_buf.normal.data(),
                s.mesh_buf.albedo.data(),
                s.mesh_buf.material.data());
        }

        color /= SAMPLES_PER_PIXEL;

        image[i] = tonemap_pixel(color);
    }
}

// OpenCL version of the renderer
void opencl_render(const scene &s, uchar4 *image, OpenCLContext &cl_context, cl_kernel path_trace_kernel, cl_kernel tonemap_kernel)
{
    cl_int err;

    // Create buffers for OpenCL
    size_t image_size = IMAGE_WIDTH * IMAGE_HEIGHT;

    // Create buffer for the final image
    cl_mem cl_output_image = clCreateBuffer(cl_context.context, CL_MEM_WRITE_ONLY,
                                            image_size * sizeof(uchar4), NULL, &err);
    if (err != CL_SUCCESS)
    {
        printf("Error creating output image buffer: %d\n", err);
        return;
    }

    // Create buffer for intermediate HDR colors
    cl_mem cl_colors = clCreateBuffer(cl_context.context, CL_MEM_READ_WRITE,
                                      image_size * sizeof(float3), NULL, &err);
    if (err != CL_SUCCESS)
    {
        printf("Error creating colors buffer: %d (%s)\n", err, getCLErrorString(err).c_str());
        return;
    }

    // printf("Subframes count: %zu, sizeof(subframe): %zu, total: %zu bytes\n",
    //        s.subframes.size(), sizeof(subframe), s.subframes.size() * sizeof(subframe));

    // Create buffers for scene data
    cl_mem cl_subframes = clCreateBuffer(cl_context.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         s.subframes.size() * sizeof(subframe), (void *)s.subframes.data(), &err);
    if (err != CL_SUCCESS)
    {
        printf("Error creating subframes buffer: %d (%s)\n", err, getCLErrorString(err).c_str());
        return;
    }

    cl_mem cl_instances = clCreateBuffer(cl_context.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         s.instances.size() * sizeof(tlas_instance), (void *)s.instances.data(), &err);
    if (err != CL_SUCCESS)
    {
        printf("Error creating instances buffer: %d (%s)\n", err, getCLErrorString(err).c_str());
        return;
    }

    cl_mem cl_bvh_nodes = clCreateBuffer(cl_context.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         s.bvh_buf.nodes.size() * sizeof(bvh_node), (void *)s.bvh_buf.nodes.data(), &err);
    if (err != CL_SUCCESS)
    {
        printf("Error creating BVH nodes buffer: %d (%s)\n", err, getCLErrorString(err).c_str());
        return;
    }

    cl_mem cl_bvh_links = clCreateBuffer(cl_context.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         s.bvh_buf.links.size() * sizeof(bvh_link), (void *)s.bvh_buf.links.data(), &err);
    if (err != CL_SUCCESS)
    {
        printf("Error creating BVH links buffer: %d (%s)\n", err, getCLErrorString(err).c_str());
        return;
    }

    cl_mem cl_mesh_indices = clCreateBuffer(cl_context.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            s.mesh_buf.indices.size() * sizeof(uint), (void *)s.mesh_buf.indices.data(), &err);
    if (err != CL_SUCCESS)
    {
        printf("Error creating mesh indices buffer: %d (%s)\n", err, getCLErrorString(err).c_str());
        return;
    }

    cl_mem cl_mesh_pos = clCreateBuffer(cl_context.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        s.mesh_buf.pos.size() * sizeof(float3), (void *)s.mesh_buf.pos.data(), &err);
    if (err != CL_SUCCESS)
    {
        printf("Error creating mesh positions buffer: %d (%s)\n", err, getCLErrorString(err).c_str());
        return;
    }

    cl_mem cl_mesh_normal = clCreateBuffer(cl_context.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           s.mesh_buf.normal.size() * sizeof(float3), (void *)s.mesh_buf.normal.data(), &err);
    if (err != CL_SUCCESS)
    {
        printf("Error creating mesh normals buffer: %d (%s)\n", err, getCLErrorString(err).c_str());
        return;
    }

    cl_mem cl_mesh_albedo = clCreateBuffer(cl_context.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           s.mesh_buf.albedo.size() * sizeof(float4), (void *)s.mesh_buf.albedo.data(), &err);
    if (err != CL_SUCCESS)
    {
        printf("Error creating mesh albedo buffer: %d (%s)\n", err, getCLErrorString(err).c_str());
        return;
    }

    cl_mem cl_mesh_material = clCreateBuffer(cl_context.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                             s.mesh_buf.material.size() * sizeof(float4), (void *)s.mesh_buf.material.data(), &err);
    if (err != CL_SUCCESS)
    {
        printf("Error creating mesh material buffer: %d (%s)\n", err, getCLErrorString(err).c_str());
        return;
    }

    // Set path tracing kernel arguments

    uint2 dimensions = {IMAGE_WIDTH, IMAGE_HEIGHT};
    uint samples = SAMPLES_PER_PIXEL;
    err = clSetKernelArg(path_trace_kernel, 0, sizeof(uint2), &dimensions);
    err |= clSetKernelArg(path_trace_kernel, 1, sizeof(uint), &samples);
    err |= clSetKernelArg(path_trace_kernel, 2, sizeof(cl_mem), &cl_colors);
    err |= clSetKernelArg(path_trace_kernel, 3, sizeof(cl_mem), &cl_subframes);
    err |= clSetKernelArg(path_trace_kernel, 4, sizeof(cl_mem), &cl_instances);
    err |= clSetKernelArg(path_trace_kernel, 5, sizeof(cl_mem), &cl_bvh_nodes);
    err |= clSetKernelArg(path_trace_kernel, 6, sizeof(cl_mem), &cl_bvh_links);
    err |= clSetKernelArg(path_trace_kernel, 7, sizeof(cl_mem), &cl_mesh_indices);
    err |= clSetKernelArg(path_trace_kernel, 8, sizeof(cl_mem), &cl_mesh_pos);
    err |= clSetKernelArg(path_trace_kernel, 9, sizeof(cl_mem), &cl_mesh_normal);
    err |= clSetKernelArg(path_trace_kernel, 10, sizeof(cl_mem), &cl_mesh_albedo);
    err |= clSetKernelArg(path_trace_kernel, 11, sizeof(cl_mem), &cl_mesh_material);

    if (err != CL_SUCCESS)
    {
        printf("Error setting path tracing kernel arguments: %d\n", err);
        return;
    }

    // Set tonemap kernel arguments
    err = clSetKernelArg(tonemap_kernel, 0, sizeof(cl_mem), &cl_colors);
    err |= clSetKernelArg(tonemap_kernel, 1, sizeof(cl_mem), &cl_output_image);

    if (err != CL_SUCCESS)
    {
        printf("Error setting tonemap kernel arguments: %d\n", err);
        return;
    }

    // Define work dimensions
    size_t global_work_size[2] = {IMAGE_WIDTH, IMAGE_HEIGHT};

    // Execute the path tracing kernel
    err = clEnqueueNDRangeKernel(cl_context.commandQueue, path_trace_kernel, 2,
                                 NULL, global_work_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error executing path tracing kernel: %d\n", err);
        return;
    }

    // Execute the tonemap kernel
    err = clEnqueueNDRangeKernel(cl_context.commandQueue, tonemap_kernel, 2,
                                 NULL, global_work_size, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error executing tonemap kernel: %d\n", err);
        return;
    }

    // Read back the result
    err = clEnqueueReadBuffer(cl_context.commandQueue, cl_output_image, CL_TRUE, 0,
                              image_size * sizeof(uchar4), image, 0, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error reading output image: %d\n", err);
        return;
    }

    // Clean up OpenCL buffers
    clReleaseMemObject(cl_output_image);
    clReleaseMemObject(cl_colors);
    clReleaseMemObject(cl_subframes);
    clReleaseMemObject(cl_instances);
    clReleaseMemObject(cl_bvh_nodes);
    clReleaseMemObject(cl_bvh_links);
    clReleaseMemObject(cl_mesh_indices);
    clReleaseMemObject(cl_mesh_pos);
    clReleaseMemObject(cl_mesh_normal);
    clReleaseMemObject(cl_mesh_albedo);
    clReleaseMemObject(cl_mesh_material);
}

int main()
{
    // since i have my executable in /build folder
    std::filesystem::current_path(std::filesystem::path(__FILE__).parent_path());

    // Make sure all text parsing is unaffected by locale
    setlocale(LC_ALL, "C");

    scene s = load_scene();

    // Initialize OpenCL
    OpenCLContext cl_context;
    bool use_opencl = true;

    cl_kernel path_trace_kernel;
    cl_kernel tonemap_kernel;

    try
    {
        cl_context = createOpenCLContext(0, 0, CL_DEVICE_TYPE_GPU);
        std::string source = readFile("path_tracer.cl");
        buildOpenCLProgram(cl_context, source);
        path_trace_kernel = createKernel(cl_context, "path_trace_pixel_kernel");
        tonemap_kernel = createKernel(cl_context, "tonemap_kernel");

        // Get device info for logging
        char device_name[256];
        clGetDeviceInfo(cl_context.deviceId, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
        printf("Using OpenCL device: %s\n", device_name);
    }
    catch (const std::exception &e)
    {
        printf("Failed to initialize OpenCL: %s\n", e.what());
        printf("Falling back to CPU implementation\n");
        use_opencl = false;
    }

    std::unique_ptr<uchar4[]> image(new uchar4[IMAGE_WIDTH * IMAGE_HEIGHT]);

    uint frame_count = get_animation_frame_count(s);
    for (uint frame_index = 0; frame_index < frame_count; ++frame_index)
    {
        // Update scene state for the current frame & render it
        setup_animation_frame(s, frame_index);

        // Use either OpenCL or CPU implementation
        if (use_opencl)
        {
            opencl_render(s, image.get(), cl_context, path_trace_kernel, tonemap_kernel);
        }
        else
        {
            baseline_render(s, image.get());
        }

        // Create string for the index number of the frame with leading zeroes.
        std::string index_str = std::to_string(frame_index);
        while (index_str.size() < 4)
            index_str.insert(index_str.begin(), '0');

        // Write output image
        write_bmp(
            ("output/frame_" + index_str + ".bmp").c_str(),
            IMAGE_WIDTH, IMAGE_HEIGHT, 4, IMAGE_WIDTH * 4,
            (uint8_t *)image.get());
    }

    // Release OpenCL resources
    if (use_opencl)
    {
        clReleaseKernel(path_trace_kernel);
        clReleaseKernel(tonemap_kernel);
        clReleaseProgram(cl_context.program);
        clReleaseCommandQueue(cl_context.commandQueue);
        clReleaseContext(cl_context.context);
    }

    return 0;
}
