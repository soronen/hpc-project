#include "scene.hh"
#include "path_tracer.hh"
#include "bmp.hh"
#include <clocale>
#include <memory>
#include <fstream>
#include <array>

#include <filesystem>
#include <omp.h>

#ifdef __APPLE__
#define CL_TARGET_OPENCL_VERSION 120
#include <OpenCL/opencl.h>
#else
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#endif
#include "opencl_utils.h"
#include <iostream>

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

OpenCLBuffers create_render_buffers(const scene &s, const OpenCLContext &cl_context)
{
    cl_int err;

    // Create buffers for OpenCL
    size_t image_size = IMAGE_WIDTH * IMAGE_HEIGHT;

    // Create buffer for the final image
    cl_mem cl_output_image = clCreateBuffer(cl_context.context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                            image_size * sizeof(uchar4), NULL, &err);
    if (err != CL_SUCCESS)
    {
        printf("Error creating output image buffer: %d\n", err);
        exit(EXIT_FAILURE);
    }

    // Create buffer for intermediate HDR colors
    cl_mem cl_colors = clCreateBuffer(cl_context.context, CL_MEM_READ_WRITE,
                                      image_size * sizeof(float3), NULL, &err);
    if (err != CL_SUCCESS)
    {
        printf("Error creating colors buffer: %d (%s)\n", err, getCLErrorString(err).c_str());
        exit(EXIT_FAILURE);
    }

    // Create buffers for scene data
    cl_mem cl_subframes = clCreateBuffer(cl_context.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         s.subframes.size() * sizeof(subframe), (void *)s.subframes.data(), &err);
    if (err != CL_SUCCESS)
    {
        printf("Error creating subframes buffer: %d (%s)\n", err, getCLErrorString(err).c_str());
        exit(EXIT_FAILURE);
    }

    cl_mem cl_instances = clCreateBuffer(cl_context.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         s.instances.size() * sizeof(tlas_instance), (void *)s.instances.data(), &err);
    if (err != CL_SUCCESS)
    {
        printf("Error creating instances buffer: %d (%s)\n", err, getCLErrorString(err).c_str());
        exit(EXIT_FAILURE);
    }

    cl_mem cl_bvh_nodes = clCreateBuffer(cl_context.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         s.bvh_buf.nodes.size() * sizeof(bvh_node), (void *)s.bvh_buf.nodes.data(), &err);
    if (err != CL_SUCCESS)
    {
        printf("Error creating BVH nodes buffer: %d (%s)\n", err, getCLErrorString(err).c_str());
        exit(EXIT_FAILURE);
    }

    cl_mem cl_bvh_links = clCreateBuffer(cl_context.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         s.bvh_buf.links.size() * sizeof(bvh_link), (void *)s.bvh_buf.links.data(), &err);
    if (err != CL_SUCCESS)
    {
        printf("Error creating BVH links buffer: %d (%s)\n", err, getCLErrorString(err).c_str());
        exit(EXIT_FAILURE);
    }

    cl_mem cl_mesh_indices = clCreateBuffer(cl_context.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                            s.mesh_buf.indices.size() * sizeof(uint), (void *)s.mesh_buf.indices.data(), &err);
    if (err != CL_SUCCESS)
    {
        printf("Error creating mesh indices buffer: %d (%s)\n", err, getCLErrorString(err).c_str());
        exit(EXIT_FAILURE);
    }

    cl_mem cl_mesh_pos = clCreateBuffer(cl_context.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        s.mesh_buf.pos.size() * sizeof(float3), (void *)s.mesh_buf.pos.data(), &err);
    if (err != CL_SUCCESS)
    {
        printf("Error creating mesh positions buffer: %d (%s)\n", err, getCLErrorString(err).c_str());
        exit(EXIT_FAILURE);
    }

    cl_mem cl_mesh_normal = clCreateBuffer(cl_context.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           s.mesh_buf.normal.size() * sizeof(float3), (void *)s.mesh_buf.normal.data(), &err);
    if (err != CL_SUCCESS)
    {
        printf("Error creating mesh normals buffer: %d (%s)\n", err, getCLErrorString(err).c_str());
        exit(EXIT_FAILURE);
    }

    cl_mem cl_mesh_albedo = clCreateBuffer(cl_context.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           s.mesh_buf.albedo.size() * sizeof(float4), (void *)s.mesh_buf.albedo.data(), &err);
    if (err != CL_SUCCESS)
    {
        printf("Error creating mesh albedo buffer: %d (%s)\n", err, getCLErrorString(err).c_str());
        exit(EXIT_FAILURE);
    }

    cl_mem cl_mesh_material = clCreateBuffer(cl_context.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                             s.mesh_buf.material.size() * sizeof(float4), (void *)s.mesh_buf.material.data(), &err);
    if (err != CL_SUCCESS)
    {
        printf("Error creating mesh material buffer: %d (%s)\n", err, getCLErrorString(err).c_str());
        exit(EXIT_FAILURE);
    }

    return OpenCLBuffers{
        cl_output_image,
        cl_colors,
        cl_subframes,
        cl_instances,
        cl_bvh_nodes,
        cl_bvh_links,
        cl_mesh_indices,
        cl_mesh_pos,
        cl_mesh_normal,
        cl_mesh_albedo,
        cl_mesh_material};
}

std::array<cl_event, 9> update_render_buffers(const scene &s, const OpenCLBuffers &buffers, const OpenCLContext &cl_context)
{
    cl_int err;
    std::array<cl_event, 9> write_events;

    // Update all scene data buffers with new animation frame data
    err = clEnqueueWriteBuffer(cl_context.commandQueue, buffers.subframes, CL_FALSE, 0,
                               s.subframes.size() * sizeof(subframe), s.subframes.data(), 0, NULL, &write_events[0]);
    if (err != CL_SUCCESS)
    {
        printf("Error updating subframes buffer: %d (%s)\n", err, getCLErrorString(err).c_str());
    }

    err = clEnqueueWriteBuffer(cl_context.commandQueue, buffers.instances, CL_FALSE, 0,
                               s.instances.size() * sizeof(tlas_instance), s.instances.data(), 0, NULL, &write_events[1]);
    if (err != CL_SUCCESS)
    {
        printf("Error updating instances buffer: %d (%s)\n", err, getCLErrorString(err).c_str());
    }

    // Update BVH structures which might change if objects move
    err = clEnqueueWriteBuffer(cl_context.commandQueue, buffers.bvh_nodes, CL_FALSE, 0,
                               s.bvh_buf.nodes.size() * sizeof(bvh_node), s.bvh_buf.nodes.data(), 0, NULL, &write_events[2]);
    if (err != CL_SUCCESS)
    {
        printf("Error updating BVH nodes buffer: %d (%s)\n", err, getCLErrorString(err).c_str());
    }

    err = clEnqueueWriteBuffer(cl_context.commandQueue, buffers.bvh_links, CL_FALSE, 0,
                               s.bvh_buf.links.size() * sizeof(bvh_link), s.bvh_buf.links.data(), 0, NULL, &write_events[3]);
    if (err != CL_SUCCESS)
    {
        printf("Error updating BVH links buffer: %d (%s)\n", err, getCLErrorString(err).c_str());
    }

    // Update mesh data which might change during animation
    err = clEnqueueWriteBuffer(cl_context.commandQueue, buffers.mesh_pos, CL_FALSE, 0,
                               s.mesh_buf.pos.size() * sizeof(float3), s.mesh_buf.pos.data(), 0, NULL, &write_events[4]);
    if (err != CL_SUCCESS)
    {
        printf("Error updating mesh positions buffer: %d (%s)\n", err, getCLErrorString(err).c_str());
    }

    err = clEnqueueWriteBuffer(cl_context.commandQueue, buffers.mesh_normal, CL_FALSE, 0,
                               s.mesh_buf.normal.size() * sizeof(float3), s.mesh_buf.normal.data(), 0, NULL, &write_events[5]);
    if (err != CL_SUCCESS)
    {
        printf("Error updating mesh normals buffer: %d (%s)\n", err, getCLErrorString(err).c_str());
    }

    // These might not change between frames but update to be safe
    err = clEnqueueWriteBuffer(cl_context.commandQueue, buffers.mesh_indices, CL_FALSE, 0,
                               s.mesh_buf.indices.size() * sizeof(uint), s.mesh_buf.indices.data(), 0, NULL, &write_events[6]);
    if (err != CL_SUCCESS)
    {
        printf("Error updating mesh indices buffer: %d (%s)\n", err, getCLErrorString(err).c_str());
    }

    err = clEnqueueWriteBuffer(cl_context.commandQueue, buffers.mesh_albedo, CL_FALSE, 0,
                               s.mesh_buf.albedo.size() * sizeof(float4), s.mesh_buf.albedo.data(), 0, NULL, &write_events[7]);
    if (err != CL_SUCCESS)
    {
        printf("Error updating mesh albedo buffer: %d (%s)\n", err, getCLErrorString(err).c_str());
    }

    err = clEnqueueWriteBuffer(cl_context.commandQueue, buffers.mesh_material, CL_FALSE, 0,
                               s.mesh_buf.material.size() * sizeof(float4), s.mesh_buf.material.data(), 0, NULL, &write_events[8]);
    if (err != CL_SUCCESS)
    {
        printf("Error updating mesh material buffer: %d (%s)\n", err, getCLErrorString(err).c_str());
    }

    clWaitForEvents(9, write_events.data());

    return write_events;
}

// OpenCL version of the renderer
void opencl_render(const scene &s, uchar4 *image, const OpenCLContext &cl_context, OpenCLBuffers buffers, cl_kernel path_trace_kernel, cl_kernel tonemap_kernel)
{
    // Set path tracing kernel arguments
    cl_int err;
    size_t image_size = IMAGE_WIDTH * IMAGE_HEIGHT;

    uint2 dimensions = {IMAGE_WIDTH, IMAGE_HEIGHT};
    uint samples = SAMPLES_PER_PIXEL;
    err = clSetKernelArg(path_trace_kernel, 0, sizeof(uint2), &dimensions);
    err |= clSetKernelArg(path_trace_kernel, 1, sizeof(uint), &samples);
    err |= clSetKernelArg(path_trace_kernel, 2, sizeof(cl_mem), &buffers.colors);
    err |= clSetKernelArg(path_trace_kernel, 3, sizeof(cl_mem), &buffers.subframes);
    err |= clSetKernelArg(path_trace_kernel, 4, sizeof(cl_mem), &buffers.instances);
    err |= clSetKernelArg(path_trace_kernel, 5, sizeof(cl_mem), &buffers.bvh_nodes);
    err |= clSetKernelArg(path_trace_kernel, 6, sizeof(cl_mem), &buffers.bvh_links);
    err |= clSetKernelArg(path_trace_kernel, 7, sizeof(cl_mem), &buffers.mesh_indices);
    err |= clSetKernelArg(path_trace_kernel, 8, sizeof(cl_mem), &buffers.mesh_pos);
    err |= clSetKernelArg(path_trace_kernel, 9, sizeof(cl_mem), &buffers.mesh_normal);
    err |= clSetKernelArg(path_trace_kernel, 10, sizeof(cl_mem), &buffers.mesh_albedo);
    err |= clSetKernelArg(path_trace_kernel, 11, sizeof(cl_mem), &buffers.mesh_material);

    if (err != CL_SUCCESS)
    {
        printf("Error setting path tracing kernel arguments: %d (%s)\n", err, getCLErrorString(err).c_str());
        return;
    }

    // Set tonemap kernel arguments
    err = clSetKernelArg(tonemap_kernel, 0, sizeof(cl_mem), &buffers.colors);
    err |= clSetKernelArg(tonemap_kernel, 1, sizeof(cl_mem), &buffers.output_image);

    if (err != CL_SUCCESS)
    {
        printf("Error setting tonemap kernel arguments: %d (%s)\n", err, getCLErrorString(err).c_str());
        return;
    }

    // Define work dimensions
    size_t global_work_size[2] = {IMAGE_WIDTH, IMAGE_HEIGHT};
    size_t local_work_size[2] = {8, 4};

    // size_t preferredWGMultiple;
    // clGetKernelWorkGroupInfo(path_trace_kernel, cl_context.deviceId, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &preferredWGMultiple, NULL);
    // std::cout << "clGetKernelWorkGroupInfo path_trace_kernel, preferred work group size multiple: " << preferredWGMultiple << std::endl;
    // clGetKernelWorkGroupInfo(tonemap_kernel, cl_context.deviceId, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &preferredWGMultiple, NULL);
    // std::cout << "clGetKernelWorkGroupInfo tonemap_kernel, preferred work group size multiple: " << preferredWGMultiple << std::endl;

    global_work_size[0] = ((IMAGE_WIDTH + local_work_size[0] - 1) / local_work_size[0]) * local_work_size[0];
    global_work_size[1] = ((IMAGE_HEIGHT + local_work_size[1] - 1) / local_work_size[1]) * local_work_size[1];

    // Execute the path tracing kernel
    cl_event kernel_events[2];
    err = clEnqueueNDRangeKernel(cl_context.commandQueue, path_trace_kernel, 2,
                                 NULL, global_work_size, local_work_size, 0, NULL, &kernel_events[0]);
    if (err != CL_SUCCESS)
    {
        printf("Error executing path tracing kernel: %d (%s)\n", err, getCLErrorString(err).c_str());
        return;
    }

    // Execute the tonemap kernel
    err = clEnqueueNDRangeKernel(cl_context.commandQueue, tonemap_kernel, 2,
                                 NULL, global_work_size, NULL, 1, &kernel_events[0], &kernel_events[1]);
    if (err != CL_SUCCESS)
    {
        printf("Error executing tonemap kernel: %d (%s)\n", err, getCLErrorString(err).c_str());
        return;
    }

    // Read back the result
    err = clEnqueueReadBuffer(cl_context.commandQueue, buffers.output_image, CL_TRUE, 0,
                              image_size * sizeof(uchar4), image, 1, &kernel_events[1], NULL);
    if (err != CL_SUCCESS)
    {
        printf("Error reading output image: %d (%s)\n", err, getCLErrorString(err).c_str());
        return;
    }

    for (int i = 0; i < 2; i++)
        clReleaseEvent(kernel_events[i]);
}

int main()
{
    // since i have my executable in /build folder
    std::filesystem::current_path(std::filesystem::path(__FILE__).parent_path());

    // Make sure all text parsing is unaffected by locale
    setlocale(LC_ALL, "C");

    scene s = load_scene();
    uint frame_count = get_animation_frame_count(s);

    // create the first frame here for the buffers
    setup_animation_frame(s, 0);

    // Initialize OpenCL
    bool use_opencl = true;
    OpenCLContext cl_context;
    cl_kernel path_trace_kernel;
    cl_kernel tonemap_kernel;
    OpenCLBuffers buffers[2];

    std::unique_ptr<uchar4[]> image(new uchar4[IMAGE_WIDTH * IMAGE_HEIGHT]);

    try
    {
        cl_context = createOpenCLContext(0, 0, CL_DEVICE_TYPE_GPU);
        std::string source = readFile("path_tracer.cl");
        buildOpenCLProgram(cl_context, source);
        path_trace_kernel = createKernel(cl_context, "path_trace_pixel_kernel");
        tonemap_kernel = createKernel(cl_context, "tonemap_kernel");
        buffers[0] = create_render_buffers(s, cl_context);
        buffers[1] = create_render_buffers(s, cl_context);

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

    int current_buffer = 0;
    int next_buffer = 1;
    std::array<cl_event, 9> update_events = update_render_buffers(s, buffers[current_buffer], cl_context);
    clWaitForEvents(update_events.size(), update_events.data());
    for (cl_event &event : update_events)
        clReleaseEvent(event);

    for (uint frame_index = 0; frame_index < frame_count; ++frame_index)
    {
        if (use_opencl)
        {
            // Prepare next frame while current frame renders
            if (frame_index + 1 < frame_count)
            {
                setup_animation_frame(s, frame_index + 1);
                update_events = update_render_buffers(s, buffers[next_buffer], cl_context);
            }

            opencl_render(s, image.get(), cl_context, buffers[0], path_trace_kernel, tonemap_kernel);

            // Save current frame
            std::string index_str = std::to_string(frame_index);
            while (index_str.size() < 4)
                index_str.insert(index_str.begin(), '0');
            write_bmp(("output/frame_" + index_str + ".bmp").c_str(),
                      IMAGE_WIDTH, IMAGE_HEIGHT, 4, IMAGE_WIDTH * 4, (uint8_t *)image.get());

            // Wait for next buffer to be ready before swapping
            if (frame_index + 1 < frame_count)
            {
                clWaitForEvents(update_events.size(), update_events.data());
                for (auto &event : update_events)
                    clReleaseEvent(event);
                std::swap(current_buffer, next_buffer);
            }
        }
        else
        {
            setup_animation_frame(s, frame_index);
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
        for (int i = 0; i < 2; i++)
        {
            clReleaseMemObject(buffers[i].output_image);
            clReleaseMemObject(buffers[i].colors);
            clReleaseMemObject(buffers[i].subframes);
            clReleaseMemObject(buffers[i].instances);
            clReleaseMemObject(buffers[i].bvh_nodes);
            clReleaseMemObject(buffers[i].bvh_links);
            clReleaseMemObject(buffers[i].mesh_indices);
            clReleaseMemObject(buffers[i].mesh_pos);
            clReleaseMemObject(buffers[i].mesh_normal);
            clReleaseMemObject(buffers[i].mesh_albedo);
            clReleaseMemObject(buffers[i].mesh_material);
        }

        clReleaseKernel(path_trace_kernel);
        clReleaseKernel(tonemap_kernel);
        clReleaseProgram(cl_context.program);
        clReleaseCommandQueue(cl_context.commandQueue);
        clReleaseContext(cl_context.context);
    }

    return 0;
}
