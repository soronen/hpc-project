#include "scene.hh"
#include "path_tracer.hh"
#include "bmp.hh"
#include "opencl_utils.hh"
#include <omp.h>
#include <clocale>
#include <memory>
#include <fstream>
#include <array>
#include <iostream>
#include <filesystem>

#ifdef USE_LUMI
#include <mpi.h>
#endif

// Renders the given scene into an image using path tracing.
void baseline_render(const scene &s, uchar4 *image)
{
#pragma omp parallel for
    for (uint i = 0; i < IMAGE_WIDTH * IMAGE_HEIGHT; ++i)
    {
        uint x = i % IMAGE_WIDTH;
        uint y = i / IMAGE_WIDTH;

        float3 color = {0, 0, 0};

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
    cl_mem accumulation_buffer;
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

    cl_mem cl_accumulation = clCreateBuffer(cl_context.context, CL_MEM_READ_WRITE,
                                            image_size * sizeof(float3), NULL, &err);
    if (err != CL_SUCCESS)
    {
        printf("Error creating accumulation buffer: %d (%s)\n", err, getCLErrorString(err).c_str());
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
        cl_mesh_material,
        cl_accumulation};
}

std::array<cl_event, 9> update_render_buffers(const scene &s, OpenCLBuffers &buffers, const OpenCLContext &cl_context)
{
    // cpu parallelize the buffer updates
    cl_int err;
    std::array<cl_event, 9> write_events;

    // Track sizes of all buffers that might change
    static size_t last_subframes_size = 0;
    static size_t last_instances_size = 0;
    static size_t last_nodes_size = 0;
    static size_t last_links_size = 0;
    static size_t last_indices_size = 0;
    static size_t last_pos_size = 0;
    static size_t last_normal_size = 0;
    static size_t last_albedo_size = 0;
    static size_t last_material_size = 0;

    bool error_occured = false;

    // Check and update subframes buffer
    size_t subframes_size = s.subframes.size();
    if (subframes_size != last_subframes_size)
    {
        printf("Recreating subframes buffer, old size: %zu, new size: %zu\n", last_subframes_size, subframes_size);
        if (buffers.subframes)
        {
            clReleaseMemObject(buffers.subframes);
        }
        buffers.subframes = clCreateBuffer(cl_context.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           subframes_size * sizeof(subframe), (void *)s.subframes.data(), &err);
        if (err != CL_SUCCESS)
        {
            printf("Error recreating subframes buffer: %d (%s)\n", err, getCLErrorString(err).c_str());
            error_occured = true;
        }
        last_subframes_size = subframes_size;
        write_events[0] = NULL;
    }
    else
    {
        err = clEnqueueWriteBuffer(cl_context.commandQueue, buffers.subframes, CL_FALSE, 0,
                                   subframes_size * sizeof(subframe), s.subframes.data(), 0, NULL, &write_events[0]);
        if (err != CL_SUCCESS)
        {
            printf("Error updating subframes buffer: %d (%s)\n", err, getCLErrorString(err).c_str());
        }
    }

    // Check and update instances buffer
    size_t instances_size = s.instances.size();
    if (instances_size != last_instances_size)
    {
        printf("Recreating instances buffer, old size: %zu, new size: %zu\n", last_instances_size, instances_size);
        if (buffers.instances)
        {
            clReleaseMemObject(buffers.instances);
        }
        buffers.instances = clCreateBuffer(cl_context.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           instances_size * sizeof(tlas_instance), (void *)s.instances.data(), &err);
        if (err != CL_SUCCESS)
        {
            printf("Error recreating instances buffer: %d (%s)\n", err, getCLErrorString(err).c_str());
            error_occured = true;
        }
        last_instances_size = instances_size;
        write_events[1] = NULL;
    }
    else
    {
        err = clEnqueueWriteBuffer(cl_context.commandQueue, buffers.instances, CL_FALSE, 0,
                                   instances_size * sizeof(tlas_instance), s.instances.data(), 0, NULL, &write_events[1]);
        if (err != CL_SUCCESS)
        {
            printf("Error updating instances buffer: %d (%s)\n", err, getCLErrorString(err).c_str());
        }
    }

    // Handle BVH nodes (existing code)
    size_t nodes_size = s.bvh_buf.nodes.size();
    if (nodes_size != last_nodes_size)
    {
        printf("Recreating BVH nodes buffer, old size: %zu, new size: %zu\n", last_nodes_size, nodes_size);
        if (buffers.bvh_nodes)
        {
            clReleaseMemObject(buffers.bvh_nodes);
        }
        buffers.bvh_nodes = clCreateBuffer(cl_context.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           nodes_size * sizeof(bvh_node), (void *)s.bvh_buf.nodes.data(), &err);
        if (err != CL_SUCCESS)
        {
            printf("Error recreating BVH nodes buffer: %d (%s)\n", err, getCLErrorString(err).c_str());
            error_occured = true;
        }
        last_nodes_size = nodes_size;
        write_events[2] = NULL;
    }
    else
    {
        err = clEnqueueWriteBuffer(cl_context.commandQueue, buffers.bvh_nodes, CL_FALSE, 0,
                                   nodes_size * sizeof(bvh_node), s.bvh_buf.nodes.data(), 0, NULL, &write_events[2]);
        if (err != CL_SUCCESS)
        {
            printf("Error updating BVH nodes buffer: %d (%s)\n", err, getCLErrorString(err).c_str());
        }
    }

    // Handle BVH links (existing code)
    size_t links_size = s.bvh_buf.links.size();
    if (links_size != last_links_size)
    {
        printf("Recreating BVH links buffer, old size: %zu, new size: %zu\n", last_links_size, links_size);
        if (buffers.bvh_links)
        {
            clReleaseMemObject(buffers.bvh_links);
        }
        buffers.bvh_links = clCreateBuffer(cl_context.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           links_size * sizeof(bvh_link), (void *)s.bvh_buf.links.data(), &err);
        if (err != CL_SUCCESS)
        {
            printf("Error recreating BVH links buffer: %d (%s)\n", err, getCLErrorString(err).c_str());
            error_occured = true;
        }
        last_links_size = links_size;
        write_events[3] = NULL;
    }
    else
    {
        err = clEnqueueWriteBuffer(cl_context.commandQueue, buffers.bvh_links, CL_FALSE, 0,
                                   links_size * sizeof(bvh_link), s.bvh_buf.links.data(), 0, NULL, &write_events[3]);
        if (err != CL_SUCCESS)
        {
            printf("Error updating BVH links buffer: %d (%s)\n", err, getCLErrorString(err).c_str());
        }
    }

    // Check and update mesh positions buffer
    size_t pos_size = s.mesh_buf.pos.size();
    if (pos_size != last_pos_size)
    {
        printf("Recreating mesh positions buffer, old size: %zu, new size: %zu\n", last_pos_size, pos_size);
        if (buffers.mesh_pos)
        {
            clReleaseMemObject(buffers.mesh_pos);
        }
        buffers.mesh_pos = clCreateBuffer(cl_context.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          pos_size * sizeof(float3), (void *)s.mesh_buf.pos.data(), &err);
        if (err != CL_SUCCESS)
        {
            printf("Error recreating mesh positions buffer: %d (%s)\n", err, getCLErrorString(err).c_str());
            error_occured = true;
        }
        last_pos_size = pos_size;
        write_events[4] = NULL;
    }
    else
    {
        err = clEnqueueWriteBuffer(cl_context.commandQueue, buffers.mesh_pos, CL_FALSE, 0,
                                   pos_size * sizeof(float3), s.mesh_buf.pos.data(), 0, NULL, &write_events[4]);
        if (err != CL_SUCCESS)
        {
            printf("Error updating mesh positions buffer: %d (%s)\n", err, getCLErrorString(err).c_str());
        }
    }

    // Check and update mesh normals buffer
    size_t normal_size = s.mesh_buf.normal.size();
    if (normal_size != last_normal_size)
    {
        printf("Recreating mesh normals buffer, old size: %zu, new size: %zu\n", last_normal_size, normal_size);
        if (buffers.mesh_normal)
        {
            clReleaseMemObject(buffers.mesh_normal);
        }
        buffers.mesh_normal = clCreateBuffer(cl_context.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                             normal_size * sizeof(float3), (void *)s.mesh_buf.normal.data(), &err);
        if (err != CL_SUCCESS)
        {
            printf("Error recreating mesh normals buffer: %d (%s)\n", err, getCLErrorString(err).c_str());
            error_occured = true;
        }
        last_normal_size = normal_size;
        write_events[5] = NULL;
    }
    else
    {
        err = clEnqueueWriteBuffer(cl_context.commandQueue, buffers.mesh_normal, CL_FALSE, 0,
                                   normal_size * sizeof(float3), s.mesh_buf.normal.data(), 0, NULL, &write_events[5]);
        if (err != CL_SUCCESS)
        {
            printf("Error updating mesh normals buffer: %d (%s)\n", err, getCLErrorString(err).c_str());
        }
    }

    // Check and update mesh indices buffer
    size_t indices_size = s.mesh_buf.indices.size();
    if (indices_size != last_indices_size)
    {
        printf("Recreating mesh indices buffer, old size: %zu, new size: %zu\n", last_indices_size, indices_size);
        if (buffers.mesh_indices)
        {
            clReleaseMemObject(buffers.mesh_indices);
        }
        buffers.mesh_indices = clCreateBuffer(cl_context.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                              indices_size * sizeof(uint), (void *)s.mesh_buf.indices.data(), &err);
        if (err != CL_SUCCESS)
        {
            printf("Error recreating mesh indices buffer: %d (%s)\n", err, getCLErrorString(err).c_str());
            error_occured = true;
        }
        last_indices_size = indices_size;
        write_events[6] = NULL;
    }
    else
    {
        err = clEnqueueWriteBuffer(cl_context.commandQueue, buffers.mesh_indices, CL_FALSE, 0,
                                   indices_size * sizeof(uint), s.mesh_buf.indices.data(), 0, NULL, &write_events[6]);
        if (err != CL_SUCCESS)
        {
            printf("Error updating mesh indices buffer: %d (%s)\n", err, getCLErrorString(err).c_str());
        }
    }

    // Check and update mesh albedo buffer
    size_t albedo_size = s.mesh_buf.albedo.size();
    if (albedo_size != last_albedo_size)
    {
        printf("Recreating mesh albedo buffer, old size: %zu, new size: %zu\n", last_albedo_size, albedo_size);
        if (buffers.mesh_albedo)
        {
            clReleaseMemObject(buffers.mesh_albedo);
        }
        buffers.mesh_albedo = clCreateBuffer(cl_context.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                             albedo_size * sizeof(float4), (void *)s.mesh_buf.albedo.data(), &err);
        if (err != CL_SUCCESS)
        {
            printf("Error recreating mesh albedo buffer: %d (%s)\n", err, getCLErrorString(err).c_str());
            error_occured = true;
        }
        last_albedo_size = albedo_size;
        write_events[7] = NULL;
    }
    else
    {
        err = clEnqueueWriteBuffer(cl_context.commandQueue, buffers.mesh_albedo, CL_FALSE, 0,
                                   albedo_size * sizeof(float4), s.mesh_buf.albedo.data(), 0, NULL, &write_events[7]);
        if (err != CL_SUCCESS)
        {
            printf("Error updating mesh albedo buffer: %d (%s)\n", err, getCLErrorString(err).c_str());
        }
    }

    // Check and update mesh material buffer
    size_t material_size = s.mesh_buf.material.size();
    if (material_size != last_material_size)
    {
        printf("Recreating mesh material buffer, old size: %zu, new size: %zu\n", last_material_size, material_size);
        if (buffers.mesh_material)
        {
            clReleaseMemObject(buffers.mesh_material);
        }
        buffers.mesh_material = clCreateBuffer(cl_context.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                               material_size * sizeof(float4), (void *)s.mesh_buf.material.data(), &err);
        if (err != CL_SUCCESS)
        {
            printf("Error recreating mesh material buffer: %d (%s)\n", err, getCLErrorString(err).c_str());
            error_occured = true;
        }
        last_material_size = material_size;
        write_events[8] = NULL;
    }
    else
    {
        err = clEnqueueWriteBuffer(cl_context.commandQueue, buffers.mesh_material, CL_FALSE, 0,
                                   material_size * sizeof(float4), s.mesh_buf.material.data(), 0, NULL, &write_events[8]);
        if (err != CL_SUCCESS)
        {
            printf("Error updating mesh material buffer: %d (%s)\n", err, getCLErrorString(err).c_str());
        }
    }

    if (error_occured)
    {
        for (uint i = 0; i < 9; i++)
        {
            clReleaseEvent(write_events[i]);
        }

        exit(EXIT_FAILURE);
    }

    // Wait for all buffer operations to complete
    clWaitForEvents(9, write_events.data());

    return write_events;
}

// OpenCL version of the renderer
void opencl_render(const scene &s, uchar4 *image, const OpenCLContext &cl_context, OpenCLBuffers buffers, cl_kernel path_trace_kernel, cl_kernel tonemap_kernel, cl_kernel path_trace_sample_kernel, cl_kernel reset_accumulation_kernel)
{
    // Define work dimensions
    size_t global_work_size[2];
    size_t local_work_size[2] = {8, 4};

    // Align work size to work group size
    global_work_size[0] = ((IMAGE_WIDTH + local_work_size[0] - 1) / local_work_size[0]) * local_work_size[0];
    global_work_size[1] = ((IMAGE_HEIGHT + local_work_size[1] - 1) / local_work_size[1]) * local_work_size[1];

    // Set path tracing kernel arguments
    cl_int err;
    size_t image_size = IMAGE_WIDTH * IMAGE_HEIGHT;

    uint2 dimensions = {IMAGE_WIDTH, IMAGE_HEIGHT};
    uint total_samples = SAMPLES_PER_PIXEL;
    cl_event reset_event;
    cl_event tonemap_event;
    cl_event readback_event;

    // Reset accumulation buffer
    err = clSetKernelArg(reset_accumulation_kernel, 0, sizeof(uint2), &dimensions);
    err |= clSetKernelArg(reset_accumulation_kernel, 1, sizeof(cl_mem), &buffers.accumulation_buffer);
    if (err != CL_SUCCESS)
    {
        printf("Error setting reset accumulation kernel arguments: %d (%s)\n", err, getCLErrorString(err).c_str());
        return;
    }

    err = clEnqueueNDRangeKernel(cl_context.commandQueue, reset_accumulation_kernel, 2,
                                 NULL, global_work_size, local_work_size, 0, NULL, &reset_event);
    if (err != CL_SUCCESS)
    {
        printf("Error executing reset accumulation kernel: %d (%s)\n", err, getCLErrorString(err).c_str());
        return;
    }

    // Process each sample
    std::vector<cl_event> sample_events(total_samples);

    // First sample waits on reset event
    uint sample_index = 0;
    err = clSetKernelArg(path_trace_sample_kernel, 0, sizeof(uint2), &dimensions);
    err |= clSetKernelArg(path_trace_sample_kernel, 1, sizeof(uint), &sample_index);
    err |= clSetKernelArg(path_trace_sample_kernel, 2, sizeof(cl_mem), &buffers.accumulation_buffer);
    err |= clSetKernelArg(path_trace_sample_kernel, 3, sizeof(cl_mem), &buffers.colors);
    err |= clSetKernelArg(path_trace_sample_kernel, 4, sizeof(uint), &total_samples);
    err |= clSetKernelArg(path_trace_sample_kernel, 5, sizeof(cl_mem), &buffers.subframes);
    err |= clSetKernelArg(path_trace_sample_kernel, 6, sizeof(cl_mem), &buffers.instances);
    err |= clSetKernelArg(path_trace_sample_kernel, 7, sizeof(cl_mem), &buffers.bvh_nodes);
    err |= clSetKernelArg(path_trace_sample_kernel, 8, sizeof(cl_mem), &buffers.bvh_links);
    err |= clSetKernelArg(path_trace_sample_kernel, 9, sizeof(cl_mem), &buffers.mesh_indices);
    err |= clSetKernelArg(path_trace_sample_kernel, 10, sizeof(cl_mem), &buffers.mesh_pos);
    err |= clSetKernelArg(path_trace_sample_kernel, 11, sizeof(cl_mem), &buffers.mesh_normal);
    err |= clSetKernelArg(path_trace_sample_kernel, 12, sizeof(cl_mem), &buffers.mesh_albedo);
    err |= clSetKernelArg(path_trace_sample_kernel, 13, sizeof(cl_mem), &buffers.mesh_material);

    if (err != CL_SUCCESS)
    {
        printf("Error setting path tracing kernel arguments: %d (%s)\n", err, getCLErrorString(err).c_str());
        clReleaseEvent(reset_event);
        return;
    }

    // Execute first sample with reset_event as dependency
    err = clEnqueueNDRangeKernel(cl_context.commandQueue, path_trace_sample_kernel, 2,
                                 NULL, global_work_size, local_work_size,
                                 1, &reset_event, &sample_events[0]);

    clReleaseEvent(reset_event); // We can release the reset event now

    if (err != CL_SUCCESS)
    {
        printf("Error executing first sample kernel: %d (%s)\n", err, getCLErrorString(err).c_str());
        return;
    }

    // Process remaining samples
    for (uint sample = 1; sample < total_samples; sample++)
    {
        // For remaining samples, re-use the same args but update sample index
        err = clSetKernelArg(path_trace_sample_kernel, 1, sizeof(uint), &sample);
        if (err != CL_SUCCESS)
        {
            printf("Error updating sample index: %d (%s)\n", err, getCLErrorString(err).c_str());
            // Clean up previous events
            for (uint i = 0; i < sample; i++)
            {
                clReleaseEvent(sample_events[i]);
            }
            return;
        }

        // Each sample depends on the previous sample
        err = clEnqueueNDRangeKernel(cl_context.commandQueue, path_trace_sample_kernel, 2,
                                     NULL, global_work_size, local_work_size,
                                     1, &sample_events[sample - 1], &sample_events[sample]);
        if (err != CL_SUCCESS)
        {
            printf("Error executing sample %d kernel: %d (%s)\n", sample, err, getCLErrorString(err).c_str());
            // Clean up previous events
            for (uint i = 0; i < sample; i++)
            {
                clReleaseEvent(sample_events[i]);
            }
            return;
        }

        // We can release the previous event once we've used it as a dependency
        if (sample > 1)
        { // Keep sample[0] for error handling
            clReleaseEvent(sample_events[sample - 2]);
        }
    }

    // Set tonemap kernel arguments
    err = clSetKernelArg(tonemap_kernel, 0, sizeof(cl_mem), &buffers.colors);
    err |= clSetKernelArg(tonemap_kernel, 1, sizeof(cl_mem), &buffers.output_image);
    if (err != CL_SUCCESS)
    {
        printf("Error setting tonemap kernel arguments: %d (%s)\n", err, getCLErrorString(err).c_str());
        // Clean up remaining events
        for (uint i = 0; i < total_samples; i++)
        {
            if (i != total_samples - 2)
            { // We already released other events
                clReleaseEvent(sample_events[i]);
            }
        }
        return;
    }

    // Execute tonemap after last sample
    err = clEnqueueNDRangeKernel(cl_context.commandQueue, tonemap_kernel, 2,
                                 NULL, global_work_size, NULL,
                                 1, &sample_events[total_samples - 1], &tonemap_event);

    // We can release the last sample event
    clReleaseEvent(sample_events[total_samples - 1]);
    if (sample_events.size() > 1)
    {
        clReleaseEvent(sample_events[total_samples - 2]); // Release the second-to-last event too
    }

    if (err != CL_SUCCESS)
    {
        printf("Error executing tonemap kernel: %d (%s)\n", err, getCLErrorString(err).c_str());
        return;
    }

    // Read back result
    err = clEnqueueReadBuffer(cl_context.commandQueue, buffers.output_image, CL_TRUE, 0,
                              image_size * sizeof(uchar4), image,
                              1, &tonemap_event, &readback_event);

    clReleaseEvent(tonemap_event);

    if (err != CL_SUCCESS)
    {
        printf("Error reading output image: %d (%s)\n", err, getCLErrorString(err).c_str());
        return;
    }

    // Final event release
    clReleaseEvent(readback_event);
}

int main(int argc, char **argv)
{
    // MPI initialization
    int rank = 0;
    int world_size = 1;

#ifdef USE_LUMI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (rank == 0)
    {
        printf("Running with MPI: %d processes\n", world_size);
    }
#endif

    // since I have my executable in /build folder
    std::filesystem::current_path(std::filesystem::path(__FILE__).parent_path());

    // Make sure all text parsing is unaffected by locale
    setlocale(LC_ALL, "C");

    scene s = load_scene();
    uint frame_count = get_animation_frame_count(s);

    // Distribute frames among MPI ranks when using MPI
    uint start_frame = 0;
    uint end_frame = frame_count;

#ifdef USE_LUMI
    uint frames_per_rank = frame_count / world_size;
    start_frame = rank * frames_per_rank;
    end_frame = (rank == world_size - 1) ? frame_count : start_frame + frames_per_rank;

    // Each rank only reports its own range
    printf("Rank %d processing frames %u to %u\n", rank, start_frame, end_frame - 1);
#endif

    // create the first frame here for the buffers
    setup_animation_frame(s, start_frame);

    // Initialize available OpenCL devices
    std::vector<OpenCLContext> gpu_contexts = initializeOpenCLDevices();
    bool use_opencl = !gpu_contexts.empty();

#ifndef USE_LUMI
    // Filter for NVIDIA GPUs only when running locally
    std::vector<OpenCLContext> filtered_contexts;
    for (const auto &context : gpu_contexts)
    {
        cl_device_type device_type;
        char vendor[256];
        clGetDeviceInfo(context.deviceId, CL_DEVICE_TYPE, sizeof(device_type), &device_type, NULL);
        clGetDeviceInfo(context.deviceId, CL_DEVICE_VENDOR, sizeof(vendor), vendor, NULL);

        std::string vendor_str(vendor);
        if (vendor_str.find("NVIDIA") != std::string::npos)
        {
            filtered_contexts.push_back(context);
            printf("Selected NVIDIA GPU: %s\n", vendor);
        }
    }
    gpu_contexts = filtered_contexts;
    use_opencl = !gpu_contexts.empty();
#else
    // When on server, HIP will be used for AMD GPUs
    // HIP initialization code would go here if using HIP directly
    // For now, we keep all OpenCL contexts as they're already filtered by initializeOpenCLDevices()
    printf("Running in server environment with all available GPUs\n");
#endif

#ifdef USE_LUMI
    // Collect total GPU count across all ranks
    int local_gpu_count = gpu_contexts.size();
    int total_gpu_count = 0;
    MPI_Reduce(&local_gpu_count, &total_gpu_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf("Total GPUs across all ranks: %d\n", total_gpu_count);
    }
#endif

    // Initialize kernels and buffers for each available GPU
    std::vector<cl_kernel> path_trace_kernels(gpu_contexts.size());
    std::vector<cl_kernel> tonemap_kernels(gpu_contexts.size());
    std::vector<cl_kernel> path_trace_sample_kernels(gpu_contexts.size());
    std::vector<cl_kernel> reset_accumulation_kernels(gpu_contexts.size());
    std::vector<std::array<OpenCLBuffers, 2>> gpu_buffers(gpu_contexts.size());

    for (size_t i = 0; i < gpu_contexts.size(); i++)
    {
        try
        {
            path_trace_kernels[i] = createKernel(gpu_contexts[i], "path_trace_pixel_kernel");
            tonemap_kernels[i] = createKernel(gpu_contexts[i], "tonemap_kernel");
            path_trace_sample_kernels[i] = createKernel(gpu_contexts[i], "path_trace_sample_kernel");
            reset_accumulation_kernels[i] = createKernel(gpu_contexts[i], "reset_accumulation_kernel");

            gpu_buffers[i][0] = create_render_buffers(s, gpu_contexts[i]);
            gpu_buffers[i][1] = create_render_buffers(s, gpu_contexts[i]);

            char device_name[256];
            clGetDeviceInfo(gpu_contexts[i].deviceId, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
            printf("Rank %d initialized GPU %zu: %s\n", rank, i, device_name);
        }
        catch (const std::exception &e)
        {
            printf("Failed to initialize GPU %zu: %s\n", i, e.what());
        }
    }

    std::unique_ptr<uchar4[]> image(new uchar4[IMAGE_WIDTH * IMAGE_HEIGHT]);

    // Process frames assigned to this rank
    for (uint frame_index = start_frame; frame_index < end_frame; ++frame_index)
    {
        // Select which GPU to use for this frame (round-robin)
        int gpu_index = 0;
        if (gpu_contexts.size() > 0)
        {
            gpu_index = (frame_index - start_frame) % gpu_contexts.size();
        }

        if (use_opencl && gpu_index < gpu_contexts.size())
        {
            int current_buffer = 0;
            int next_buffer = 1;

            // Prepare current frame
            setup_animation_frame(s, frame_index);
            std::array<cl_event, 9> update_events = update_render_buffers(s, gpu_buffers[gpu_index][current_buffer], gpu_contexts[gpu_index]);
            clWaitForEvents(update_events.size(), update_events.data());
            for (cl_event &event : update_events)
            {
                clReleaseEvent(event);
            }

            // Render the frame
            opencl_render(s, image.get(), gpu_contexts[gpu_index], gpu_buffers[gpu_index][current_buffer],
                          path_trace_kernels[gpu_index], tonemap_kernels[gpu_index],
                          path_trace_sample_kernels[gpu_index], reset_accumulation_kernels[gpu_index]);
        }
        else
        {
            setup_animation_frame(s, frame_index);
            baseline_render(s, image.get());
        }

        // Create string for the index number of the frame with leading zeroes
        std::string index_str = std::to_string(frame_index);
        while (index_str.size() < 4)
            index_str.insert(index_str.begin(), '0');

#ifdef USE_LUMI
        // Each rank writes to its own output directory
        std::string output_dir = "output/rank_" + std::to_string(rank);
        std::filesystem::create_directories(output_dir);
        std::string output_path = output_dir + "/frame_" + index_str + ".bmp";
#else
        std::string output_path = "output/frame_" + index_str + ".bmp";
#endif

        // Write output image
        write_bmp(output_path.c_str(), IMAGE_WIDTH, IMAGE_HEIGHT, 4, IMAGE_WIDTH * 4, (uint8_t *)image.get());
    }

    // Release OpenCL resources
    if (use_opencl)
    {
        for (size_t g = 0; g < gpu_contexts.size(); g++)
        {
            for (int i = 0; i < 2; i++)
            {
                clReleaseMemObject(gpu_buffers[g][i].output_image);
                clReleaseMemObject(gpu_buffers[g][i].colors);
                clReleaseMemObject(gpu_buffers[g][i].subframes);
                clReleaseMemObject(gpu_buffers[g][i].instances);
                clReleaseMemObject(gpu_buffers[g][i].bvh_nodes);
                clReleaseMemObject(gpu_buffers[g][i].bvh_links);
                clReleaseMemObject(gpu_buffers[g][i].mesh_indices);
                clReleaseMemObject(gpu_buffers[g][i].mesh_pos);
                clReleaseMemObject(gpu_buffers[g][i].mesh_normal);
                clReleaseMemObject(gpu_buffers[g][i].mesh_albedo);
                clReleaseMemObject(gpu_buffers[g][i].mesh_material);
                clReleaseMemObject(gpu_buffers[g][i].accumulation_buffer);
            }

            clReleaseKernel(path_trace_kernels[g]);
            clReleaseKernel(tonemap_kernels[g]);
            clReleaseKernel(path_trace_sample_kernels[g]);
            clReleaseKernel(reset_accumulation_kernels[g]);

            clReleaseProgram(gpu_contexts[g].program);
            clReleaseCommandQueue(gpu_contexts[g].commandQueue);
            clReleaseContext(gpu_contexts[g].context);
        }
    }

#ifdef USE_LUMI
    MPI_Finalize();
#endif

    return 0;
}