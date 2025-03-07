#include "path_tracer.hh"

// Process one sample per thread instead of all samples per thread
__kernel void path_trace_sample_kernel(
    uint2 dimensions,
    uint sample_index,
    __global float3* accumulation_buffer,
    __global float3* output_colors,
    uint total_samples,

    // Scene data
    __global const subframe* subframes,
    __global const tlas_instance* instances,
    __global const bvh_node* node_array,
    __global const bvh_link* link_array,
    __global const uint* mesh_indices,
    __global const float3* mesh_pos,
    __global const float3* mesh_normal,
    __global const float4* mesh_albedo,
    __global const float4* mesh_material)
{
    // Get global IDs
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    
    // Check if within image bounds
    if (x >= dimensions.x || y >= dimensions.y)
        return;
    
    uint pixel_idx = y * dimensions.x + x;
    uint2 xy = {x, y};
    
    // Process just one sample
    float3 sample_color = path_trace_pixel(
        xy,
        sample_index,
        subframes,
        instances,
        node_array,
        link_array,
        mesh_indices,
        mesh_pos,
        mesh_normal,
        mesh_albedo,
        mesh_material);
    
    // Accumulate to buffer
    accumulation_buffer[pixel_idx] += sample_color;
    
    // If this is the last sample, normalize and copy to output
    if (sample_index == total_samples - 1) {
        output_colors[pixel_idx] = accumulation_buffer[pixel_idx] / total_samples;
    }
}

// Original implementation kept for compatibility 
__kernel void path_trace_pixel_kernel(
    uint2 dimensions,
    uint samples_per_pixel,
    __global float3* output_colors,

    // Scene data
    __global const subframe* subframes,
    __global const tlas_instance* instances,
    __global const bvh_node* node_array,
    __global const bvh_link* link_array,
    __global const uint* mesh_indices,
    __global const float3* mesh_pos,
    __global const float3* mesh_normal,
    __global const float4* mesh_albedo,
    __global const float4* mesh_material)
{
    // Get global IDs
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    
    // Check if within image bounds
    if (x >= dimensions.x || y >= dimensions.y)
        return;
    
    uint pixel_idx = y * dimensions.x + x;
    uint2 xy = {x, y};
    float3 color = {0.0f, 0.0f, 0.0f};
    
    // Accumulate samples
    for (uint j = 0; j < samples_per_pixel; ++j) {
        color += path_trace_pixel(
            xy,
            j,
            subframes,
            instances,
            node_array,
            link_array,
            mesh_indices,
            mesh_pos,
            mesh_normal,
            mesh_albedo,
            mesh_material);
    }
    
    // Average the samples
    color /= samples_per_pixel;
    
    // Store the result
    output_colors[pixel_idx] = color;
}

// Tonemap kernel - converts HDR colors to 8-bit BGRA
__kernel void tonemap_kernel(
    __global const float3* input_colors,
    __global uchar4* output_image)
{
    // Get global IDs
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint pixel_idx = y * get_global_size(0) + x;
    
    // Apply tonemapping
    output_image[pixel_idx] = tonemap_pixel(input_colors[pixel_idx]);
}

// Reset the accumulation buffer to zero
__kernel void reset_accumulation_kernel(
    uint2 dimensions,
    __global float3* accumulation_buffer)
{
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    
    if (x >= dimensions.x || y >= dimensions.y)
        return;
        
    uint pixel_idx = y * dimensions.x + x;
    accumulation_buffer[pixel_idx] = (float3)(0.0f, 0.0f, 0.0f);
}