#include "path_tracer.hh"

__kernel void path_trace_pixel_kernel(
    uint2 dimensions,
    uint samples_per_pixel,
    __global float3* output_colors,
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
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    
    if (x >= dimensions.x || y >= dimensions.y)
        return;
    
    uint pixel_idx = y * dimensions.x + x;
    uint2 xy = {x, y};
    float3 color = {0.0f, 0.0f, 0.0f};
    
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
    
    color /= samples_per_pixel;
    
    output_colors[pixel_idx] = color;
}

__kernel void tonemap_kernel(
    __global const float3* input_colors,
    __global uchar4* output_image)
{
    uint x = get_global_id(0);
    uint y = get_global_id(1);
    uint pixel_idx = y * get_global_size(0) + x;
    
    output_image[pixel_idx] = tonemap_pixel(input_colors[pixel_idx]);
}
