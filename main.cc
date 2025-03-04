#include "scene.hh"
#include "path_tracer.hh"
#include "bmp.hh"
#include <clocale>
#include <memory>

#include <filesystem>
#include <omp.h>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

// Renders the given scene into an image using path tracing.
void baseline_render(const scene& s, uchar4* image)
{
    for(uint i = 0; i < IMAGE_WIDTH * IMAGE_HEIGHT; ++i)
    {
        uint x = i % IMAGE_WIDTH;
        uint y = i / IMAGE_WIDTH;

        float3 color = {0,0,0};

#pragma omp parallel for
        for(uint j = 0; j < SAMPLES_PER_PIXEL; ++j)
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
                s.mesh_buf.material.data()
            );
        }

        color /= SAMPLES_PER_PIXEL;

        image[i] = tonemap_pixel(color);
    }
}

int main()
{
    // since i have my executable in /build folder
    std::filesystem::current_path(std::filesystem::path(__FILE__).parent_path());

    // Make sure all text parsing is unaffected by locale
    setlocale(LC_ALL, "C");

    scene s = load_scene();

    std::unique_ptr<uchar4[]> image(new uchar4[IMAGE_WIDTH * IMAGE_HEIGHT]);

    uint frame_count = get_animation_frame_count(s);
    frame_count = 1; // for testing
    for(uint frame_index = 0; frame_index < frame_count; ++frame_index)
    {
        // Update scene state for the current frame & render it
        setup_animation_frame(s, frame_index);
        baseline_render(s, image.get());

        // Create string for the index number of the frame with leading zeroes.
        std::string index_str = std::to_string(frame_index);
        while(index_str.size() < 4) index_str.insert(index_str.begin(), '0');

        // Write output image
        write_bmp(
            ("output/frame_"+index_str+".bmp").c_str(),
            IMAGE_WIDTH, IMAGE_HEIGHT, 4, IMAGE_WIDTH * 4,
            (uint8_t*)image.get()
        );
    }

    return 0;
}
