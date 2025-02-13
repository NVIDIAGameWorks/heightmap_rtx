> [!IMPORTANT]
> This project has been archived and is no longer maintained. The vulkan
> extension `VK_NV_displacement_micromap` is no longer available.
>
> We recommend exploring [**NVIDIA RTX Mega
> Geometry**](https://developer.nvidia.com/blog/nvidia-rtx-mega-geometry-now-available-with-new-vulkan-samples/),
> which can provide similar functionality with greater flexibility. See
> [vk_tessellated_clusters](https://github.com/nvpro-samples/vk_tessellated_clusters),
> which demonstrates raytracing displacement with Vulkan.

# Heightmap RTX

![raytraced displacement using heightmap_rtx](doc/preview.jpg "Raytraced
displacement in micromesh_toolbox using this library. Turtle Barbarian model by
Jesse Sandifer, courtesy of Autodesk.")
<br/><sub><sup>Raytraced displacement in <a
href="https://github.com/NVIDIAGameWorks/Displacement-MicroMap-Toolkit">micromesh_toolbox</a>
using this library. "Turtle Barbarian" model by Jesse Sandifer, courtesy of
Autodesk.</sup></sub>

Heightmap RTX is a small Vulkan library to displace raytraced triangles with a
heightmap. It uses [NVIDIA
Micro-Mesh](https://developer.nvidia.com/rtx/ray-tracing/micro-mesh)
([Toolkit](https://github.com/NVIDIAGameWorks/Displacement-MicroMap-Toolkit))
internally. It can also be seen as a cheap runtime micromap baker.

[VK_NV_displacement_micromap](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_NV_displacement_micromap.html)
is required. See [device
support](https://vulkan.gpuinfo.org/listdevicescoverage.php?extension=VK_NV_displacement_micromap&platform=all).

## API Guide

1. Include [CMakeLists.txt](CMakeLists.txt) to provide the `heightmap_rtx` static library. The C interface is defined in [heightmap_rtx.h](include/heightmap_rtx/heightmap_rtx.h).
2. Create a `HrtxMap` object from an image and the geometry that would normally be added to the acceleration structure build. This depends on a common `HrtxPipeline` object.
3. Set the geometry's `pNext` to the micromap description returned by `hrtxMapDesc(HrtxMap)` before building the acceleration structure.
4. Make sure the vulkan raytracing pipeline is created with `VK_PIPELINE_CREATE_RAY_TRACING_DISPLACEMENT_MICROMAP_BIT_NV`.

For a complete example, see the nvpro_core sample [vk_raytrace_displacement](https://github.com/nvpro-samples/vk_raytrace_displacement).

```c++
#include <heightmap_rtx.h>

...

// Library input data
geometry.vertexTexcoordBuffer.address = ...;
geometry.vertexNormalBufferHVec4.address = ...;
VkAccelerationStructureGeometryTrianglesDataKHR triangles = {...};
VkAccelerationStructureBuildRangeInfoKHR        buildRange = {...};
VkImage                                         heightmapImage = ...;
VkDescriptorImageInfo                           heightmapImageInfo = ...;
float                                           heightmapBias = 0.0f;
float                                           heightmapScale = 1.0f;

// A callback must be provided for buffer allocation (this example uses nvpro_core's AllocVma).
HrtxAllocatorCallbacks allocatorCallbacks{
    [](const VkBufferCreateInfo bufferCreateInfo, const VkMemoryPropertyFlags memoryProperties, void* userPtr) {
      auto alloc  = reinterpret_cast<nvvkhl::AllocVma*>(userPtr);
      auto result = new nvvk::Buffer();
      *result     = alloc->createBuffer(bufferCreateInfo, memoryProperties);
      return &result->buffer;  // return pointer to member
    },
    [](VkBuffer* bufferPtr, void* userPtr) {
      auto alloc = reinterpret_cast<nvvkhl::AllocVma*>(userPtr);
      // reconstruct from pointer to member
      auto nvvkBuffer = reinterpret_cast<nvvk::Buffer*>(reinterpret_cast<char*>(bufferPtr) - offsetof(nvvk::Buffer, buffer));
      alloc->destroy(*nvvkBuffer);
      delete nvvkBuffer;
    },
    alloc,
};

// This example assumes these vulkan objects exist
VkPhysicalDevice physicalDevice = ...;
VkDevice         device = ...;
VkCommandBuffer  cmd = ...;

// Create a HrtxPipeline object. This holds the shader and resources for baking
HrtxPipeline pipeline;
HrtxPipelineCreate hrtxPipelineCreate{
    physicalDevice, device, allocatorCallbacks, VK_NULL_HANDLE, nullptr, nullptr, VK_NULL_HANDLE,
    [](VkResult result) {
      ... handle error
    }};
if(hrtxCreatePipeline(cmd, &hrtxPipelineCreate, &pipeline) != VK_SUCCESS)
{
  ... handle error
}

// Create a HrtxMap object from an image to displace some geometry
// This adds a call to 'cmd' to execute a compute shader to bake a micromap
... use hrtxBarrierFlags() to synchronize inputs if needed
HrtxMap        hrtxMap;
const uint32_t subdivLevel = 5;
HrtxMapCreate  mapCreate{
    &triangles,
    buildRange.primitiveCount,
    geometry.vertexTexcoordBuffer.address,
    VK_FORMAT_R32G32_SFLOAT,
    sizeof(float) * 2,
    geometry.vertexNormalBuffer.address,  // displacement directions
    VK_FORMAT_R32G32B32_SFLOAT,
    sizeof(float) * 3,
    heightmapImageInfo,
    heightmapBias,
    heightmapScale,
    subdivLevel,
};
if(hrtxCmdCreateMap(cmd, pipeline, &mapCreate, &hrtxMap) != VK_SUCCESS)
{
  ... handle error
}

// Library output is a micromap
VkAccelerationStructureTrianglesDisplacementMicromapNV micromapDesc = hrtxMapDesc(hrtxMap);
triangles.pNext = &micromapDesc;

// Build the acceleration structure normally
... vkCmdBuildAccelerationStructureNV()

// Make sure the pipeline has micromaps enabled
VkRayTracingPipelineCreateInfoKHR pipelineCreateInfo = {...};
pipelineCreateInfo.flags |= VK_PIPELINE_CREATE_RAY_TRACING_DISPLACEMENT_MICROMAP_BIT_NV;
```

## Rendering Differences

Micro-Mesh was designed to be as seamless as possible. By setting the pipeline
flag VK_PIPELINE_CREATE_RAY_TRACING_DISPLACEMENT_MICROMAP_BIT_NV and
VkAccelerationStructureGeometryTrianglesDataKHR::pNext, rays will just start
hitting displaced geometry. One caveat is when existing shaders interpolate the
original triangle positions. An alternative is to use gl_WorldRayOriginEXT +
gl_WorldRayDirectionEXT * gl_HitTEXT, which will produce a position on the
displaced surface. gl_HitMicroTriangleVertexPositionsNV can also be used if it’s
necessary to specialize. Micromesh also produces a gl_HitKindEXT of
gl_HitKindFrontFacingMicroTriangleNV and gl_HitKindBackFacingMicroTriangleNV
instead of gl_HitKindFrontFacingTriangleEXT and gl_HitKindBackFacingTriangleEXT.

## Limitations

- The baked micromap is not well compressed, using lossless unorm11 packed
  encoding.
- Displacement bounds are not generated, possibly resulting in poor raytracing
  performance.
- This library supports a maximum subdivision level of 5, so each triangle can
  be subdivided into at most 1024 micro-triangles. This might be too
  low-resolution for some heightmaps. Larger libraries like the Micro-Mesh
  Toolkit can pre-tessellate the input mesh to avoid this limitation.
- Micromesh direction vectors are not normalized after interpolation and this is
  not compensated for during baking, resulting in flatter displacement in across
  triangles of high curvature.
- Discontinuities across UV seams are not stitched, which can produce small
  cracks.
- Hard edge normals can produce large cracks. It is up to the application to
  provide smooth displacement direction vectors.

The intent of this library is to give quick easy access to
[VK_NV_displacement_micromap](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_NV_displacement_micromap.html).
Ultimately, baking a micromap offline with displacement bounds fitting and
compression optimization will give better results, e.g. using `micromesh_tool`
from the
[Toolkit](https://github.com/NVIDIAGameWorks/Displacement-MicroMap-Toolkit).
