/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Heightmap RTX is a vulkan middle layer library to apply heightmap
// displacement from a vulkan image to raytraced geometry. It does this using
// VK_NV_displacement_micromap. The functions in this header ultimately allow
// extending a VkAccelerationStructureGeometryKHR structure with micromap
// displacement before building the accelleration structure.

#ifndef HEIGHTMAP_RTX_H
#define HEIGHTMAP_RTX_H

#ifdef __cplusplus
extern "C" {
#endif

#include <vulkan/vulkan_core.h>

#if defined(VK_ENABLE_BETA_EXTENSIONS) || !defined(VK_NV_displacement_micromap)
#include <vulkan/vulkan_beta.h>
#endif

// Common resources such as shaders for creating HrtxMap objects.
// Usage:
//  pipeline = hrtxCreatePipeline(...);
//  ... image barrier based on hrtxBarrierFlags()
//  hrtxMap = hrtxCmdCreateMap(cmd, pipeline, ...)
//  ... memory barrier based on hrtxBarrierFlags()
typedef struct HrtxPipeline_T* HrtxPipeline;

// Heightmap displacement object for raytracing displaced geometry.
// Usage:
//  micromapDesc = hrtxMapDesc(hrtxMap);
//  accelerationStructureGeometry.geometry.triangles.pNext = &micromapDesc;
//  ...
//  vkCmdWriteAccelerationStructuresPropertiesKHR(...)
//  ... create pipeline with VK_PIPELINE_CREATE_RAY_TRACING_DISPLACEMENT_MICROMAP_BIT_NV
typedef struct HrtxMap_T* HrtxMap;

typedef VkBuffer* (*PFN_hrtxCreateBuffer)(const VkBufferCreateInfo    bufferCreateInfo,
                                          const VkMemoryPropertyFlags memoryProperties,
                                          void*                       userPtr);
typedef void (*PFN_hrtxDestroyBuffer)(VkBuffer* bufferPtr, void* userPtr);
typedef void (*PFN_hrtxCheckVkResult)(VkResult result);

typedef struct HrtxAllocatorCallbacks
{
  PFN_hrtxCreateBuffer  createBuffer;
  PFN_hrtxDestroyBuffer destroyBuffer;
  void*                 userPtr;

  // Optional
  const VkAllocationCallbacks* systemAllocator;
} HrtxAllocatorCallbacks;

typedef struct HrtxPipelineCreate
{
  VkPhysicalDevice       physicalDevice;
  VkDevice               device;
  HrtxAllocatorCallbacks allocator;

  // Optional: when set, vulkan functions are loaded dynamically
  VkInstance                instance;
  PFN_vkGetInstanceProcAddr getInstanceProcAddr;
  PFN_vkGetDeviceProcAddr   getDeviceProcAddr;

  // Optional: cache internal shaders
  VkPipelineCache pipelineCache;

  // Optional: callback to catch any failures from internal vulkan calls, e.g. to
  // throw an exception from and abort creating displacement.
  PFN_hrtxCheckVkResult checkResultCallback;
} HrtxPipelineCreate;

// Takes a command buffer that will be filled with initialization operations,
// e.g. compiling shaders and device transfers for common data used to create
// HrtxMap objects. Memory barriers for these are inserted into cmd.
VkResult hrtxCreatePipeline(VkCommandBuffer cmd, const HrtxPipelineCreate* create, HrtxPipeline* hrtxPipeline);

typedef struct HrtxMapCreate
{
  // Currently only VK_INDEX_TYPE_UINT32 is supported
  // Indices must have VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT set
  const VkAccelerationStructureGeometryTrianglesDataKHR* triangles;
  uint32_t                                               primitiveCount;
  // Currently only VK_FORMAT_R32G32_SFLOAT is supported
  // Texture coords must have VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT set
  VkDeviceOrHostAddressConstKHR textureCoordsBuffer;
  VkFormat                      textureCoordsFormat;
  VkDeviceSize                  textureCoordsStride;
  // Currently only VK_FORMAT_R16G16B16A16_SFLOAT is supported
  VkDeviceOrHostAddressConstKHR directionsBuffer;
  VkFormat                      directionsFormat;
  VkDeviceSize                  directionsStride;
  VkDescriptorImageInfo         heightmapImage;
  float                         heightmapBias;
  float                         heightmapScale;
  uint32_t                      subdivisionLevel;
} HrtxMapCreate;

void hrtxDestroyPipeline(HrtxPipeline hrtxPipeline);

// Barriers for the following input data that must be inserted before calls to
// hrtxCmdCreateMap().
//
// - HrtxMapCreate::textureCoordsBuffer
// - HrtxMapCreate::directionsBuffer
// - HrtxMapCreate::heightmapImage
//
// Barriers for resources created and returned by hrtxMapDesc() will be inserted
// during HrtxMap creation as it is assumed these will be passed to an
// acceleration structure build at some point. Out parameters that are null will
// be ignored.
void hrtxBarrierFlags(VkPipelineStageFlags2* textureCoordsDstStageMask,
                      VkAccessFlags2*        textureCoordsDstAccessMask,
                      VkPipelineStageFlags2* directionsDstStageMask,
                      VkAccessFlags2*        directionsDstAccessMask,
                      VkImageLayout*         heightmapLayout);

VkResult hrtxCmdCreateMap(VkCommandBuffer cmd, HrtxPipeline hrtxPipeline, const HrtxMapCreate* create, HrtxMap* hrtxMap);

void hrtxDestroyMap(HrtxMap hrtxMap);

// See definition of HrtxMap for usage
// NOTE: VK_PIPELINE_CREATE_RAY_TRACING_DISPLACEMENT_MICROMAP_BIT_NV must be set
// on the raytracing pipeline
VkAccelerationStructureTrianglesDisplacementMicromapNV hrtxMapDesc(HrtxMap hrtxMap);

#ifdef __cplusplus
}
#endif

#endif  // HEIGHTMAP_RTX_H
