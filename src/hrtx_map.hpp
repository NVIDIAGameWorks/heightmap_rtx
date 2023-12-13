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

#pragma once

#include "vulkan/vulkan_core.h"
#include <cstdint>
#include <heightmap_rtx.h>
#include <hrtx_pipeline.hpp>
#include <context.hpp>
#include <memory>
#include <vulkan_objects.hpp>

inline VkDeviceSize microVertsPerTriangle(uint32_t subdivisionLevel)
{
  uint32_t microVertsPerEdge = (1U << subdivisionLevel) + 1U;
  return (microVertsPerEdge * (microVertsPerEdge + 1U)) / 2U;
}

inline VkDeviceSize baryLosslessBlocks(const HrtxMapCreate& create)
{
  uint32_t micromap6464BlocksPerTriangle = 1U << ((std::max(3U, create.subdivisionLevel) - 3U) * 2U);
  uint32_t micromapBlockCount            = create.primitiveCount * micromap6464BlocksPerTriangle;
  return micromapBlockCount;
}

inline uint32_t tightIndexStrideBytes(VkIndexType type)
{
  switch(type)
  {
    case VK_INDEX_TYPE_UINT8_EXT:
      return 1;
    case VK_INDEX_TYPE_UINT16:
      return 2;
    case VK_INDEX_TYPE_UINT32:
      return 4;
    default:
      return 0;
  }
}

template <class T>
constexpr T align_up(T x, T alignPOT) noexcept
{
  return (x + (alignPOT - 1)) & ~(alignPOT - 1);
}

VkDeviceSize micromapScratchAlignment(const HrtxContext& ctx)
{
  // For each element of pInfos, its scratchData.deviceAddress member must: be
  // a multiple of
  // VkPhysicalDeviceAccelerationStructurePropertiesKHR::minAccelerationStructureScratchOffsetAlignment
  VkPhysicalDeviceAccelerationStructurePropertiesKHR asProps{};
  asProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;
  VkPhysicalDeviceProperties2 props2{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
      &asProps,
      {},
  };
  ctx.vk.vkGetPhysicalDeviceProperties2(ctx.physicalDevice, &props2);
  VkDeviceSize scratchAlignment = asProps.minAccelerationStructureScratchOffsetAlignment;
  return scratchAlignment;
}

class BaryDataVk
{
public:
  BaryDataVk(VkCommandBuffer cmd, const HrtxPipeline_T& hrtxPipeline, const HrtxMapCreate& create)
      : m_heightmapDescriptors(hrtxPipeline.createHeightmapDescriptors(create.heightmapImage))
      , m_triangleCount(create.primitiveCount)
      , m_subdivisionLevel(create.subdivisionLevel)
      , m_baryValues(hrtxPipeline.ctx(),
                     baryLosslessBlocks(create) * 64,
                     VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                         | VK_BUFFER_USAGE_MICROMAP_BUILD_INPUT_READ_ONLY_BIT_EXT)
      , m_baryTriangles(hrtxPipeline.ctx(),
                        create.primitiveCount * sizeof(VkMicromapTriangleEXT),
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                            | VK_BUFFER_USAGE_MICROMAP_BUILD_INPUT_READ_ONLY_BIT_EXT)
  {
    // Clear the displacements as the shader uses atomicOr()s to fill them.
    m_baryValues.clear(cmd);
    m_baryTriangles.clear(cmd);
    memoryBarrier(cmd, hrtxPipeline.ctx(), VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

    assert(create.triangles->indexType == VK_INDEX_TYPE_UINT32);
    assert(create.textureCoordsFormat == VK_FORMAT_R32G32_SFLOAT);
    assert(create.textureCoordsStride % (sizeof(float) * 2) == 0);

    shaders::CompressPushConstants pushConstants{create.textureCoordsBuffer.deviceAddress,
                                                 create.triangles->indexData.deviceAddress,
                                                 m_baryValues.address(),
                                                 m_baryTriangles.address(),
                                                 static_cast<uint32_t>(create.textureCoordsStride / (sizeof(float) * 2)),
                                                 m_triangleCount,
                                                 m_subdivisionLevel};

    const int32_t microTrianglesPerBlock = 45;
    int32_t       threadCount =
        static_cast<int32_t>(m_subdivisionLevel > 3 ? microTrianglesPerBlock * baryLosslessBlocks(create) :
                                                      m_triangleCount * microVertsPerTriangle(m_subdivisionLevel));
    int32_t groupCount = (threadCount + COMPRESS_WORKGROUP_SIZE - 1) / COMPRESS_WORKGROUP_SIZE;
    hrtxPipeline.bindAndDispatch(cmd, *m_heightmapDescriptors, pushConstants, groupCount);

    // Barrier between the compute shader and vkCmdBuildMicromapsEXT().
    memoryBarrier2(cmd, hrtxPipeline.ctx(), VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_MICROMAP_BUILD_BIT_EXT, VK_ACCESS_2_MICROMAP_READ_BIT_EXT);
  }
  const Buffer& values() const { return m_baryValues; }
  const Buffer& triangles() const { return m_baryTriangles; }
  uint32_t      triangleCount() const { return m_triangleCount; }
  uint32_t      subdivisionLevel() const { return m_subdivisionLevel; }

  BaryDataVk(const BaryDataVk& other)            = delete;
  BaryDataVk& operator=(const BaryDataVk& other) = delete;

public:
  std::unique_ptr<SingleDescriptorSet> m_heightmapDescriptors;
  uint32_t                             m_triangleCount;
  uint32_t                             m_subdivisionLevel;
  Buffer                               m_baryValues;
  Buffer                               m_baryTriangles;
};

class Micromap
{
public:
  Micromap(const HrtxContext& ctx, VkDeviceSize size)
      : m_ctx(ctx)
      , m_data(ctx, size, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_MICROMAP_STORAGE_BIT_EXT)
  {
    VkMicromapCreateInfoEXT mmCreateInfo = {
        VK_STRUCTURE_TYPE_MICROMAP_CREATE_INFO_EXT,
        nullptr,
        0,
        m_data,
        0,
        m_data.size(),
        VK_MICROMAP_TYPE_DISPLACEMENT_MICROMAP_NV,
        0ull,  //
    };
    ctx.checkResult(ctx.vk.vkCreateMicromapEXT(ctx.device, &mmCreateInfo, nullptr, &m_micromap));
  }
  ~Micromap() noexcept { m_ctx.vk.vkDestroyMicromapEXT(m_ctx.device, m_micromap, m_ctx.allocator.systemAllocator); }
  operator const VkMicromapEXT&() const { return m_micromap; }

  Micromap(const Micromap& other)            = delete;
  Micromap& operator=(const Micromap& other) = delete;

private:
  const HrtxContext& m_ctx;
  Buffer             m_data;
  VkMicromapEXT      m_micromap;
};

class BuiltMicromap
{
public:
  BuiltMicromap(VkCommandBuffer cmd, const HrtxContext& ctx, const BaryDataVk& baryDataVk)
  {
    // One format for all triangles
    m_usages.push_back(VkMicromapUsageEXT{baryDataVk.triangleCount(), baryDataVk.subdivisionLevel(),
                                          VK_DISPLACEMENT_MICROMAP_FORMAT_64_TRIANGLES_64_BYTES_NV});

    // Ask vulkan for the required micromap buffer sizes
    VkMicromapBuildInfoEXT buildInfo{
        VK_STRUCTURE_TYPE_MICROMAP_BUILD_INFO_EXT,
        nullptr,
        VK_MICROMAP_TYPE_DISPLACEMENT_MICROMAP_NV,
        0,
        VK_BUILD_MICROMAP_MODE_BUILD_EXT,
        VK_NULL_HANDLE,
        static_cast<uint32_t>(m_usages.size()),
        m_usages.data(),
        nullptr,
        VkDeviceOrHostAddressConstKHR{0ull},
        VkDeviceOrHostAddressKHR{0ull},
        VkDeviceOrHostAddressConstKHR{0ull},
        0ull,
    };
    VkMicromapBuildSizesInfoEXT sizeInfo = {
        VK_STRUCTURE_TYPE_MICROMAP_BUILD_SIZES_INFO_EXT,
        nullptr,
        0ull,
        0ull,
        VK_FALSE,  //
    };
    ctx.vk.vkGetMicromapBuildSizesEXT(ctx.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &sizeInfo);
    assert(sizeInfo.micromapSize && "sizeInfo.micromeshSize was zero");

    m_micromap = std::make_unique<Micromap>(ctx, sizeInfo.micromapSize);

    // The driver may use this
    VkDeviceSize scratchSize = align_up(std::max(sizeInfo.buildScratchSize, VkDeviceSize(4)), micromapScratchAlignment(ctx));
    m_micromapScratch        = std::make_unique<Buffer>(
        ctx, scratchSize, VK_BUFFER_USAGE_MICROMAP_BUILD_INPUT_READ_ONLY_BIT_EXT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

    // Build the micromap structure
    buildInfo.dstMicromap                 = *m_micromap;
    buildInfo.scratchData.deviceAddress   = m_micromapScratch->address();
    buildInfo.data.deviceAddress          = baryDataVk.values().address();
    buildInfo.triangleArray.deviceAddress = baryDataVk.triangles().address();
    buildInfo.triangleArrayStride         = sizeof(VkMicromapTriangleEXT);
    ctx.vk.vkCmdBuildMicromapsEXT(cmd, 1, &buildInfo);

    memoryBarrier2(cmd, ctx, VK_PIPELINE_STAGE_2_MICROMAP_BUILD_BIT_EXT, VK_ACCESS_2_MICROMAP_WRITE_BIT_EXT,
                   VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
  }
  const Micromap&                        micromap() const { return *m_micromap; }
  const std::vector<VkMicromapUsageEXT>& usages() const { return m_usages; }

  BuiltMicromap(const BuiltMicromap& other)            = delete;
  BuiltMicromap& operator=(const BuiltMicromap& other) = delete;

private:
  // TODO: a lot of this could be freed after the command buffer is submitted,
  // but there's no API. A cleanup thread with a fence might be inefficient
  // and/or overkill.
  std::unique_ptr<Micromap>       m_micromap;
  std::vector<VkMicromapUsageEXT> m_usages;
  std::unique_ptr<Buffer>         m_micromapScratch;
};

struct HrtxMap_T
{
  HrtxMap_T(VkCommandBuffer cmd, const HrtxPipeline_T& hrtxPipeline, const HrtxMapCreate& create)
      : m_biasAndScale(hrtxPipeline.ctx(), sizeof(float) * 2, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT)
      , m_directionsBuffer(create.directionsBuffer)
      , m_directionsFormat(create.directionsFormat)
      , m_directionsStride(create.directionsStride)
      , m_baryData(cmd, hrtxPipeline, create)
      , m_builtMicromap(cmd, hrtxPipeline.ctx(), m_baryData)
  {
    float biasScale[2]{create.heightmapBias, create.heightmapScale};
    m_biasAndScale.update(cmd, &biasScale);

    // Barrier between writing m_biasAndScale and reading in the user's BVH
    // build. vkCmdCopyBuffer() is treated as a "transfer" operation.
    memoryBarrier2(cmd, hrtxPipeline.ctx(), VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
  }

  HrtxMap_T(const HrtxMap_T& other)            = delete;
  HrtxMap_T& operator=(const HrtxMap_T& other) = delete;

  VkAccelerationStructureTrianglesDisplacementMicromapNV descriptor()
  {
    return VkAccelerationStructureTrianglesDisplacementMicromapNV{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_TRIANGLES_DISPLACEMENT_MICROMAP_NV,
        nullptr,
        VK_FORMAT_R32G32_SFLOAT,
        m_directionsFormat,
        {m_biasAndScale.address()},
        0,  // same bias and scale for all directions
        m_directionsBuffer,
        m_directionsStride,
        {},
        0,
        VK_INDEX_TYPE_NONE_KHR,
        {},
        0,
        0,
        static_cast<uint32_t>(m_builtMicromap.usages().size()),
        m_builtMicromap.usages().data(),
        nullptr,
        m_builtMicromap.micromap(),
    };
  }

private:
  Buffer                        m_biasAndScale;
  VkDeviceOrHostAddressConstKHR m_directionsBuffer;
  VkFormat                      m_directionsFormat;
  VkDeviceSize                  m_directionsStride;
  BaryDataVk                    m_baryData;
  BuiltMicromap                 m_builtMicromap;
};
