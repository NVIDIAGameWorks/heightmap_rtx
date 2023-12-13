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

#include <cstdint>
#include <heightmap_rtx.h>
#include <array>
#include <memory>
#include <algorithm>
#include <vulkan/vulkan_core.h>
#include <context.hpp>
#include <vulkan_objects.hpp>
#include <vulkan_bindings.hpp>
#include <compress.comp.h>
#include <bird_curve_table.h>

namespace shaders {
#include "shader_definitions.h"
};

class BlockToBirdUVTable : public std::array<BaryUV16, birdVertexToBaryTableOffsets[4] + (4 + 16) * 45 + 1>
{
public:
  BlockToBirdUVTable()
  {
    // Subdivision levels 1 to 3 are all within one compression block
    auto next = std::copy(birdVertexToBaryTable, birdVertexToBaryTable + birdVertexToBaryTableOffsets[4], this->begin());

    // For larger subdivision levels, birdVertexToBaryTable coordinates need to be
    // duplicated to account for shared edges between blocks.
    next = std::transform(&birdIndexBlockLocalToGlobal[0][0], &birdIndexBlockLocalToGlobal[4 - 1][45], next,
                          [](const uint32_t& globalBirdIndex) {
                            return birdVertexToBaryTable[birdVertexToBaryTableOffsets[4] + globalBirdIndex];
                          });
    next = std::transform(&birdIndexBlockLocalToGlobal[4][0], &birdIndexBlockLocalToGlobal[20 - 1][45], next,
                          [](const uint32_t& globalBirdIndex) {
                            return birdVertexToBaryTable[birdVertexToBaryTableOffsets[5] + globalBirdIndex];
                          });
    BaryUV16 padding{};  // vkCmdUpdateBuffer size must be multiple of 4
    next = std::copy(&padding, &padding + 1, next);
    assert(next == this->end());
    assert(this->size() == 969 + 1);
  }
};

struct HrtxPipeline_T
{
public:
  using BirdTableBinding = SingleBinding<BINDING_COMPRESS_BIRD_TABLE, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER>;
  using HeightmapBinding = SingleBinding<BINDING_COMPRESS_HEIGHTMAP, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER>;

  HrtxPipeline_T(VkCommandBuffer        initCommands,
                 VkPhysicalDevice       physicalDevice,
                 VkDevice               device,
                 HrtxAllocatorCallbacks allocator,
                 PFN_hrtxCheckVkResult  checkResultCallback,
                 VkPipelineCache        pipelineCache)
      : m_ctx(physicalDevice, device, allocator, checkResultCallback)
      , m_shaderCompress(m_ctx, compress_comp, sizeof(compress_comp))
      , m_birdTableBinding(m_ctx)
      , m_birdTable(m_ctx,
                    static_cast<VkDeviceSize>(m_blockToBirdUVTable.size() * sizeof(m_blockToBirdUVTable[0])),
                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT)
      , m_birdTableDescriptors(m_ctx, m_birdTableBinding, m_birdTable.descriptor())
      , m_heightmapBinding(m_ctx)
      , m_pipelineLayout(m_ctx,
                         {m_birdTableBinding.layout(), m_heightmapBinding.layout()},
                         {VkPushConstantRange{VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                              static_cast<uint32_t>(sizeof(shaders::CompressPushConstants))}})
      , m_pipeline(m_ctx, m_pipelineLayout, m_shaderCompress, nullptr, pipelineCache)
  {
    m_birdTable.update(initCommands, m_blockToBirdUVTable.data());
  }
  HrtxPipeline_T(VkCommandBuffer           initCommands,
                 VkInstance                instance,
                 PFN_vkGetInstanceProcAddr getInstanceProcAddr,
                 VkPhysicalDevice          physicalDevice,
                 VkDevice                  device,
                 PFN_vkGetDeviceProcAddr   getDeviceProcAddr,
                 HrtxAllocatorCallbacks    allocator,
                 PFN_hrtxCheckVkResult     checkResultCallback,
                 VkPipelineCache           pipelineCache)
      : m_ctx(instance, getInstanceProcAddr, physicalDevice, device, getDeviceProcAddr, allocator, checkResultCallback)
      , m_shaderCompress(m_ctx, compress_comp, sizeof(compress_comp))
      , m_birdTableBinding(m_ctx)
      , m_birdTable(m_ctx,
                    static_cast<VkDeviceSize>(m_blockToBirdUVTable.size() * sizeof(m_blockToBirdUVTable[0])),
                    VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT)
      , m_birdTableDescriptors(m_ctx, m_birdTableBinding, m_birdTable.descriptor())
      , m_heightmapBinding(m_ctx)
      , m_pipelineLayout(m_ctx,
                         {m_birdTableBinding.layout(), m_heightmapBinding.layout()},
                         {VkPushConstantRange{VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                              static_cast<uint32_t>(sizeof(shaders::CompressPushConstants))}})
      , m_pipeline(m_ctx, m_pipelineLayout, m_shaderCompress, nullptr, pipelineCache)
  {
    m_birdTable.update(initCommands, m_blockToBirdUVTable.data());
  }
  HrtxPipeline_T(const HrtxPipeline_T& other)                                 = delete;
  HrtxPipeline_T&                      operator=(const HrtxPipeline_T& other) = delete;
  std::unique_ptr<SingleDescriptorSet> createHeightmapDescriptors(VkDescriptorImageInfo heightmapDescriptorInfo) const
  {
    return std::make_unique<SingleDescriptorSet>(m_ctx, m_heightmapBinding, heightmapDescriptorInfo);
  }
  void bindAndDispatch(VkCommandBuffer                      cmd,
                       const SingleDescriptorSet&           heightmapDescriptors,
                       const shaders::CompressPushConstants pushConstants,
                       int32_t                              groupCountX) const
  {
    std::vector<VkDescriptorSet> descriptorSets = {m_birdTableDescriptors, heightmapDescriptors};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0,
                            static_cast<uint32_t>(descriptorSets.size()), descriptorSets.data(), 0, nullptr);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
    vkCmdDispatch(cmd, groupCountX, 1, 1);
  }
  const HrtxContext& ctx() const { return m_ctx; }

private:
  BlockToBirdUVTable  m_blockToBirdUVTable;
  HrtxContext         m_ctx;
  ShaderModule        m_shaderCompress;
  BirdTableBinding    m_birdTableBinding;
  Buffer              m_birdTable;
  SingleDescriptorSet m_birdTableDescriptors;
  HeightmapBinding    m_heightmapBinding;
  PipelineLayout      m_pipelineLayout;
  ComputePipeline     m_pipeline;
};
