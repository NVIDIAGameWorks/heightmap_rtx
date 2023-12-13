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

#include <cstddef>
#include <vulkan/vulkan_core.h>
#include <heightmap_rtx.h>
#include <context.hpp>
#include <cassert>

class Buffer
{
public:
  Buffer(const HrtxContext& ctx, VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
      : m_ctx(ctx)
      , m_size(size)
      , m_buffer(m_ctx.allocator.createBuffer(
            VkBufferCreateInfo{
                VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                nullptr,
                0,
                size,
                usage,
                VK_SHARING_MODE_EXCLUSIVE,
                0,
                nullptr,
            },
            props,
            m_ctx.allocator.userPtr))
  {
    assert(m_size % 4 == 0 && "vkCmdUpdateBuffer() size must be a multiple of 4");
  }
  Buffer(const Buffer& other)            = delete;
  Buffer& operator=(const Buffer& other) = delete;
  ~Buffer() noexcept { m_ctx.allocator.destroyBuffer(m_buffer, m_ctx.allocator.userPtr); }
  operator const VkBuffer&() const { return *m_buffer; }

  VkDeviceAddress address() const
  {
    VkBufferDeviceAddressInfo info = {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, nullptr, *this};
    return m_ctx.vk.vkGetBufferDeviceAddress(m_ctx.device, &info);
  }
  VkDeviceSize           size() const { return m_size; }
  VkDescriptorBufferInfo descriptor() const { return {*this, 0, m_size}; }
  void update(VkCommandBuffer cmd, const void* data) const { m_ctx.vk.vkCmdUpdateBuffer(cmd, *this, 0, m_size, data); }
  void copy(VkCommandBuffer cmd, const Buffer& other) const
  {
    assert(size() == other.size());
    VkBufferCopy copyRange{0, 0, size()};
    m_ctx.vk.vkCmdCopyBuffer(cmd, *this, other, 1, &copyRange);
  }
  void clear(VkCommandBuffer cmd, uint32_t value = 0) const { m_ctx.vk.vkCmdFillBuffer(cmd, *this, 0, m_size, value); }

private:
  const HrtxContext& m_ctx;
  VkDeviceSize       m_size;
  VkBuffer*          m_buffer;
};

class ShaderModule
{
public:
  ShaderModule(const ShaderModule& other)            = delete;
  ShaderModule& operator=(const ShaderModule& other) = delete;
  ShaderModule(const HrtxContext& ctx, const uint32_t* code, size_t codeSize)
      : m_ctx(ctx)
  {
    VkShaderModuleCreateInfo moduleCreateInfo{
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        nullptr,
        0,
        codeSize,
        code,  //
    };
    m_ctx.checkResult(m_ctx.vk.vkCreateShaderModule(m_ctx.device, &moduleCreateInfo, m_ctx.allocator.systemAllocator, &m_module));
  }
  ~ShaderModule() { m_ctx.vk.vkDestroyShaderModule(m_ctx.device, m_module, m_ctx.allocator.systemAllocator); }
  operator const VkShaderModule&() const { return m_module; }

private:
  const HrtxContext& m_ctx;
  VkShaderModule     m_module;
};

class PipelineLayout
{
public:
  PipelineLayout(const PipelineLayout& other)            = delete;
  PipelineLayout& operator=(const PipelineLayout& other) = delete;
  PipelineLayout(const HrtxContext&                        ctx,
                 const std::vector<VkDescriptorSetLayout>& descriptorSetLayouts,
                 const std::vector<VkPushConstantRange>&   pushConstantRanges,
                 VkPipelineLayoutCreateFlags               flags = 0)
      : m_ctx(ctx)
  {
    VkPipelineLayoutCreateInfo pipelineLayoutCreate{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        nullptr,
        flags,
        static_cast<uint32_t>(descriptorSetLayouts.size()),
        descriptorSetLayouts.data(),
        static_cast<uint32_t>(pushConstantRanges.size()),
        pushConstantRanges.data(),
    };
    m_ctx.checkResult(m_ctx.vk.vkCreatePipelineLayout(m_ctx.device, &pipelineLayoutCreate,
                                                      m_ctx.allocator.systemAllocator, &m_pipelineLayout));
  }
  ~PipelineLayout() noexcept
  {
    m_ctx.vk.vkDestroyPipelineLayout(m_ctx.device, m_pipelineLayout, m_ctx.allocator.systemAllocator);
  }
  operator const VkPipelineLayout&() const { return m_pipelineLayout; }

private:
  const HrtxContext& m_ctx;
  VkPipelineLayout   m_pipelineLayout;
};

class ComputePipeline
{
public:
  ComputePipeline(const ComputePipeline& other)            = delete;
  ComputePipeline& operator=(const ComputePipeline& other) = delete;
  ComputePipeline(const HrtxContext&          ctx,
                  VkPipelineLayout            pipelineLayout,
                  VkShaderModule              shaderModule,
                  const VkSpecializationInfo* specialization = nullptr,
                  VkPipelineCache             pipelineCache  = VK_NULL_HANDLE)
      : ComputePipeline(ctx,
                        VkComputePipelineCreateInfo{
                            VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                            nullptr,
                            0,
                            VkPipelineShaderStageCreateInfo{
                                VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                nullptr,
                                0,
                                VK_SHADER_STAGE_COMPUTE_BIT,
                                shaderModule,
                                "main",
                                specialization,
                            },
                            pipelineLayout,
                            VK_NULL_HANDLE,
                            0,
                        },
                        pipelineCache)
  {
  }
  ComputePipeline(const HrtxContext& ctx, const VkComputePipelineCreateInfo& pipelineCreate, VkPipelineCache pipelineCache = VK_NULL_HANDLE)
      : m_ctx(ctx)
  {
    m_ctx.checkResult(m_ctx.vk.vkCreateComputePipelines(m_ctx.device, pipelineCache, 1, &pipelineCreate,
                                                        m_ctx.allocator.systemAllocator, &m_pipeline));
  }
  ~ComputePipeline() noexcept { m_ctx.vk.vkDestroyPipeline(m_ctx.device, m_pipeline, m_ctx.allocator.systemAllocator); }
  operator const VkPipeline&() const { return m_pipeline; }

private:
  const HrtxContext& m_ctx;
  VkPipeline         m_pipeline;
};

inline void memoryBarrier(VkCommandBuffer&     cmd,
                          const HrtxContext&   ctx,
                          VkPipelineStageFlags srcStageMask,
                          VkAccessFlags        srcAccessMask,
                          VkPipelineStageFlags dstStageMask,
                          VkAccessFlags        dstAccessMask,
                          VkDependencyFlags    dependencyFlags = 0)
{
  VkMemoryBarrier barrier{
      VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      nullptr,
      srcAccessMask,
      dstAccessMask,
  };
  ctx.vk.vkCmdPipelineBarrier(cmd, srcStageMask, dstStageMask, dependencyFlags, 1, &barrier, 0, nullptr, 0, nullptr);
}

inline void memoryBarrier2(VkCommandBuffer&      cmd,
                           const HrtxContext&    ctx,
                           VkPipelineStageFlags2 srcStageMask,
                           VkAccessFlags2        srcAccessMask,
                           VkPipelineStageFlags2 dstStageMask,
                           VkAccessFlags2        dstAccessMask,
                           VkDependencyFlags     dependencyFlags = 0)
{
  VkMemoryBarrier2 memoryBarrier = {
      VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      nullptr,
      srcStageMask,
      srcAccessMask,
      dstStageMask,
      dstAccessMask,  //
  };
  VkDependencyInfo depencencyInfo = {
      VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      nullptr,
      dependencyFlags,
      1,
      &memoryBarrier,
      0,
      nullptr,
      0,
      nullptr,  //
  };
  ctx.vk.vkCmdPipelineBarrier2(cmd, &depencencyInfo);
}
