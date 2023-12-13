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

#include <vulkan/vulkan_core.h>
#include <heightmap_rtx.h>
#include <context.hpp>
#include <vulkan_objects.hpp>
#include <vector>
#include <unordered_map>
#include <cassert>

struct DescriptorBindingAndFlags
{
  VkDescriptorSetLayoutBinding binding;
  VkDescriptorBindingFlags     bindingFlags;
};

using DescriptorSetLayoutBindings = std::vector<DescriptorBindingAndFlags>;

class DescriptorSetLayout
{
public:
  DescriptorSetLayout(const HrtxContext&                          ctx,
                      const DescriptorSetLayoutBindings&          bindingsAndFlags,
                      VkDescriptorSetLayoutCreateInfo             layoutCreate         = {},
                      VkDescriptorSetLayoutBindingFlagsCreateInfo layoutBindingsCreate = {})
      : m_ctx(ctx)
  {
    // Temporary vectors to split the combined DescriptorBindingAndFlags structs
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    std::vector<VkDescriptorBindingFlags>     bindingsFlags;
    for(const DescriptorBindingAndFlags& obj : bindingsAndFlags)
    {
      bindings.push_back(obj.binding);
      bindingsFlags.push_back(obj.bindingFlags);
    }

    // Create the chain layoutCreate -> layoutBindingsCreate -> [old layoutCreate.pNext]
    assert(!layoutBindingsCreate.pNext);
    layoutBindingsCreate.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
    layoutBindingsCreate.pNext         = layoutCreate.pNext;
    layoutBindingsCreate.bindingCount  = static_cast<uint32_t>(bindingsFlags.size());
    layoutBindingsCreate.pBindingFlags = bindingsFlags.data();
    layoutCreate.sType                 = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutCreate.pNext                 = &layoutBindingsCreate;
    layoutCreate.bindingCount          = static_cast<uint32_t>(bindings.size());
    layoutCreate.pBindings             = bindings.data();

    m_ctx.checkResult(vkCreateDescriptorSetLayout(m_ctx.device, &layoutCreate, m_ctx.allocator.systemAllocator, &m_layout));
  }
  ~DescriptorSetLayout() { vkDestroyDescriptorSetLayout(m_ctx.device, m_layout, m_ctx.allocator.systemAllocator); }

  operator VkDescriptorSetLayout() const { return m_layout; }
  DescriptorSetLayout(const DescriptorSetLayout& other)            = delete;
  DescriptorSetLayout& operator=(const DescriptorSetLayout& other) = delete;
  const HrtxContext&   ctx() const { return m_ctx; }

private:
  const HrtxContext&    m_ctx;
  VkDescriptorSetLayout m_layout;
};

// A VkDescriptorPool with just enough space for the given bindings
class SingleDescriptorSetPool
{
public:
  SingleDescriptorSetPool(const HrtxContext& ctx, const DescriptorSetLayoutBindings& bindingsAndFlags, VkDescriptorPoolCreateFlags flags = 0)
      : m_ctx(ctx)
  {
    std::unordered_map<VkDescriptorType, uint32_t> typeSizes;
    for(auto& obj : bindingsAndFlags)
    {
      typeSizes[obj.binding.descriptorType]++;
    }
    std::vector<VkDescriptorPoolSize> poolSizes;
    for(auto& typeSize : typeSizes)
    {
      poolSizes.push_back({typeSize.first, typeSize.second});
    }

    VkDescriptorPoolCreateInfo descriptorPoolCreate = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        nullptr,
        flags,
        1,
        uint32_t(poolSizes.size()),
        poolSizes.data(),  //
    };
    m_ctx.checkResult(vkCreateDescriptorPool(m_ctx.device, &descriptorPoolCreate, m_ctx.allocator.systemAllocator, &m_pool));
  }
  ~SingleDescriptorSetPool() { vkDestroyDescriptorPool(m_ctx.device, m_pool, m_ctx.allocator.systemAllocator); }

  operator VkDescriptorPool() const { return m_pool; }
  SingleDescriptorSetPool(const SingleDescriptorSetPool& other)            = delete;
  SingleDescriptorSetPool& operator=(const SingleDescriptorSetPool& other) = delete;

private:
  const HrtxContext& m_ctx;
  VkDescriptorPool   m_pool;
};

class DescriptorSet
{
public:
  DescriptorSet(const HrtxContext& ctx, VkDescriptorPool pool, VkDescriptorSetLayout descriptorSetLayout)
      : m_ctx(ctx)
  {
    VkDescriptorSetAllocateInfo allocInfo = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        nullptr,
        pool,
        1,
        &descriptorSetLayout,  //
    };
    m_ctx.checkResult(vkAllocateDescriptorSets(m_ctx.device, &allocInfo, &m_set));
  }
  // TODO: if VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT support is
  // needed, a call to vkFreeDescriptorSets() should be in a destructor.

  operator VkDescriptorSet() const { return m_set; }

  DescriptorSet(const DescriptorSet& other)            = delete;
  DescriptorSet& operator=(const DescriptorSet& other) = delete;

private:
  const HrtxContext& m_ctx;
  VkDescriptorSet    m_set;
};

// Instantiates a descriptor set for a given layout, i.e. buffer bindings etc.
// for some shaders in a pipeline, and a pool just big enough for it.
class SingleDescriptorSet
{
public:
  template <class SingleBinding, class DescriptorInfo>
  SingleDescriptorSet(const HrtxContext& ctx, const SingleBinding& binding, const DescriptorInfo& descriptor)
      : SingleDescriptorSet(ctx, binding.bindings(), binding.layout())
  {
    binding.write(*this, descriptor);
  }
  SingleDescriptorSet(const HrtxContext& ctx, const DescriptorSetLayoutBindings& bindingsAndFlags, const DescriptorSetLayout& layout)
      : m_pool(ctx, bindingsAndFlags)
      , m_set(ctx, m_pool, layout)
  {
  }
  operator VkDescriptorSet() const { return m_set; }
  SingleDescriptorSet(const SingleDescriptorSet& other)            = delete;
  SingleDescriptorSet& operator=(const SingleDescriptorSet& other) = delete;

private:
  SingleDescriptorSetPool m_pool;
  DescriptorSet           m_set;
};

// clang-format off
template <class T> struct ValidDescriptorTypes;
template <> struct ValidDescriptorTypes<VkDescriptorImageInfo> { static constexpr bool validate(const VkDescriptorType& type) { return
  type == VK_DESCRIPTOR_TYPE_SAMPLER ||
  type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ||
  type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE ||
  type == VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT; } };
template <> struct ValidDescriptorTypes<VkDescriptorBufferInfo> { static constexpr bool validate(const VkDescriptorType& type) { return
  type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ||
  type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC ||
  type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ||
  type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC; } };
template <> struct ValidDescriptorTypes<VkBufferView> { static constexpr bool validate(const VkDescriptorType& type) { return
  type == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER ||
  type == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER; } };
template <> struct ValidDescriptorTypes<VkWriteDescriptorSetAccelerationStructureNV> { static constexpr bool validate(const VkDescriptorType& type) { return
  type == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_NV; } };
template <> struct ValidDescriptorTypes<VkWriteDescriptorSetAccelerationStructureKHR> { static constexpr bool validate(const VkDescriptorType& type) { return
  type == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR; } };
template <> struct ValidDescriptorTypes<VkWriteDescriptorSetInlineUniformBlockEXT> { static constexpr bool validate(const VkDescriptorType& type) { return
  type == VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK_EXT; } };
// clang-format on

template <class DescriptorInfo>
VkWriteDescriptorSet makeWriteDescriptorSet(const DescriptorBindingAndFlags& binding,
                                            VkDescriptorSet                  descriptorSet,
                                            const DescriptorInfo*            descriptorInfoPtr,
                                            uint32_t                         element = 0)
{
  VkWriteDescriptorSet result{
      VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      nullptr,
      descriptorSet,
      binding.binding.binding,
      element,
      binding.binding.descriptorCount,
      binding.binding.descriptorType,
      nullptr,
      nullptr,
      nullptr,
  };
  setWriteDescriptorSetPtr(result, descriptorInfoPtr);
  assert(result.descriptorCount == 1);  // If not one, should be using the vector overload
  return result;
}

template <class DescriptorInfo>
VkWriteDescriptorSet makeWriteDescriptorSet(const DescriptorBindingAndFlags&   binding,
                                            VkDescriptorSet                    descriptorSet,
                                            const std::vector<DescriptorInfo>& descriptorInfo)
{
  VkWriteDescriptorSet result{
      VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      descriptorSet,
      binding.binding.binding,
      static_cast<uint32_t>(descriptorInfo.size()),
      binding.binding.descriptorType,
  };
  setWriteDescriptorSetPtr(result, descriptorInfo.data());
  assert((binding.bindingFlags & VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT) ?
             result.descriptorCount <= binding.binding.descriptorCount :  // Not required to bind everything
             result.descriptorCount == binding.binding.descriptorCount);  // Must bind everything without the partial flag
  return result;
}

template <class DescriptorInfo>
void setWriteDescriptorSetPtr(VkWriteDescriptorSet& write, const DescriptorInfo* ptr)
{
  assert(ValidDescriptorTypes<DescriptorInfo>::validate(write.descriptorType));
  if constexpr(std::is_same_v<DescriptorInfo, VkDescriptorImageInfo>)
  {
    write.pImageInfo = ptr;
  }
  else if constexpr(std::is_same_v<DescriptorInfo, VkDescriptorBufferInfo>)
  {
    write.pBufferInfo = ptr;
  }
  else if constexpr(std::is_same_v<DescriptorInfo, VkBufferView>)
  {
    write.pTexelBufferView = ptr;
  }
  else
  {
    write.pNext = ptr;
  }
}

using DescriptorSetWrites = std::vector<VkWriteDescriptorSet>;

void updateDescriptorSets(VkDevice device, const DescriptorSetWrites& writes)
{
  vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

template <uint32_t BindingIndex, VkDescriptorType BindAs, VkShaderStageFlags Stages = VK_SHADER_STAGE_ALL>
class SingleBinding
{
public:
  SingleBinding(const HrtxContext& ctx)
      : m_bindings{DescriptorBindingAndFlags{
          {

              BindingIndex,
              BindAs,
              1,
              Stages,
              nullptr,
          },
          0,
      }}
      , m_layout(ctx, m_bindings)
  {
  }
  template <class DescriptorInfo>
  void write(VkDescriptorSet descriptorSet, const DescriptorInfo& descriptor) const
  {
    DescriptorSetWrites writes{makeWriteDescriptorSet(m_bindings[0], descriptorSet, &descriptor)};
    updateDescriptorSets(m_layout.ctx().device, writes);
  }
  const DescriptorSetLayoutBindings& bindings() const { return m_bindings; }
  const DescriptorSetLayout&         layout() const { return m_layout; }

private:
  DescriptorSetLayoutBindings m_bindings;
  DescriptorSetLayout         m_layout;
};
