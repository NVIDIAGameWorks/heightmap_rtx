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
#include <heightmap_rtx.h>
#include <vector>
#include <vulkan_functions.h>

struct VulkanTable
{
  // Initialize the table from static functions
  VulkanTable()
      :
#define COMMA ,
#define VULKAN_FUNCTION(name) name(::name)
      FOREACH_VULKAN_INSTANCE_FUNCTION(COMMA)
      , FOREACH_VULKAN_DEVICE_FUNCTION(COMMA)
#undef VULKAN_FUNCTION
  {
  }

  // Initialize the table dynamically by loading function pointers
  VulkanTable(VkInstance instance, PFN_vkGetInstanceProcAddr getInstanceProcAddr, VkDevice device, PFN_vkGetDeviceProcAddr getDeviceProcAddr)
      :
#define VULKAN_FUNCTION(name) name(reinterpret_cast<PFN_##name>(getInstanceProcAddr(instance, #name)))
      FOREACH_VULKAN_INSTANCE_FUNCTION(COMMA)
      ,
#undef VULKAN_FUNCTION
#define VULKAN_FUNCTION(name) name(reinterpret_cast<PFN_##name>(getDeviceProcAddr(device, #name)))
      FOREACH_VULKAN_DEVICE_FUNCTION(COMMA)
#undef VULKAN_FUNCTION
  {
  }

#define EMPTY
#define VULKAN_FUNCTION(name) const PFN_##name name;
  FOREACH_VULKAN_INSTANCE_FUNCTION(EMPTY)
  FOREACH_VULKAN_DEVICE_FUNCTION(EMPTY)
#undef VULKAN_FUNCTION
};

struct HrtxContext
{
  HrtxContext(VkPhysicalDevice physicalDevice, VkDevice device, HrtxAllocatorCallbacks allocator, PFN_hrtxCheckVkResult checkResultCallback)
      : physicalDevice(physicalDevice)
      , device(device)
      , allocator(allocator)
      , vk()
      , checkResultCallback(checkResultCallback)
  {
  }
  HrtxContext(VkInstance                instance,
              PFN_vkGetInstanceProcAddr getInstanceProcAddr,
              VkPhysicalDevice          physicalDevice,
              VkDevice                  device,
              PFN_vkGetDeviceProcAddr   getDeviceProcAddr,
              HrtxAllocatorCallbacks    allocator,
              PFN_hrtxCheckVkResult     checkResultCallback)
      : physicalDevice(physicalDevice)
      , device(device)
      , allocator(allocator)
      , vk(instance, getInstanceProcAddr, device, getDeviceProcAddr)
      , checkResultCallback(checkResultCallback)
  {
  }
  void checkResult(VkResult result) const
  {
    if(checkResultCallback)
    {
      checkResultCallback(result);
    }
  }

  VkPhysicalDevice       physicalDevice;
  VkDevice               device;
  HrtxAllocatorCallbacks allocator;
  VulkanTable            vk;
  PFN_hrtxCheckVkResult  checkResultCallback;
};
