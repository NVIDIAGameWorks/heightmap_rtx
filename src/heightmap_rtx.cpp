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

#include "vulkan/vulkan_core.h"
#include <heightmap_rtx.h>
#include <vulkan_objects.hpp>
#include <context.hpp>
#include <hrtx_pipeline.hpp>
#include <hrtx_map.hpp>

VkResult hrtxCreatePipeline(VkCommandBuffer cmd, const HrtxPipelineCreate* create, HrtxPipeline* hrtxPipeline)
{
  if(create->instance != VK_NULL_HANDLE || create->getInstanceProcAddr || create->getDeviceProcAddr)
  {
    *hrtxPipeline = new HrtxPipeline_T(cmd, create->instance, create->getInstanceProcAddr, create->physicalDevice,
                                       create->device, create->getDeviceProcAddr, create->allocator,
                                       create->checkResultCallback, create->pipelineCache);
  }
  else
  {
    *hrtxPipeline = new HrtxPipeline_T(cmd, create->physicalDevice, create->device, create->allocator,
                                       create->checkResultCallback, create->pipelineCache);
  }
  return VK_SUCCESS;
}

void hrtxDestroyPipeline(HrtxPipeline hrtxPipeline)
{
  delete hrtxPipeline;
}


void hrtxBarrierFlags(VkPipelineStageFlags2* textureCoordsDstStageMask,
                      VkAccessFlags2*        textureCoordsDstAccessMask,
                      VkPipelineStageFlags2* directionsDstStageMask,
                      VkAccessFlags2*        directionsDstAccessMask,
                      VkImageLayout*         heightmapLayout)
{
  if(textureCoordsDstStageMask)
    *textureCoordsDstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
  if(textureCoordsDstAccessMask)
    *textureCoordsDstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
  if(directionsDstStageMask)
    *directionsDstStageMask = VK_PIPELINE_STAGE_2_MICROMAP_BUILD_BIT_EXT;
  if(directionsDstAccessMask)
    *directionsDstAccessMask = VK_ACCESS_2_MICROMAP_READ_BIT_EXT;
  if(heightmapLayout)
    *heightmapLayout = VK_IMAGE_LAYOUT_GENERAL;
}

VkResult hrtxCmdCreateMap(VkCommandBuffer cmd, HrtxPipeline hrtxPipeline, const HrtxMapCreate* create, HrtxMap* hrtxMap)
{
  if(!hrtxPipeline)
  {
    return VK_ERROR_INITIALIZATION_FAILED;
  }

  // TODO: add support for other formats
  if(create->triangles->indexType != VK_INDEX_TYPE_UINT32 || create->textureCoordsFormat != VK_FORMAT_R32G32_SFLOAT
     || create->textureCoordsStride % (sizeof(float) * 2) != 0)
  {
    return VK_ERROR_FORMAT_NOT_SUPPORTED;
  }

  if(create->primitiveCount == 0)
  {
    return VK_INCOMPLETE;  // ??
  }

  // TODO: passing 'cmd' to the constructor to fill it as a side-effect is a bit of a smell
  *hrtxMap = new HrtxMap_T(cmd, *hrtxPipeline, *create);
  return VK_SUCCESS;
}

void hrtxDestroyMap(HrtxMap hrtxMap)
{
  delete hrtxMap;
}

VkAccelerationStructureTrianglesDisplacementMicromapNV hrtxMapDesc(HrtxMap hrtxMap)
{
  return hrtxMap->descriptor();
}
