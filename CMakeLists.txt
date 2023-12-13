# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set(HEIGHTMAP_RTX_LIB "heightmap_rtx")

# Find VulkanSDK
find_package(Vulkan REQUIRED)
if(NOT Vulkan_FOUND)
    message(FATAL_ERROR "VulkanSDK not found")
endif()
set(GLSLC_COMPILER Vulkan::glslangValidator)
message(STATUS "Found Vulkan package. Using: ${GLSLC_COMPILER}")

# Headers to appear in visual studio projects
file(GLOB VS_PROJECT_HEADERS include/*.h src/*.h src/*.hpp)

set(SOURCE_FILES
  src/heightmap_rtx.cpp
)

add_library(${HEIGHTMAP_RTX_LIB} ${SOURCE_FILES} ${VS_PROJECT_HEADERS})

if(MSVC)
	target_compile_options(${HEIGHTMAP_RTX_LIB} PRIVATE /W4 /WX)
else()
	target_compile_options(${HEIGHTMAP_RTX_LIB} PRIVATE -Wall -Wextra -Werror -Wpedantic)
endif()

option(VK_ENABLE_BETA_EXTENSIONS "Enable beta extensions provided by the Vulkan SDK" OFF)
add_definitions(-DVK_ENABLE_BETA_EXTENSIONS)

# Internal include directories
target_include_directories(${HEIGHTMAP_RTX_LIB} PRIVATE
  src
  shaders
)

# Export include directories to use this library
target_include_directories(${HEIGHTMAP_RTX_LIB} PUBLIC include ${Vulkan_INCLUDE_DIRS})

# Compile shader
set(SPIRV_OUTPUT_DIR "${CMAKE_BINARY_DIR}/spirv_headers")
set(GLSL_COMPRESS_COMP "${CMAKE_CURRENT_LIST_DIR}/shaders/compress.comp")
set(SPIRVH_COMPRESS_COMP "${SPIRV_OUTPUT_DIR}/compress.comp.h")
set(GLSL_COMPRESS_COMP_DEPS
    shaders/sample_default.h
    shaders/shader_definitions.h
)
file(MAKE_DIRECTORY ${SPIRV_OUTPUT_DIR})
add_custom_command(
  OUTPUT ${SPIRVH_COMPRESS_COMP}
  COMMAND ${GLSLC_COMPILER}
    --target-env vulkan1.3
    -Ishaders
    ${GLSL_COMPRESS_COMP}
    --vn compress_comp
    -o ${SPIRVH_COMPRESS_COMP}
    $<$<CONFIG:Debug>:-g>
  DEPENDS
    ${GLSL_COMPRESS_COMP}
    ${GLSL_COMPRESS_COMP_DEPS}
  COMMENT "Compiling GLSL ${SPIRVH_COMPRESS_COMP}"
)
target_include_directories(${HEIGHTMAP_RTX_LIB} PRIVATE
  ${SPIRV_OUTPUT_DIR}
)
add_custom_target(${HEIGHTMAP_RTX_LIB}_shaders DEPENDS
  ${SPIRVH_COMPRESS_COMP}
)
add_dependencies(${HEIGHTMAP_RTX_LIB} ${HEIGHTMAP_RTX_LIB}_shaders)

source_group("Shaders" FILES ${GLSL_COMPRESS_COMP} ${GLSL_COMPRESS_COMP_DEPS})
set_target_properties(${HEIGHTMAP_RTX_LIB} PROPERTIES FOLDER "heightmap_rtx")
