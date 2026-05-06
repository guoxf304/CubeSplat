/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,            // 高斯投影到屏幕坐标 [P, 2]
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		// 获取了该高斯点影响的tile区域（左上和右下的tile坐标）
		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth.
		// 遍历所有的tile，构造 key-value
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,   // 几何预处理内存块
	std::function<char* (size_t)> binningBuffer,    // binning排序的临时数据，如tile-key
	std::function<char* (size_t)> imageBuffer,      // 每个像素渲染的中间变量，如颜色累积、alpha融合等
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	float* out_color,
	float* out_depth,
	float* out_opacity,
	int* radii,
	int* n_touched,
	bool debug)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	// chunk_size：P 存储P个高斯点几何数据的内存大小
	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	// GeometryState 几何预处理内存块，包含了（投影位置、协方差、深度等），以高斯点的个数 P 为长度
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	// 每个像素的中间量内存
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	// 像素渲染的中间变量，以每张图像的像素个数 W * H 为长度
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,                  // 高斯点的透明度
		shs,
		geomState.clamped,          // 是否需要裁切
		cov3D_precomp,              // 预计算的 3D 协方差
		colors_precomp,             // 预计算的颜色
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,                      // 每个高斯点在屏幕的投影半径
		geomState.means2D,          // 每个高斯点的 2D 投影坐标
		geomState.depths,           // 高斯点的深度
		geomState.cov3D,            // 高斯点的协方差矩阵
		geomState.rgb,              // 高斯点的颜色
		geomState.conic_opacity,    // 协方差矩阵的逆矩阵和透明度
		tile_grid,
		geomState.tiles_touched,    // 每个高斯点覆盖的tile
		prefiltered
	), debug)

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	// geomState.scanning_space：临时缓存区
	// geomState.scan_size：临时缓存区的大小
	// geomState.tiles_touched：输入数组，每个高斯点覆盖的tile
	// geomState.point_offsets：输出数组，每个高斯点在瓦片列表的偏移量

	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	// geomState.point_offsets + P - 1 表示指向最后一个数组的指针，拷贝到 num_rendered 中
	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	// binning_chunk_size：渲染总数需要的内存
	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	// 健对存到 point_list_keys_unsorted，每个3D高斯的index存到point_list_unsorted中
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid)
	CHECK_CUDA(, debug)

	// 瓦片数量最高有效位（MSB）的下一位
	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	// 对键对进行排序
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space, // 排序所需的临时空间大小
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)

	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// Identify start and end of per-tile workloads in sorted list
	// 找出瓦片在排序后在键值对列表的范围（起始和结束索引），存储在ranges中
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	CHECK_CUDA(FORWARD::render(
		tile_grid, block,
		imgState.ranges,            //每个瓦片中高斯点的起始和结束索引
		binningState.point_list,    // 排序后高斯点索引列表
		width, height,
		geomState.means2D,
		feature_ptr,                // 高斯点的颜色
		geomState.conic_opacity,
		imgState.accum_alpha,       // 每个像素积累的 alpha 值
		imgState.n_contrib,         // 每个像素最后一个高斯点的贡献信息
		background,
		out_color,                  // 最终渲染结果
		geomState.depths,
		out_depth, 
		out_opacity,
		n_touched
    ), debug)

	return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
	const int P, int D, int M, int R,
	const float* background,                // [3]
	const int width, int height,
	const float* means3D,                   // [P, 3]
	const float* shs,
	const float* colors_precomp,            // none
	const float* scales,                    // [P, 3]
	const float scale_modifier,
	const float* rotations,                 // [P, 4]
	const float* cov3D_precomp,             // none 预计算
	const float* viewmatrix,                // [4, 4]
	const float* projmatrix,                // [4, 4]
    const float* projmatrix_raw,            // [4, 4]
    const float* campos,                    // [3,]
	const float tan_fovx, float tan_fovy,
	const int* radii,                       // [P,]
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	const float* dL_dpix,                   // [3, W, H] L 对像素颜色的梯度
	const float* dL_dpix_depth,             // [1, W, H] L 对深度的梯度
	float* dL_dmean2D,                      // [P, 2] L 对高斯中心在屏幕位置的梯度
	float* dL_dconic,                       // [P, 2, 2]
	float* dL_dopacity,                     // [P, 1] L 对高斯透明度的梯度
	float* dL_dcolor,                       // [P, 3] L 对高斯颜色的梯度
	float* dL_ddepth,                       // [P, 1] L 对高斯点深度的梯度
	float* dL_dmean3D,                      // [P, 3] L 对高斯点坐标的梯度
	float* dL_dcov3D,                       // [P, 6] L 对3D协方差的梯度
	float* dL_dsh,                          // [P, M, 3]
	float* dL_dscale,                       // [P, 3]
	float* dL_drot,                         // [P, 4] L 对高斯点旋转的梯度
	float* dL_dtau,                         // [P, 6] L 对位姿的梯度
	bool debug)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	// 计算（1.高斯点在屏幕位置 dL_dmean2D 2.2D协方差梯度dL_dconic2D 3.不透明度dL_dopacity 4.高斯点颜色dL_dcolors 5.深度dL_ddepths） 的梯度
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
    const float* depth_ptr = geomState.depths;

	CHECK_CUDA(BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		color_ptr,
		depth_ptr,
		imgState.accum_alpha,
		imgState.n_contrib,
		dL_dpix,
		dL_dpix_depth,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor,
		dL_ddepth
    ), debug)

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
        projmatrix_raw,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,        // [P, 2] L 对高斯中心在屏幕位置的梯度，render已计算
		dL_dconic,                  // [P, 2, 2]，render已计算
		(glm::vec3*)dL_dmean3D,     // [P, 3] L 对高斯点坐标的梯度
		dL_dcolor,                  // [P, 3] L 对高斯颜色的梯度，render已计算
		dL_ddepth,                  // [P, 1] L 对高斯点深度的梯度，render已计算
		dL_dcov3D,                  // [P, 6] L 对3D协方差的梯度
		dL_dsh,                     // [P, M, 3]
		(glm::vec3*)dL_dscale,      // [P, 3]
		(glm::vec4*)dL_drot,        // [P, 4] L 对高斯点旋转的梯度
		dL_dtau), debug)            // [P, 6] L 对位姿的梯度
}