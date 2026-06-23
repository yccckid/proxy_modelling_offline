#pragma once
#include <torch/torch.h>

namespace spc_ops {
torch::Tensor quantize_points(const torch::Tensor &x, int level);

torch::Tensor quantized_points_to_fpoints(const torch::Tensor &qpts, int level);

torch::Tensor points_to_morton(torch::Tensor points);

torch::Tensor morton_to_points(torch::Tensor morton);

torch::Tensor dilate_points(torch::Tensor points, int level);

torch::Tensor unbatched_points_to_octree(torch::Tensor points, int level,
                                         bool sorted = false);

torch::Tensor points_to_corners(const torch::Tensor &points);

torch::Tensor points_to_neighbors(const torch::Tensor &points);

torch::Tensor points_to_125neighbors(const torch::Tensor &points);

torch::Tensor unbatched_get_level_points(const torch::Tensor &point_hierarchy,
                                         const torch::Tensor &pyramid,
                                         int level);
} // namespace spc_ops