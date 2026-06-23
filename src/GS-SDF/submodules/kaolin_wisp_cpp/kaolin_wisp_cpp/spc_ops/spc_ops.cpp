#include "spc_ops.h"
#include <kaolin/csrc/ops/spc/point_utils.h>
#include <kaolin/csrc/ops/spc/spc.h>

namespace spc_ops {
torch::Tensor quantize_points(const torch::Tensor &x, int level) {
  // r"""Quantizes the input points to the given level.
  // Maps the input points [-1,1-r] to the quantized points in the range [0,
  // 2^level - 1].
  int res = std::pow(2, level);
  torch::Tensor qpts =
      torch::floor(torch::clamp(res * (x + 1.0) / 2.0, 0, res - 1))
          .to(torch::kInt16);
  return qpts;
}

torch::Tensor quantized_points_to_fpoints(const torch::Tensor &qpts,
                                          int level) {
  // r"""Converts the quantized points to the floating point points.
  // Maps the quantized points in the range [0, 2^level - 1] to the floating
  // point points in the range [-1, 1-r].
  float r = 1.0 / ((float)(0x1 << level));
  torch::Tensor fpts = r * (2.0 * qpts.to(torch::kFloat32) + 1.0) - 1.0;
  return fpts;
}

torch::Tensor points_to_morton(torch::Tensor points) {
  std::vector<int64_t> shape(points.sizes().begin(), points.sizes().end() - 1);
  points = points.reshape({-1, 3});
  return kaolin::points_to_morton_cuda(points.contiguous()).reshape(shape);
}

torch::Tensor morton_to_points(torch::Tensor morton) {
  std::vector<int64_t> shape(morton.sizes().begin(), morton.sizes().end());
  shape.push_back(3);
  morton = morton.reshape({-1});
  return kaolin::morton_to_points_cuda(morton.contiguous()).reshape(shape);
}

torch::Tensor dilate_points(torch::Tensor points, int level) {
  auto tensor_option =
      torch::TensorOptions().dtype(torch::kInt16).device(points.device());
  torch::Tensor _x = torch::tensor({{1, 0, 0}}, tensor_option);
  torch::Tensor _y = torch::tensor({{0, 1, 0}}, tensor_option);
  torch::Tensor _z = torch::tensor({{0, 0, 1}}, tensor_option);
  points = torch::cat(
      {points + _x,           points - _x,           points + _y,
       points - _y,           points + _z,           points - _z,
       points + _x + _y,      points + _x - _y,      points + _x + _z,
       points + _x - _z,      points + _y + _x,      points + _y - _x,
       points + _y + _z,      points + _y - _z,      points + _z + _x,
       points + _z - _x,      points + _z + _y,      points + _z - _y,
       points + _x + _y + _z, points + _x + _y - _z, points + _x - _y + _z,
       points + _x - _y - _z, points - _x + _y + _z, points - _x + _y - _z,
       points - _x - _y + _z, points - _x - _y - _z},
      0);
  points = torch::clamp(points, 0, pow(2, level) - 1);

  auto unique =
      std::get<0>(torch::unique_dim(points.contiguous(), 0, true, true, true));

  auto morton = std::get<0>(
      torch::sort(points_to_morton(unique.contiguous()).contiguous()));

  points = morton_to_points(morton.contiguous());

  return points;
}

torch::Tensor unbatched_points_to_octree(torch::Tensor points, int level,
                                         bool sorted) {
  if (!sorted) {
    torch::Tensor unique =
        std::get<0>(torch::unique_dim(points.contiguous(), 0)).contiguous();
    torch::Tensor morton =
        std::get<0>(torch::sort(points_to_morton(unique).contiguous()));
    points = morton_to_points(morton.contiguous());
  }
  return kaolin::points_to_octree(points.contiguous(), level);
}

torch::Tensor points_to_corners(const torch::Tensor &points) {
  /* r"""Calculates the corners of the points assuming each point is the 0th bit
    corner.

    Args:
        points (torch.ShortTensor): Quantized 3D points,
                                    of shape :math:`(\text{num_points}, 3)`.

    Returns:
        (torch.ShortTensor):
            Quantized 3D new points,
            of shape :math:`(\text{num_points}, 8, 3)`.

    Examples:
        >>> inputs = torch.tensor([
        ...     [0, 0, 0],
        ...     [0, 2, 0]], device='cuda', dtype=torch.int16)
        >>> points_to_corners(inputs)
        tensor([[[0, 0, 0],
                 [0, 0, 1],
                 [0, 1, 0],
                 [0, 1, 1],
                 [1, 0, 0],
                 [1, 0, 1],
                 [1, 1, 0],
                 [1, 1, 1]],
        <BLANKLINE>
                [[0, 2, 0],
                 [0, 2, 1],
                 [0, 3, 0],
                 [0, 3, 1],
                 [1, 2, 0],
                 [1, 2, 1],
                 [1, 3, 0],
                 [1, 3, 1]]], device='cuda:0', dtype=torch.int16)
    """ */
  return kaolin::points_to_corners_cuda(points.to(torch::kShort).contiguous());
}

torch::Tensor points_to_neighbors(const torch::Tensor &points) {
  return kaolin::points_to_neighbors_cuda(
      points.to(torch::kShort).contiguous());
}

torch::Tensor points_to_125neighbors(const torch::Tensor &points) {
  return kaolin::points_to_125neighbors_cuda(
      points.to(torch::kShort).contiguous());
}

torch::Tensor unbatched_get_level_points(const torch::Tensor &point_hierarchy,
                                         const torch::Tensor &pyramid,
                                         int level) {
  /* r"""Returns the point set for the given level from the point hierarchy.

  Args:
      point_hierarchy (torch.ShortTensor):
          The point hierarchy of shape :math:`(\text{num_points}, 3)`.
          See :ref:`point_hierarchies <spc_points>` for a detailed description.

      pyramid (torch.IntTensor):
          The pyramid of shape :math:`(2, \text{max_level}+2)`
          See :ref:`pyramids <spc_pyramids>` for a detailed description.

      level (int): The level of the point hierarchy to retrieve.

  Returns:
      (torch.ShortTensor): The pointset of shape
  :math:`(\text{num_points_on_level}, 3)`.
  """ */
  return point_hierarchy.slice(0, pyramid[1][level].item<int64_t>(),
                               pyramid[1][level + 1].item<int64_t>());
}
} // namespace spc_ops