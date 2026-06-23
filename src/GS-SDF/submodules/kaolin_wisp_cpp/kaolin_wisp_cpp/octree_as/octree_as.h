#pragma once
#include <torch/torch.h>

struct ASRaytraceResults {
  /*
  A data holder for keeping the results of a single acceleration structure
  raytrace() call. A raytrace operation returns all intersections of the ray set
  with the acceleration structure cells. Ray/cell intersections are also
  referred to in Kaolin & Wisp as "nuggets".
  */
  ASRaytraceResults() = default;
  ASRaytraceResults(const torch::Tensor &_ridx, const torch::Tensor &_pidx,
                    const torch::Tensor &_depth)
      : ridx(_ridx), pidx(_pidx), depth(_depth) {}

  torch::Tensor ridx;
  /*
  A tensor containing the ray index of the ray that intersected each cell
  [num_nuggets]. (can be used to index into rays.origins and rays.dirs)
  */

  torch::Tensor pidx;
  /*
  Point indices into the cells of the acceleration structure, where the ray
  intersected a cell [num_nuggets]
  */
  torch::Tensor depth;
  /*
  Depths of each nugget, representing:
    - The first intersection point of the ray with the cell (entry), and
    - Optionally also a second intersection point of the ray with the cell
  (exit). A tensor of [num_intersections, 1 or 2].
  */
};

struct ASRaymarchResults {
  /* """ A data holder for keeping the results of a single acceleration
  structure raymarching() call. A raymarch operation advances the ray set within
  the acceleration structure and generates samples along the rays.
  """ */
  ASRaymarchResults() = default;
  ASRaymarchResults(const torch::Tensor &_ridx, const torch::Tensor &_samples,
                    const torch::Tensor &_depth_samples = torch::Tensor(),
                    const torch::Tensor &_deltas = torch::Tensor(),
                    const torch::Tensor &_boundary = torch::Tensor(),
                    const torch::Tensor &_pack_info = torch::Tensor())
      : ridx(_ridx), samples(_samples), depth_samples(_depth_samples),
        deltas(_deltas), boundary(_boundary), pack_info(_pack_info) {}

  torch::Tensor ridx;
  /* """ A tensor containing the ray index of the ray that generated each sample
  [num_hit_samples]. Can be used to index into rays.origins and rays.dirs.
  """ */

  torch::Tensor samples;
  /* """ Sample coordinates of shape [num_hit_samples, 3]""" */

  torch::Tensor depth_samples;
  /* """ Depths (Distances) of each sample, a tensor of shape [num_hit_samples,
   * 1] """ */

  torch::Tensor deltas;
  /* """ Depth diffs between each sample and the previous one, a tensor of shape
   * [num_hit_samples, 1] """ */

  torch::Tensor boundary;
  /* """ Boundary tensor which marks the beginning of each variable-sized sample
  pack of shape [num_hit_samples]. That is: [True, False, False, False, True,
  False False] represents two rays of 4 and 3 samples respectively.
  """ */

  torch::Tensor pack_info;
  /* """ Encodes the position of boundaries as an int tensor. Equivalent to
  `boundary.nonzero()`.
  """ */
};

struct ASQueryResults {
  /* """ A data holder for keeping the results of a single acceleration
  structure query() call. A query receives a set of input coordinates and
  returns the cell indices of the acceleration structure where the query
  coordinates fall.
  """ */
  ASQueryResults() = default;
  ASQueryResults(const torch::Tensor &_pidx) : pidx(_pidx) {}

  torch::Tensor pidx;
  /* """ Holds the query results.
  - If the query is invoked with `with_parents=False`, this field is a tensor of
  shape [num_coords], containing indices of cells of the acceleration structure,
  where the query coordinates match.
  - If the query is invoked with `with_parents=True`, this field is a tensor of
  shape [num_coords, level+1], containing indices of the cells of the
  acceleration structure + the full parent hierarchy of each cell query result.
  """ */
};

class OctreeAS {
public:
  OctreeAS(const torch::Tensor &_octree,
           const torch::Tensor &_attributes = torch::Tensor());

  torch::Tensor get_quantized_points();

  ASQueryResults query(const torch::Tensor &coords, int level = -1,
                       bool with_parents = false);

  ASRaytraceResults raytrace(const torch::Tensor &origins,
                             const torch::Tensor &dirs, int level = -1,
                             bool with_exit = false);

  ASRaymarchResults _raymarch_voxel(const torch::Tensor &origins,
                                    const torch::Tensor &dirs, int num_samples,
                                    int level = -1);

  //   ASRaymarchResults _raymarch_ray(const Rays &rays, int num_samples,
  //                                   int level = -1) {
  //     if (level == -1) {
  //       level = max_level;
  //     }

  //     torch::Tensor depth =
  //         torch::linspace(0, 1.0, num_samples, rays.origins.device())
  //             .unsqueeze(0)
  //             .add(torch::rand({rays.origins.size(0), num_samples},
  //                              rays.origins.device())
  //                      .div(num_samples));

  //     depth.mul_(rays.dist_max - rays.dist_min).add_(rays.dist_min);

  //     int num_rays = rays.origins.size(0);
  //     torch::Tensor samples = torch::addcmul(
  //         rays.origins.unsqueeze(1), rays.dirs.unsqueeze(1),
  //         depth.unsqueeze(-1));
  //     ASQueryResults query_results =
  //         query(samples.reshape({num_rays * num_samples, 3}), level);
  //     torch::Tensor pidx = query_results.pidx;
  //     pidx = pidx.reshape({num_rays, num_samples});
  //     torch::Tensor mask = pidx.gt(-1);
  //     torch::Tensor non_masked_idx = torch::nonzero(mask);

  //     torch::Tensor deltas =
  //         torch::diff(depth, 1, -1)
  //             .prepend(torch::zeros({rays.origins.size(0), 1},
  //             depth.device())
  //                          .add_(rays.dist_min));

  //     torch::Tensor depth_samples, deltas_resized, samples_resized, ridx;
  //     fast_filter_method(non_masked_idx, depth, deltas, samples,
  //     num_samples,
  //                        num_rays, pidx.device(), depth_samples,
  //                        deltas_resized, samples_resized, ridx);

  //     torch::Tensor boundary = mark_pack_boundaries(ridx);

  //     return ASRaymarchResults(
  //         ridx.to(torch::kLong), samples_resized.to(torch::kFloat),
  //         depth_samples.to(torch::kFloat),
  //         deltas_resized.to(torch::kFloat),
  //         boundary, nullptr);
  //   }
  //   ASRaymarchResults _raymarch_uniform(const Rays &rays, int
  //   num_samples,
  //                                       int level = -1) {
  //     if (level == -1) {
  //       level = max_level;
  //     }

  //     ASRaytraceResults raytrace_results = raytrace(rays, level, true);

  //     float step_size = 2 * sqrt(3) / num_samples;
  //     int scale = ceil(1.0 / step_size);
  //     step_size = 1.0 / scale;

  //     torch::Tensor ia = torch::ceil(scale *
  //     raytrace_results.depth.select(-1, 0))
  //                            .to(torch::kInt);
  //     torch::Tensor ib = torch::ceil(scale *
  //     raytrace_results.depth.select(-1, 1))
  //                            .to(torch::kInt);
  //     torch::Tensor interval_cnt = ib - ia;

  //     torch::Tensor non_zero_elements = interval_cnt.ne(0);

  //     torch::Tensor filtered_ridx =
  //         raytrace_results.ridx.masked_select(non_zero_elements);
  //     torch::Tensor filtered_depth =
  //         raytrace_results.depth.masked_select(non_zero_elements);
  //     torch::Tensor filtered_interval_cnt =
  //         interval_cnt.masked_select(non_zero_elements);

  //     torch::Tensor insum = inclusive_sum_cuda(filtered_interval_cnt);

  //     torch::Tensor results = uniform_sample_cuda(
  //         scale, filtered_ridx.contiguous(), filtered_depth, insum);

  //     torch::Tensor ridx = results[0];
  //     torch::Tensor depth_samples = results[1];
  //     torch::Tensor boundary = results[2];
  //     torch::Tensor deltas =
  //         step_size *
  //         torch::ones({ridx.size(0), 1}, torch::kFloat32,
  //         depth_samples.device());
  //     torch::Tensor samples =
  //         torch::addcmul(rays.origins.index_select(0, ridx),
  //                        rays.dirs.index_select(0, ridx), depth_samples);

  //     return ASRaymarchResults(ridx, samples, depth_samples, deltas,
  //     boundary);
  //   }
  ASRaymarchResults raymarch(const torch::Tensor &origins,
                             const torch::Tensor &dirs,
                             const std::string &raymarch_type, int num_samples,
                             int level = -1);

  ASRaymarchResults raymarch(const torch::Tensor &origins,
                             const torch::Tensor &dirs,
                             const torch::Tensor &depths,
                             const std::string &raymarch_type, int num_samples,
                             int level = -1);
  //   std::vector<int> occupancy();

  //   std::vector<int> capacity();

  std::string name();

private:
  torch::Tensor octree_;
  torch::Tensor points_;
  torch::Tensor pyramid_;
  torch::Tensor prefix_;

  int max_level_;
  // std::unordered_map<std::string, std::string> extent;
};

// static OctreeAS from_mesh(const std::string &mesh_path, int level,
//                           bool sample_tex = false,
//                           int num_samples_on_mesh = 100000000) {
//   torch::Tensor vertices, faces, texture_vertices, texture_faces,
//   materials;

//   if (sample_tex) {
//     std::tie(vertices, faces, texture_vertices, texture_faces, materials) =
//         mesh_ops::load_obj(mesh_path, true);
//   } else {
//     std::tie(vertices, faces) = mesh_ops::load_obj(mesh_path);
//   }

//   std::tie(vertices, faces) = mesh_ops::normalize(vertices, faces,
//   "sphere");

//   torch::Tensor octree =
//       wisp_spc_ops::mesh_to_octree(vertices, faces, level,
//       num_samples_on_mesh);
//   OctreeAS accel_struct(octree);
//   accel_struct.extent["vertices"] = vertices;
//   accel_struct.extent["faces"] = faces;

//   if (sample_tex) {
//     accel_struct.extent["texv"] = texture_vertices;
//     accel_struct.extent["texf"] = texture_faces;
//     accel_struct.extent["mats"] = materials;
//   }

//   return accel_struct;
// }

OctreeAS *from_quantized_points(torch::Tensor quantized_points, int level);

// static OctreeAS make_dense(int level) {
//   torch::Tensor octree = wisp_spc_ops::create_dense_octree(level);
//   return OctreeAS(octree);
// }

OctreeAS *from_pointcloud(const torch::Tensor &pointcloud, const int &level);