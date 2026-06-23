#include "octree_as.h"

#include <kaolin/csrc/ops/spc/query.h>
#include <kaolin/csrc/ops/spc/spc.h>
#include <kaolin/csrc/render/spc/raytrace.h>

#include "spc_ops/spc_ops.h"
#include "wisp_spc_ops/wisp_spc_ops.h"

OctreeAS *from_pointcloud(const torch::Tensor &pointcloud, const int &level) {
  /*
  Builds the acceleration structure and initializes occupancy of cells from a
  pointcloud. The cells occupancy will be determined by points occupying the
  octree cells.

  Args:
      pointcloud (torch.FloatTensor): 3D coordinates of shape [num_coords,
  3] in normalized space [-1, 1].
      level (int): Depth of the octree (number of occupancy levels), used by
  acceleration structure for fast raymarching. This is essentially the depth of
  the octree
   */
  auto octree_pair = wisp_spc_ops::pointcloud_to_octree(pointcloud, level);
  return new OctreeAS(octree_pair.first, octree_pair.second);
}

OctreeAS *from_quantized_points(torch::Tensor quantized_points, int level) {
  torch::Tensor octree =
      spc_ops::unbatched_points_to_octree(quantized_points, level, false);
  return new OctreeAS(octree);
}

OctreeAS::OctreeAS(const torch::Tensor &_octree,
                   const torch::Tensor &_attributes)
    : octree_(_octree) {

  auto octree_info = wisp_spc_ops::octree_to_spc(octree_);
  points_ = std::get<0>(octree_info);
  pyramid_ = std::get<1>(octree_info);
  prefix_ = std::get<2>(octree_info);

  max_level_ = pyramid_.size(-1) - 2;
}

torch::Tensor OctreeAS::get_quantized_points() {
  return spc_ops::unbatched_get_level_points(points_, pyramid_, max_level_);
}

ASQueryResults OctreeAS::query(const torch::Tensor &coords, int level,
                               bool with_parents) {
  /*
  """Returns the ``pidx`` for the sample coordinates (indices of acceleration
  structure cells returned by this query).

  Args:
      coords (torch.FloatTensor) : 3D coordinates of shape [num_coords, 3] in
  normalized [-1, 1] space. level (int) : The depth of the octree to query. If
  None, queries the highest level. with_parents (bool) : If True, also returns
  hierarchical parent indices.

  Returns:
      (ASQueryResults): containing the indices into the point hierarchy of shape
  [num_query]. If with_parents is True, then the query result will be of shape
  [num_query, level+1].
  """
   */
  if (level < 0) {
    level = this->max_level_;
  }

  torch::Tensor input_coords;
  if (coords.is_floating_point()) {
    input_coords = coords;
  } else {
    input_coords = (coords.to(torch::kFloat32) / pow(2, level)) * 2.0 - 1.0;
  }

  torch::Tensor pidx;
  if (with_parents) {
    kaolin::query_multiscale_cuda(octree_, prefix_, input_coords.contiguous(),
                                  level)
        .to(torch::kLong);
  } else {
    pidx =
        kaolin::query_cuda(octree_, prefix_, input_coords.contiguous(), level)
            .to(torch::kLong);
  }
  return {pidx};
}

ASRaytraceResults OctreeAS::raytrace(const torch::Tensor &origins,
                                     const torch::Tensor &dirs, int level,
                                     bool with_exit) {
  /* """Traces rays against the SPC structure, returning all intersections along
  the ray with the SPC points (SPC points are quantized, and can be interpreted
  as octree cell centers or corners).

  Args:
      rays (wisp.core.Rays): Ray origins and directions of shape [batch, 3].
      level (int) : The level of the octree to raytrace. If None, traces the
  highest level. with_exit (bool) : If True, also returns exit depth.

  Returns:
      (ASRaytraceResults): with fields containing -
          - Indices into rays.origins and rays.dirs of shape [num_intersections]
          - Indices into the point_hierarchy of shape [num_intersections]
          - Depths of shape [num_intersections, 1 or 2]
  """ */
  if (level == -1) {
    level = max_level_;
  }

  auto raytrace_results =
      kaolin::raytrace_cuda(octree_, points_, pyramid_, prefix_, origins, dirs,
                            level, true, with_exit);

  auto ridx = raytrace_results[0].select(-1, 0);
  auto pidx = raytrace_results[0].select(-1, 1);
  auto depth = raytrace_results[1];

  return {ridx, pidx, depth};
}

ASRaymarchResults OctreeAS::_raymarch_voxel(const torch::Tensor &origins,
                                            const torch::Tensor &dirs,
                                            int num_samples, int level) {
  /*
  """Samples points along the ray inside the SPC structure.
  Raymarch is achieved by intersecting the rays with the SPC cells.
  Then among the intersected cells, each cell is sampled num_samples times.
  In this scheme, num_hit_samples <= num_intersections*num_samples

  Args:
      rays (wisp.core.Rays): Ray origins and directions of shape [batch, 3].
      num_samples (int) : Number of samples generated per voxel. The total
  number of samples generated will also depend on the number of cells a ray have
  intersected. level (int) : The level of the octree to raytrace. If None,
  traces the highest level.

  Returns:
      (ASRaymarchResults) with fields containing:
          - Indices into rays.origins and rays.dirs of shape [num_hit_samples]
          - Sample coordinates of shape [num_hit_samples, 3]
          - Sample depths of shape [num_hit_samples, 1]
          - Sample depth diffs of shape [num_hit_samples, 1]
          - Boundary tensor which marks the beginning of each variable-sized
            sample pack of shape [num_hit_samples]
  """
   */
  if (level == -1) {
    level = max_level_;
  }

  // # NUM_INTERSECTIONS = number of nuggets: ray / cell intersections
  // # NUM_INTERSECTIONS can be 0!
  // # ridx, pidx ~ (NUM_INTERSECTIONS,)
  // # depth ~ (NUM_INTERSECTIONS, 2)
  auto raytrace_results = raytrace(origins, dirs, level, true);
  auto ridx = raytrace_results.ridx.to(torch::kLong);
  auto num_intersections = ridx.size(0);

  // # depth_samples ~ (NUM_INTERSECTIONS, NUM_SAMPLES, 1)
  auto depth = raytrace_results.depth;
  auto depth_samples =
      wisp_spc_ops::sample_from_depth_intervals(depth, num_samples)
          .unsqueeze(-1);
  auto deltas = torch::diff(depth_samples.select(-1, 0), 1, -1,
                            depth.select(-1, 0).unsqueeze(-1))
                    .reshape({num_intersections * num_samples, 1});

  // # samples ~ (NUM_INTERSECTIONS, NUM_SAMPLES, 1)
  auto samples = torch::addcmul(origins.index({ridx}).unsqueeze(1),
                                dirs.index({ridx}).unsqueeze(1), depth_samples);

  // # boundary ~ (NUM_INTERSECTIONS * NUM_SAMPLES,)
  // # (each intersected cell is sampled NUM_SAMPLES times)
  auto boundary = wisp_spc_ops::expand_pack_boundary(
                      kaolin::mark_pack_boundaries_cuda(ridx.to(torch::kInt))
                          .to(torch::kBool),
                      num_samples)
                      .to(torch::kBool);

  ridx = ridx.unsqueeze(1)
             .expand({num_intersections, num_samples})
             .reshape({num_intersections * num_samples});
  samples = samples.reshape({num_intersections * num_samples, 3});
  depth_samples = depth_samples.reshape({num_intersections * num_samples, 1});

  return {ridx, samples, depth_samples, deltas, boundary};
}

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
//             .prepend(torch::zeros({rays.origins.size(0), 1}, depth.device())
//                          .add_(rays.dist_min));

//     torch::Tensor depth_samples, deltas_resized, samples_resized, ridx;
//     fast_filter_method(non_masked_idx, depth, deltas, samples, num_samples,
//                        num_rays, pidx.device(), depth_samples,
//                        deltas_resized, samples_resized, ridx);

//     torch::Tensor boundary = mark_pack_boundaries(ridx);

//     return ASRaymarchResults(
//         ridx.to(torch::kLong), samples_resized.to(torch::kFloat),
//         depth_samples.to(torch::kFloat), deltas_resized.to(torch::kFloat),
//         boundary, nullptr);
//   }
//   ASRaymarchResults _raymarch_uniform(const Rays &rays, int num_samples,
//                                       int level = -1) {
//     if (level == -1) {
//       level = max_level;
//     }

//     ASRaytraceResults raytrace_results = raytrace(rays, level, true);

//     float step_size = 2 * sqrt(3) / num_samples;
//     int scale = ceil(1.0 / step_size);
//     step_size = 1.0 / scale;

//     torch::Tensor ia = torch::ceil(scale * raytrace_results.depth.select(-1,
//     0))
//                            .to(torch::kInt);
//     torch::Tensor ib = torch::ceil(scale * raytrace_results.depth.select(-1,
//     1))
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

//     return ASRaymarchResults(ridx, samples, depth_samples, deltas, boundary);
//   }
ASRaymarchResults OctreeAS::raymarch(const torch::Tensor &origins,
                                     const torch::Tensor &dirs,
                                     const std::string &raymarch_type,
                                     int num_samples, int level) {
  /*
  """Samples points along the ray inside the SPC structure.
  The exact algorithm employed for raymarching is determined by `raymarch_type`.

  Args:
      rays (wisp.core.Rays): Ray origins and directions of shape [batch, 3].
      raymarch_type (str): Sampling strategy to use for raymarch.
          'voxel' - intersects the rays with the SPC cells. Then among the
  intersected cells, each cell is sampled num_samples times. In this scheme,
  num_hit_samples <= num_intersections*num_samples 'ray' - samples num_samples
  along each ray, and then filters out samples which falls outside of occupied
              cells.
              In this scheme, num_hit_samples <= num_rays * num_samples
      num_samples (int) : Number of samples generated per voxel or ray. The
  exact meaning of this arg depends on the value of `raymarch_type`. level (int)
  : The level of the octree to raytrace. If None, traces the highest level.

  Returns:
      (ASRaymarchResults) with fields containing:
          - Indices into rays.origins and rays.dirs of shape [num_hit_samples]
          - Sample coordinates of shape [num_hit_samples, 3]
          - Sample depths of shape [num_hit_samples, 1]
          - Sample depth diffs of shape [num_hit_samples, 1]
          - Boundary tensor which marks the beginning of each variable-sized
            sample pack of shape [num_hit_samples]
  """
   */

  if (level == -1) {
    level = max_level_;
  }

  ASRaymarchResults raymarch_results;

  /* Samples points along the rays by first tracing it against the SPC object.
  Then, given each SPC voxel hit, will sample some number of samples in
  each voxel. This setting is pretty nice for getting decent outputs from
  outside-looking-in scenes, but in general it's not very robust or proper
  since the ray samples will be weirdly distributed # and or aliased. */
  if (raymarch_type == "voxel") {
    raymarch_results = _raymarch_voxel(origins, dirs, num_samples, level);
  }
  /* Samples points along the rays, and then uses the SPC object the filter
  out samples that don't hit the SPC objects. This is a much more
  well-spaced-out sampling scheme and will work well for inside-looking-out
  scenes. The camera near and far planes will have to be adjusted carefully,
  however.  */
  // else if (raymarch_type == "ray") {
  //   raymarch_results = _raymarch_ray(origins, dirs, num_samples, level);
  // }
  // else if (raymarch_type == "uniform") {
  //     raymarch_results = _raymarch_uniform(rays, num_samples, level);
  //   }
  else {
    throw std::invalid_argument(
        "Raymarch sampler type is not supported by OctreeAS.");
  }

  return raymarch_results;
}

ASRaymarchResults OctreeAS::raymarch(const torch::Tensor &origins,
                                     const torch::Tensor &dirs,
                                     const torch::Tensor &depths,
                                     const std::string &raymarch_type,
                                     int num_samples, int level) {
  /*
  """Samples points along the ray inside the SPC structure.
  The exact algorithm employed for raymarching is determined by `raymarch_type`.

  Args:
      rays (wisp.core.Rays): Ray origins and directions of shape [batch, 3].
      raymarch_type (str): Sampling strategy to use for raymarch.
          'voxel' - intersects the rays with the SPC cells. Then among the
  intersected cells, each cell is sampled num_samples times. In this scheme,
  num_hit_samples <= num_intersections*num_samples 'ray' - samples num_samples
  along each ray, and then filters out samples which falls outside of occupied
              cells.
              In this scheme, num_hit_samples <= num_rays * num_samples
      num_samples (int) : Number of samples generated per voxel or ray. The
  exact meaning of this arg depends on the value of `raymarch_type`. level (int)
  : The level of the octree to raytrace. If None, traces the highest level.

  Returns:
      (ASRaymarchResults) with fields containing:
          - Indices into rays.origins and rays.dirs of shape [num_hit_samples]
          - Sample coordinates of shape [num_hit_samples, 3]
          - Sample depths of shape [num_hit_samples, 1]
          - Sample depth diffs of shape [num_hit_samples, 1]
          - Boundary tensor which marks the beginning of each variable-sized
            sample pack of shape [num_hit_samples]
  """
   */

  if (level == -1) {
    level = max_level_;
  }

  // # NUM_INTERSECTIONS = number of nuggets: ray / cell intersections
  // # NUM_INTERSECTIONS can be 0!
  // # ridx, pidx ~ (NUM_INTERSECTIONS,)
  // # depth ~ (NUM_INTERSECTIONS, 2)
  auto raytrace_results = raytrace(origins, dirs, level, true);
  auto ridx = raytrace_results.ridx.to(torch::kLong);
  auto num_intersections = ridx.size(0);

  // # depth_samples ~ (NUM_INTERSECTIONS, NUM_SAMPLES, 1)
  auto depth = raytrace_results.depth;
  auto depth_samples =
      wisp_spc_ops::sample_from_depth_intervals(depth, num_samples)
          .unsqueeze(-1);
  auto deltas = torch::diff(depth_samples.select(-1, 0), 1, -1,
                            depth.select(-1, 0).unsqueeze(-1))
                    .reshape({num_intersections * num_samples, 1});

  // # samples ~ (NUM_INTERSECTIONS, NUM_SAMPLES, 1)
  auto samples = torch::addcmul(origins.index({ridx}).unsqueeze(1),
                                dirs.index({ridx}).unsqueeze(1), depth_samples);

  // # boundary ~ (NUM_INTERSECTIONS * NUM_SAMPLES,)
  // # (each intersected cell is sampled NUM_SAMPLES times)
  auto boundary = wisp_spc_ops::expand_pack_boundary(
                      kaolin::mark_pack_boundaries_cuda(ridx.to(torch::kInt))
                          .to(torch::kBool),
                      num_samples)
                      .to(torch::kBool);

  ridx = ridx.unsqueeze(1)
             .expand({num_intersections, num_samples})
             .reshape({num_intersections * num_samples});
  samples = samples.reshape({num_intersections * num_samples, 3});
  depth_samples = depth_samples.reshape({num_intersections * num_samples, 1});

  return {ridx, samples, depth_samples, deltas, boundary};
}

// std::vector<int> OctreeAS::occupancy() {
//   std::vector<int> occupancy_list;
//   for (int lod = 0; lod < LODs; lod++) {
//     occupancy_list.push_back(pyramid[0][lod]);
//   }
//   return occupancy_list;
// }

// std::vector<int> OctreeAS::capacity() {
//   std::vector<int> capacity_list;
//   for (int lod = 0; lod < LODs; lod++) {
//     capacity_list.push_back(pow(8, lod));
//   }
//   return capacity_list;
// }

std::string OctreeAS::name() { return "Octree"; }

// static OctreeAS from_mesh(const std::string &mesh_path, int level,
//                           bool sample_tex = false,
//                           int num_samples_on_mesh = 100000000) {
//   torch::Tensor vertices, faces, texture_vertices, texture_faces, materials;

//   if (sample_tex) {
//     std::tie(vertices, faces, texture_vertices, texture_faces, materials) =
//         mesh_ops::load_obj(mesh_path, true);
//   } else {
//     std::tie(vertices, faces) = mesh_ops::load_obj(mesh_path);
//   }

//   std::tie(vertices, faces) = mesh_ops::normalize(vertices, faces, "sphere");

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

// static OctreeAS make_dense(int level) {
//   torch::Tensor octree = wisp_spc_ops::create_dense_octree(level);
//   return OctreeAS(octree);
// }
