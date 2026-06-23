#pragma once

#include <ATen/ATen.h>

namespace render_ops {

at::Tensor find_depth_interval_cuda(at::Tensor start_idxes_in,
                                    at::Tensor depth_io);

std::vector<at::Tensor> find_depth_bound_cuda(at::Tensor query,
                                              at::Tensor start_idxes_in,
                                              at::Tensor curr_idxes_in,
                                              at::Tensor depth);

at::Tensor find_depth_interval_cuda2(at::Tensor start_idxes_in,
                                     at::Tensor depth_io);

at::Tensor find_depth_bound_cuda2(at::Tensor undone_mask, at::Tensor query,
                                  at::Tensor start_idxes_in,
                                  at::Tensor curr_idxes_in,
                                  at::Tensor depth_io);

// reserve depth_io that between start_idxes_in and curr_idxes_in
// [start_idxes_in, curr_idxes_in)
std::vector<at::Tensor> clean_depth_bound_cuda(at::Tensor start_idxes_in,
                                               at::Tensor curr_idxes_in,
                                               at::Tensor depth_io_in);

// reserve depth_io that between start_idxes_in and curr_idxes_in
// [start_idxes_in, curr_idxes_in]
at::Tensor clean_depth_bound_cuda2(at::Tensor start_idxes_in,
                                   at::Tensor curr_idxes_in,
                                   at::Tensor depth_io_in);

at::Tensor sort_depth_samples_cuda(at::Tensor depth_samples,
                                   at::Tensor start_idxes,
                                   at::Tensor append_depth);

at::Tensor sort_ray_depth_cuda(const at::Tensor &start_idxes,
                               at::Tensor depths);

std::vector<at::Tensor> cal_depth_delta_cuda(const at::Tensor &start_idxes,
                                             at::Tensor ridx, at::Tensor depths,
                                             at::Tensor slope);

std::vector<at::Tensor> find_next_depth_bound_cuda(at::Tensor query,
                                                   at::Tensor start_idxes_in,
                                                   at::Tensor curr_idxes_in,
                                                   at::Tensor depth_io);
} // namespace render_ops
