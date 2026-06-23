#include <ATen/ATen.h>

namespace render_ops {

void find_depth_interval_cuda_impl(int64_t num_packs, int64_t num_nugs,
                                   at::Tensor start_idxes_in,
                                   at::Tensor depth_io,
                                   at::Tensor break_mask_out);

at::Tensor find_depth_interval_cuda(at::Tensor start_idxes_in,
                                    at::Tensor depth_io) {
#ifdef WITH_CUDA
  TORCH_CHECK(start_idxes_in.is_contiguous() && depth_io.is_contiguous() &&
              "Input tensors must be contiguous");
  int64_t num_packs = start_idxes_in.size(0);
  int64_t num_nugs = depth_io.size(0);
  auto break_mask_out =
      at::zeros({num_nugs}, depth_io.options().dtype(at::kBool));
  find_depth_interval_cuda_impl(num_packs, num_nugs, start_idxes_in, depth_io,
                                break_mask_out);
  return break_mask_out;
#else
  AT_ERROR(__func__);
#endif // WITH_CUDA
}

void find_depth_bound_cuda_impl(int64_t num_packs, int64_t num_nugs,
                                at::Tensor query, at::Tensor start_idxes_in,
                                at::Tensor curr_idxes_in, at::Tensor depth,
                                at::Tensor curr_idxes_out,
                                at::Tensor inbound_mask_out,
                                at::Tensor change_mask_out,
                                at::Tensor break_mask_out);

std::vector<at::Tensor> find_depth_bound_cuda(at::Tensor query,
                                              at::Tensor start_idxes_in,
                                              at::Tensor curr_idxes_in,
                                              at::Tensor depth) {
#ifdef WITH_CUDA
  TORCH_CHECK(query.is_contiguous() && start_idxes_in.is_contiguous() &&
              curr_idxes_in.is_contiguous() && depth.is_contiguous() &&
              "Input tensors must be contiguous");
  int64_t num_packs = query.size(0);
  int64_t num_nugs = depth.size(0);
  at::Tensor curr_idxes_out =
      at::zeros({num_packs}, query.options().dtype(at::kInt)) - 1;
  auto change_mask_out =
      at::zeros({num_packs}, query.options().dtype(at::kBool));
  auto inbound_mask_out = change_mask_out.clone();
  auto break_mask_out = change_mask_out.clone();
  find_depth_bound_cuda_impl(num_packs, num_nugs, query, start_idxes_in,
                             curr_idxes_in, depth, curr_idxes_out,
                             change_mask_out, inbound_mask_out, break_mask_out);
  return {curr_idxes_out, change_mask_out, inbound_mask_out, break_mask_out};
#else
  AT_ERROR(__func__);
#endif // WITH_CUDA
}

void find_depth_bound_cuda_impl2(int64_t num_packs, int64_t num_nugs,
                                 at::Tensor undone_mask, at::Tensor query,
                                 at::Tensor start_idxes_in,
                                 at::Tensor curr_idxes_in, at::Tensor depth_io,
                                 at::Tensor break_mask_out);

at::Tensor find_depth_bound_cuda2(at::Tensor undone_mask, at::Tensor query,
                                  at::Tensor start_idxes_in,
                                  at::Tensor curr_idxes_in,
                                  at::Tensor depth_io) {
  /* special function for adaptive shpere trace */
#ifdef WITH_CUDA
  TORCH_CHECK(undone_mask.is_contiguous() && query.is_contiguous() &&
              start_idxes_in.is_contiguous() && curr_idxes_in.is_contiguous() &&
              depth_io.is_contiguous());
  int64_t num_packs = query.size(0);
  int64_t num_nugs = depth_io.size(0);
  auto break_mask_out = at::zeros_like(query.view({-1}), at::kBool);
  find_depth_bound_cuda_impl2(num_packs, num_nugs, undone_mask, query,
                              start_idxes_in, curr_idxes_in, depth_io,
                              break_mask_out);
  return break_mask_out;

#else
  AT_ERROR(__func__);
#endif // WITH_CUDA
}

void find_next_depth_bound_cuda_impl(
    int64_t num_packs, int64_t num_nugs, at::Tensor query,
    at::Tensor start_idxes_in, at::Tensor curr_idxes_in, at::Tensor depth_io,
    at::Tensor depth_bound_start_out, at::Tensor depth_bound_end_out);

std::vector<at::Tensor> find_next_depth_bound_cuda(at::Tensor query,
                                                   at::Tensor start_idxes_in,
                                                   at::Tensor curr_idxes_in,
                                                   at::Tensor depth_io) {
  /* special function for adaptive shpere trace */
#ifdef WITH_CUDA
  TORCH_CHECK(query.is_contiguous() && start_idxes_in.is_contiguous() &&
              curr_idxes_in.is_contiguous() && depth_io.is_contiguous());
  int64_t num_packs = query.size(0);
  int64_t num_nugs = depth_io.size(0);
  auto depth_bound_start_out = at::zeros_like(query);
  auto depth_bound_end_out = at::zeros_like(query);
  find_next_depth_bound_cuda_impl(num_packs, num_nugs, query, start_idxes_in,
                                  curr_idxes_in, depth_io,
                                  depth_bound_start_out, depth_bound_end_out);
  return {depth_bound_start_out, depth_bound_end_out};

#else
  AT_ERROR(__func__);
#endif // WITH_CUDA
}

void clean_depth_bound_cuda_impl(int64_t num_packs, at::Tensor start_idxes_in,
                                 at::Tensor curr_idxes_in,
                                 at::Tensor depth_io_out,
                                 at::Tensor occluded_mask_out);
std::vector<at::Tensor> clean_depth_bound_cuda(at::Tensor start_idxes_in,
                                               at::Tensor curr_idxes_in,
                                               at::Tensor depth_io_in) {
#ifdef WITH_CUDA
  TORCH_CHECK(start_idxes_in.is_contiguous() && curr_idxes_in.is_contiguous() &&
              depth_io_in.is_contiguous() &&
              "Input tensors must be contiguous");
  int64_t num_packs = start_idxes_in.size(0);
  at::Tensor valid_mask_out =
      at::zeros({depth_io_in.size(0)}, depth_io_in.options().dtype(at::kBool));
  at::Tensor occluded_mask_out = at::ones(
      {start_idxes_in.size(0)}, start_idxes_in.options().dtype(at::kBool));
  clean_depth_bound_cuda_impl(num_packs, start_idxes_in, curr_idxes_in,
                              valid_mask_out, occluded_mask_out);

  return {valid_mask_out, occluded_mask_out};
#else
  AT_ERROR(__func__);
#endif // WITH_CUDA
}

void clean_depth_bound_cuda_impl2(int64_t num_packs, at::Tensor start_idxes_in,
                                  at::Tensor curr_idxes_in,
                                  at::Tensor depth_io_out);
at::Tensor clean_depth_bound_cuda2(at::Tensor start_idxes_in,
                                   at::Tensor curr_idxes_in,
                                   at::Tensor depth_io_in) {
#ifdef WITH_CUDA
  TORCH_CHECK(start_idxes_in.is_contiguous() && curr_idxes_in.is_contiguous() &&
              depth_io_in.is_contiguous() &&
              "Input tensors must be contiguous");
  int64_t num_packs = start_idxes_in.size(0);
  at::Tensor valid_mask_out =
      at::zeros({depth_io_in.size(0)}, depth_io_in.options().dtype(at::kBool));
  clean_depth_bound_cuda_impl2(num_packs, start_idxes_in, curr_idxes_in,
                               valid_mask_out);

  return valid_mask_out;
#else
  AT_ERROR(__func__);
#endif // WITH_CUDA
}

void sort_depth_samples_cuda_impl(int64_t num_packs, int64_t num_depths,
                                  at::Tensor dept_samples,
                                  at::Tensor start_idxes,
                                  at::Tensor append_depth,
                                  at::Tensor depth_delta_out);
at::Tensor sort_depth_samples_cuda(at::Tensor depth_samples,
                                   at::Tensor start_idxes,
                                   at::Tensor append_depth) {
#ifdef WITH_CUDA
  TORCH_CHECK(depth_samples.is_contiguous() && start_idxes.is_contiguous() &&
              append_depth.is_contiguous() &&
              "Input tensors must be contiguous");
  int64_t num_packs = start_idxes.size(0);
  at::Tensor depth_delta_out = at::empty_like(depth_samples);
  int64_t num_depths = depth_samples.size(0);
  sort_depth_samples_cuda_impl(num_packs, num_depths, depth_samples,
                               start_idxes, append_depth, depth_delta_out);

  return depth_delta_out;
#else
  AT_ERROR(__func__);
#endif // WITH_CUDA
}

void sort_ray_depth_cuda_impl(int64_t num_packs, int64_t num_depths,
                              const at::Tensor &start_idxes, at::Tensor depth);
at::Tensor sort_ray_depth_cuda(const at::Tensor &start_idxes,
                               at::Tensor depth_samples) {
#ifdef WITH_CUDA
  TORCH_CHECK(start_idxes.is_contiguous() && depth_samples.is_contiguous() &&
              "Input tensors must be contiguous");
  int64_t num_packs = start_idxes.numel();
  auto sorted_depth = depth_samples.contiguous();
  int64_t num_depths = sorted_depth.size(0);
  sort_ray_depth_cuda_impl(num_packs, num_depths, start_idxes, sorted_depth);

  return sorted_depth;
#else
  AT_ERROR(__func__);
#endif // WITH_CUDA
}

void cal_depth_delta_cuda_impl(int64_t num_packs, int64_t num_depths,
                               const at::Tensor &start_idxes, at::Tensor ridx,
                               at::Tensor depth, at::Tensor slope,
                               at::Tensor out_ridx, at::Tensor out_depth,
                               at::Tensor out_delta, at::Tensor out_slope);
std::vector<at::Tensor> cal_depth_delta_cuda(const at::Tensor &start_idxes,
                                             at::Tensor ridx, at::Tensor depth,
                                             at::Tensor slope) {
#ifdef WITH_CUDA
  TORCH_CHECK(start_idxes.is_contiguous() && ridx.is_contiguous() &&
              depth.is_contiguous());
  int64_t num_packs = start_idxes.numel();
  int64_t num_depths = depth.size(0);
  auto out_ridx = at::zeros({num_depths - num_packs}, ridx.options());
  auto out_depth = at::zeros({num_depths - num_packs, 1}, depth.options());
  auto out_delta = at::zeros({num_depths - num_packs, 1}, depth.options());
  auto out_slope = at::zeros({num_depths - num_packs, 1}, depth.options());
  cal_depth_delta_cuda_impl(num_packs, num_depths, start_idxes, ridx, depth,
                            slope, out_ridx, out_depth, out_delta, out_slope);
  return {out_ridx, out_depth, out_delta, out_slope};
#else
  AT_ERROR(__func__);
#endif // WITH_CUDA
}
} // namespace render_ops
