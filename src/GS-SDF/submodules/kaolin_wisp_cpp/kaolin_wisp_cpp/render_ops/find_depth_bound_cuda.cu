#include <ATen/ATen.h>

namespace render_ops {
typedef unsigned int uint;

__global__ void find_depth_interval_cuda_kernel(int64_t num_packs,
                                                int64_t num_nugs,
                                                const int *start_idxes_in,
                                                const float2 *depth_io,
                                                bool *break_mask_out) {
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;
  if (tidx < num_packs) {
    uint iidx = start_idxes_in[tidx];
    uint max_iidx =
        (tidx == num_packs - 1) ? num_nugs : start_idxes_in[tidx + 1];

    while (iidx < (max_iidx - 1)) {
      if (abs(depth_io[iidx].y - depth_io[iidx + 1].x) > 1e-6) {
        // printf("depth_io[%d].y = %f, depth_io[%d].x = %f\n", iidx,
        //        depth_io[iidx].y, iidx + 1, depth_io[iidx + 1].x);
        break_mask_out[tidx] = true;
      }
      iidx++;
    }
  }
}

void find_depth_interval_cuda_impl(int64_t num_packs, int64_t num_nugs,
                                   at::Tensor start_idxes_in,
                                   at::Tensor depth_io,
                                   at::Tensor break_mask_out) {
  find_depth_interval_cuda_kernel<<<(num_packs + 1023) / 1024, 1024>>>(
      num_packs, num_nugs, start_idxes_in.data_ptr<int>(),
      reinterpret_cast<float2 *>(depth_io.data_ptr<float>()),
      break_mask_out.data_ptr<bool>());
}

__global__ void find_depth_bound_cuda_kernel(
    int64_t num_packs, int64_t num_nugs, const float *query_depth,
    const int *start_idxes_in, const int *curr_idxes_in, const float2 *depth,
    int *curr_idxes_out, bool *change_mask_out, bool *inbound_mask_out,
    bool *break_mask_out) {
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;
  if (tidx < num_packs && curr_idxes_in[tidx] > -1) {

    uint iidx = curr_idxes_in[tidx];
    uint max_iidx =
        (tidx == num_packs - 1) ? num_nugs : start_idxes_in[tidx + 1];
    float query = query_depth[tidx];

    float entry = depth[iidx].x;
    float exit = depth[iidx].y;
    while (true) {
      if (query >= entry && query <= exit) {
        curr_idxes_out[tidx] = iidx;
        inbound_mask_out[tidx] = true;
        return;
      } else if (query < entry) {
        curr_idxes_out[tidx] = iidx;
        inbound_mask_out[tidx] = true;
        break_mask_out[tidx] = true;
        return;
      }
      iidx++;
      change_mask_out[tidx] = true;
      if (iidx >= max_iidx) {
        return;
      }

      entry = depth[iidx].x;
      if ((entry - exit) > 1e-6) {
        break_mask_out[tidx] = true;
      }
      exit = depth[iidx].y;
    }
  }
}

void find_depth_bound_cuda_impl(int64_t num_packs, int64_t num_nugs,
                                at::Tensor query, at::Tensor start_idxes_in,
                                at::Tensor curr_idxes_in, at::Tensor depth,
                                at::Tensor curr_idxes_out,
                                at::Tensor change_mask_out,
                                at::Tensor inbound_mask_out,
                                at::Tensor break_mask_out) {
  find_depth_bound_cuda_kernel<<<(num_packs + 1023) / 1024, 1024>>>(
      num_packs, num_nugs, query.data_ptr<float>(),
      start_idxes_in.data_ptr<int>(), curr_idxes_in.data_ptr<int>(),
      reinterpret_cast<float2 *>(depth.data_ptr<float>()),
      curr_idxes_out.data_ptr<int>(), change_mask_out.data_ptr<bool>(),
      inbound_mask_out.data_ptr<bool>(), break_mask_out.data_ptr<bool>());
}

__global__ void
find_depth_bound_cuda_kernel2(int64_t num_packs, int64_t num_nugs,
                              bool *undone_mask, float *query_depth,
                              const int *start_idxes_in, int *curr_idxes_in,
                              const float2 *depth_io, bool *break_mask_out) {
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;
  if (tidx < num_packs && curr_idxes_in[tidx] > -1) {
    if (!undone_mask[tidx]) {
      return;
    }
    uint max_iidx =
        (tidx == num_packs - 1) ? num_nugs : start_idxes_in[tidx + 1];
    float query = query_depth[tidx];

    int iidx = curr_idxes_in[tidx];
    float entry = depth_io[iidx].x;
    float exit = depth_io[iidx].y;
    while (true) {
      if ((query >= entry && query <= exit) || (query < entry)) {
        break;
      }
      iidx++;
      if (iidx >= max_iidx) {
        // it matter: helps improve strcuture
        if (query > exit) {
          undone_mask[tidx] = false;
        }
        return;
      }

      entry = depth_io[iidx].x;
      if ((entry - exit) > 1e-6) {
        query_depth[tidx] = entry + 1e-6f;
        break_mask_out[tidx] = true;
      }
      exit = depth_io[iidx].y;
    }
    curr_idxes_in[tidx] = iidx;
  }
}

void find_depth_bound_cuda_impl2(int64_t num_packs, int64_t num_nugs,
                                 at::Tensor undone_mask, at::Tensor query,
                                 at::Tensor start_idxes_in,
                                 at::Tensor curr_idxes_in, at::Tensor depth_io,
                                 at::Tensor break_mask_out) {
  find_depth_bound_cuda_kernel2<<<(num_packs + 1023) / 1024, 1024>>>(
      num_packs, num_nugs, undone_mask.data_ptr<bool>(),
      query.data_ptr<float>(), start_idxes_in.data_ptr<int>(),
      curr_idxes_in.data_ptr<int>(),
      reinterpret_cast<float2 *>(depth_io.data_ptr<float>()),
      break_mask_out.data_ptr<bool>());
}

__global__ void find_next_depth_bound_cuda_kernel(
    int64_t num_packs, int64_t num_nugs, float *query_depth,
    const int *start_idxes_in, int *curr_idxes_in, const float2 *depth_io,
    float *depth_bound_start_out, float *depth_bound_end_out) {
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;
  if (tidx < num_packs && curr_idxes_in[tidx] > -1) {
    uint max_iidx =
        (tidx == num_packs - 1) ? num_nugs : start_idxes_in[tidx + 1];
    float query = query_depth[tidx];

    int iidx = curr_idxes_in[tidx] + 1;
    float entry, exit;
    if (iidx >= max_iidx) {
      int tmp_iidx = curr_idxes_in[tidx];
      depth_bound_start_out[tidx] = query;
      depth_bound_end_out[tidx] = depth_io[tmp_iidx].y;
      return;
    } else {
      entry = depth_io[iidx].x;
      exit = depth_io[iidx].y;
      depth_bound_start_out[tidx] = entry;
    }
    bool get_entry = false;
    while (iidx < max_iidx) {
      iidx++;
      if (iidx >= max_iidx) {
        depth_bound_end_out[tidx] = exit;
        return;
      }

      entry = depth_io[iidx].x;
      if ((entry - exit) > 1e-6) {
        if (!get_entry) {
          depth_bound_start_out[tidx] = entry;
          get_entry = true;
        } else {
          depth_bound_end_out[tidx] = entry;
          return;
        }
      }
      exit = depth_io[iidx].y;
    }
  }
}

void find_next_depth_bound_cuda_impl(
    int64_t num_packs, int64_t num_nugs, at::Tensor query,
    at::Tensor start_idxes_in, at::Tensor curr_idxes_in, at::Tensor depth_io,
    at::Tensor depth_bound_start_out, at::Tensor depth_bound_end_out) {
  find_next_depth_bound_cuda_kernel<<<(num_packs + 1023) / 1024, 1024>>>(
      num_packs, num_nugs, query.data_ptr<float>(),
      start_idxes_in.data_ptr<int>(), curr_idxes_in.data_ptr<int>(),
      reinterpret_cast<float2 *>(depth_io.data_ptr<float>()),
      depth_bound_start_out.data_ptr<float>(),
      depth_bound_end_out.data_ptr<float>());
}

__global__ void clean_depth_bound_cuda_kernel(int64_t num_packs,
                                              const int *start_idxes_in,
                                              const int *curr_idxes_in,
                                              bool *valid_mask_out,
                                              bool *occluded_mask_out) {
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;
  if (tidx < num_packs) {
    uint sidx = start_idxes_in[tidx];
    uint cidx = curr_idxes_in[tidx];

    if (sidx == cidx) {
      occluded_mask_out[tidx] = false;
    }

    while (sidx < cidx) {
      valid_mask_out[sidx] = true;
      sidx++;
    }
  }
}

void clean_depth_bound_cuda_impl(int64_t num_packs, at::Tensor start_idxes_in,
                                 at::Tensor curr_idxes_in,
                                 at::Tensor valid_mask_out,
                                 at::Tensor occluded_mask_out) {
  clean_depth_bound_cuda_kernel<<<(num_packs + 1023) / 1024, 1024>>>(
      num_packs, start_idxes_in.data_ptr<int>(), curr_idxes_in.data_ptr<int>(),
      valid_mask_out.data_ptr<bool>(), occluded_mask_out.data_ptr<bool>());
}

__global__ void clean_depth_bound_cuda_kernel2(int64_t num_packs,
                                               const int *start_idxes_in,
                                               const int *curr_idxes_in,
                                               bool *valid_mask_out) {
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;
  if (tidx < num_packs) {
    uint sidx = start_idxes_in[tidx];
    uint cidx = curr_idxes_in[tidx];

    while (sidx <= cidx) {
      valid_mask_out[sidx] = true;
      sidx++;
    }
  }
}

void clean_depth_bound_cuda_impl2(int64_t num_packs, at::Tensor start_idxes_in,
                                  at::Tensor curr_idxes_in,
                                  at::Tensor valid_mask_out) {
  clean_depth_bound_cuda_kernel2<<<(num_packs + 1023) / 1024, 1024>>>(
      num_packs, start_idxes_in.data_ptr<int>(), curr_idxes_in.data_ptr<int>(),
      valid_mask_out.data_ptr<bool>());
}

__global__ void sort_depth_samples_cuda_kernel(int64_t num_packs,
                                               int64_t num_depths,
                                               const float *depth_samples,
                                               const int *start_idxes_in,
                                               const float *append_depth,
                                               float *depth_delta_out) {
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;
  if (tidx < num_packs) {
    uint iidx = start_idxes_in[tidx];
    uint max_iidx =
        (tidx == num_packs - 1) ? num_depths : start_idxes_in[tidx + 1];

    while (iidx < (max_iidx - 1)) {
      depth_delta_out[iidx] = depth_samples[iidx + 1] - depth_samples[iidx];
      iidx++;
    }
    depth_delta_out[iidx] = append_depth[tidx] - depth_samples[iidx];
  }
}
void sort_depth_samples_cuda_impl(int64_t num_packs, int64_t num_depths,
                                  at::Tensor depth_samples,
                                  at::Tensor start_idxes_in,
                                  at::Tensor append_depth,
                                  at::Tensor depth_delta_out) {
  sort_depth_samples_cuda_kernel<<<(num_packs + 1023) / 1024, 1024>>>(
      num_packs, num_depths, depth_samples.data_ptr<float>(),
      start_idxes_in.data_ptr<int>(), append_depth.data_ptr<float>(),
      depth_delta_out.data_ptr<float>());
}

__global__ void sort_ray_depth_cuda_kernel(int64_t num_packs,
                                           int64_t num_depths,
                                           const int *start_idxes,
                                           float *depth) {
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;
  if (tidx < num_packs) {

    uint sidx = start_idxes[tidx];
    uint max_iidx =
        (tidx == num_packs - 1) ? num_depths : start_idxes[tidx + 1];

    uint iidx = sidx + 1;
    while (iidx < max_iidx) {
      uint bidx = iidx;
      while (bidx > sidx) {
        if (depth[bidx] < depth[bidx - 1]) {
          float tmp = depth[bidx];
          depth[bidx] = depth[bidx - 1];
          depth[bidx - 1] = tmp;
        } else {
          break;
        }
        bidx--;
      }
      iidx++;
    }
  }
}
void sort_ray_depth_cuda_impl(int64_t num_packs, int64_t num_depths,
                              const at::Tensor &start_idxes, at::Tensor depth) {
  sort_ray_depth_cuda_kernel<<<(num_packs + 1023) / 1024, 1024>>>(
      num_packs, num_depths, start_idxes.data_ptr<int>(),
      depth.data_ptr<float>());
}

__global__ void cal_depth_delta_cuda_kernel(
    int64_t num_packs, int64_t num_depths, const int *start_idxes,
    const long *ridx, float *depth, float *slope, long *out_ridx,
    float *out_depth, float *out_delta, float *out_slope) {
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;
  if (tidx < num_packs) {
    uint iidx = start_idxes[tidx];
    uint max_iidx =
        (tidx == num_packs - 1) ? num_depths : start_idxes[tidx + 1];

    while (iidx < (max_iidx - 1)) {
      out_ridx[iidx - tidx] = ridx[iidx];
      // out_depth[iidx - tidx] = 0.5f * (depth[iidx] + depth[iidx + 1]);
      out_depth[iidx - tidx] = depth[iidx];
      out_delta[iidx - tidx] = depth[iidx + 1] - depth[iidx];
      out_slope[iidx - tidx] = slope[iidx];
      iidx++;
    }
    // out_delta[iidx - tidx - 1] = 100.0f;
  }
}
void cal_depth_delta_cuda_impl(int64_t num_packs, int64_t num_depths,
                               const at::Tensor &start_idxes, at::Tensor ridx,
                               at::Tensor depth, at::Tensor slope,
                               at::Tensor out_ridx, at::Tensor out_depth,
                               at::Tensor out_delta, at::Tensor out_slope) {
  cal_depth_delta_cuda_kernel<<<(num_packs + 1023) / 1024, 1024>>>(
      num_packs, num_depths, start_idxes.data_ptr<int>(), ridx.data_ptr<long>(),
      depth.data_ptr<float>(), slope.data_ptr<float>(),
      out_ridx.data_ptr<long>(), out_depth.data_ptr<float>(),
      out_delta.data_ptr<float>(), out_slope.data_ptr<float>());
}
} // namespace render_ops
