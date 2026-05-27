// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/device/device_histogram.cuh>
#include <cub/thread/thread_search.cuh>

#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>

#include <cuda/std/type_traits>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#if !TUNE_BASE

#  if TUNE_LOAD == 0
#    define TUNE_LOAD_MODIFIER cub::LOAD_DEFAULT
#  elif TUNE_LOAD == 1
#    define TUNE_LOAD_MODIFIER cub::LOAD_LDG
#  else // TUNE_LOAD == 2
#    define TUNE_LOAD_MODIFIER cub::LOAD_CA
#  endif // TUNE_LOAD

#  define TUNE_VEC_SIZE (1 << TUNE_VEC_SIZE_POW)

#  if TUNE_MEM_PREFERENCE == 0
constexpr cub::BlockHistogramMemoryPreference MEM_PREFERENCE = cub::GMEM;
#  elif TUNE_MEM_PREFERENCE == 1
constexpr cub::BlockHistogramMemoryPreference MEM_PREFERENCE = cub::SMEM;
#  else // TUNE_MEM_PREFERENCE == 2
constexpr cub::BlockHistogramMemoryPreference MEM_PREFERENCE = cub::BLEND;
#  endif // TUNE_MEM_PREFERENCE

#  if TUNE_LOAD_ALGORITHM_ID == 0
#    define TUNE_LOAD_ALGORITHM cub::BLOCK_LOAD_DIRECT
#  elif TUNE_LOAD_ALGORITHM_ID == 1
#    define TUNE_LOAD_ALGORITHM cub::BLOCK_LOAD_WARP_TRANSPOSE
#  else
#    define TUNE_LOAD_ALGORITHM cub::BLOCK_LOAD_STRIPED
#  endif // TUNE_LOAD_ALGORITHM_ID

template <typename SampleT, int NUM_CHANNELS, int NUM_ACTIVE_CHANNELS>
struct policy_hub_t
{
  struct policy_t : cub::ChainedPolicy<500, policy_t, policy_t>
  {
    static constexpr cub::BlockLoadAlgorithm load_algorithm =
      (TUNE_LOAD_ALGORITHM == cub::BLOCK_LOAD_STRIPED)
        ? (NUM_CHANNELS == 1 ? cub::BLOCK_LOAD_STRIPED : cub::BLOCK_LOAD_DIRECT)
        : TUNE_LOAD_ALGORITHM;

    using AgentHistogramPolicyT = cub::AgentHistogramPolicy<
      TUNE_THREADS,
      TUNE_ITEMS,
      load_algorithm,
      TUNE_LOAD_MODIFIER,
      TUNE_RLE_COMPRESS,
      MEM_PREFERENCE,
      TUNE_WORK_STEALING,
      TUNE_VEC_SIZE>;
  };

  using MaxPolicy = policy_t;
};
#endif // !TUNE_BASE

template <class SampleT, class OffsetT>
SampleT get_upper_level(OffsetT bins, OffsetT elements)
{
  if constexpr (cuda::std::is_integral_v<SampleT>)
  {
    if constexpr (sizeof(SampleT) < sizeof(OffsetT))
    {
      const SampleT max_key = ::cuda::std::numeric_limits<SampleT>::max();
      return static_cast<SampleT>(std::min(bins, static_cast<OffsetT>(max_key)));
    }
    else
    {
      return static_cast<SampleT>(bins);
    }
  }

  return static_cast<SampleT>(elements);
}

// ----------------------------------------------------------------------------
// Correctness checks for the histogram benchmarks.
//
// The verifier runs OUTSIDE NVBench's timed region (once per cell, before
// `state.exec`) and compares the optimized CUB result bin-by-bin against an
// independent reference histogram built with `thrust::for_each` + global
// `atomicAdd`. The reference shares no kernels with the dispatch under test,
// so silent dispatch failures or sample-loss bugs cannot inflate the
// reported bandwidth.
// ----------------------------------------------------------------------------

inline void bench_check_cuda(cudaError_t e, const char* what)
{
  if (e != cudaSuccess)
  {
    throw std::runtime_error(std::string{"bench correctness check: "} + what + " -> " + cudaGetErrorString(e));
  }
}

// Functor for the EVEN path. Closed-form bin computation: matches what
// DispatchEven's ScaleTransform does (sample - lo) * num_bins / (hi - lo),
// but runs as a flat one-thread-per-pixel kernel with global atomicAdd into
// the per-channel reference histograms.
template <int NumChannels, int NumActiveChannels, typename SampleT, typename CounterT, typename OffsetT>
struct bench_ref_even_op
{
  const SampleT* d_input;
  CounterT* d_hist[NumActiveChannels];
  OffsetT num_pixels;
  int num_bins;
  SampleT lower_level;
  SampleT upper_level;

  __device__ void operator()(OffsetT pixel) const
  {
    if (pixel >= num_pixels)
    {
      return;
    }

    const double L     = static_cast<double>(lower_level);
    const double U     = static_cast<double>(upper_level);
    const double scale = static_cast<double>(num_bins) / (U - L);

    const SampleT* px = d_input + static_cast<std::size_t>(pixel) * NumChannels;
#pragma unroll
    for (int c = 0; c < NumActiveChannels; ++c)
    {
      const SampleT s = px[c];
      if (s < lower_level || s >= upper_level)
      {
        continue;
      }
      int bin = static_cast<int>((static_cast<double>(s) - L) * scale);
      if (bin >= num_bins)
      {
        bin = num_bins - 1;
      }
      if (bin >= 0)
      {
        atomicAdd(d_hist[c] + bin, CounterT{1});
      }
    }
  }
};

// Functor for the RANGE path. Levels are arbitrary, so we classify with
// `cub::UpperBound` (binary search on the per-channel level array).
template <int NumChannels, int NumActiveChannels, typename SampleT, typename CounterT, typename OffsetT>
struct bench_ref_range_op
{
  const SampleT* d_input;
  CounterT* d_hist[NumActiveChannels];
  const SampleT* d_levels[NumActiveChannels];
  OffsetT num_pixels;
  int num_levels;

  __device__ void operator()(OffsetT pixel) const
  {
    if (pixel >= num_pixels)
    {
      return;
    }

    const int num_bins = num_levels - 1;
    const SampleT* px  = d_input + static_cast<std::size_t>(pixel) * NumChannels;

#pragma unroll
    for (int c = 0; c < NumActiveChannels; ++c)
    {
      const SampleT s = px[c];
      const int idx   = cub::UpperBound(d_levels[c], num_levels, s);
      const int bin   = idx - 1;
      if (bin >= 0 && bin < num_bins)
      {
        atomicAdd(d_hist[c] + bin, CounterT{1});
      }
    }
  }
};

template <typename CounterT>
void bench_compare_histograms(
  const std::vector<thrust::host_vector<CounterT>>& opt_hists,
  const std::vector<thrust::host_vector<CounterT>>& ref_hists,
  const char* bench_label,
  int num_channels)
{
  for (int c = 0; c < num_channels; ++c)
  {
    const auto& opt = opt_hists[c];
    const auto& ref = ref_hists[c];
    if (opt.size() != ref.size())
    {
      throw std::runtime_error(std::string{"bench correctness check ["} + bench_label + "]: channel "
                               + std::to_string(c) + " size mismatch: opt=" + std::to_string(opt.size())
                               + " ref=" + std::to_string(ref.size()));
    }
    long long opt_total = 0;
    long long ref_total = 0;
    int first_mismatch  = -1;
    for (std::size_t b = 0; b < opt.size(); ++b)
    {
      opt_total += static_cast<long long>(opt[b]);
      ref_total += static_cast<long long>(ref[b]);
      if (first_mismatch < 0 && opt[b] != ref[b])
      {
        first_mismatch = static_cast<int>(b);
      }
    }
    if (opt_total != ref_total || first_mismatch >= 0)
    {
      char msg[512];
      std::snprintf(
        msg,
        sizeof(msg),
        "bench correctness check [%s]: channel=%d total opt=%lld ref=%lld; first mismatched bin=%d "
        "(opt=%lld ref=%lld)",
        bench_label,
        c,
        opt_total,
        ref_total,
        first_mismatch,
        first_mismatch >= 0 ? static_cast<long long>(opt[first_mismatch]) : -1LL,
        first_mismatch >= 0 ? static_cast<long long>(ref[first_mismatch]) : -1LL);
      throw std::runtime_error(msg);
    }
  }
}

template <typename CounterT>
std::vector<thrust::host_vector<CounterT>>
bench_snapshot_histograms(const std::vector<thrust::device_vector<CounterT>>& d_hists)
{
  std::vector<thrust::host_vector<CounterT>> out;
  out.reserve(d_hists.size());
  for (const auto& d : d_hists)
  {
    out.emplace_back(d);
  }
  return out;
}

// Public entry point: EVEN bench. Builds the reference on device with
// thrust::for_each + atomicAdd into a fresh per-channel histogram, then
// compares bin-by-bin against the already-produced optimized histograms.
template <int NumChannels,
          int NumActiveChannels,
          typename SampleT,
          typename CounterT,
          typename OffsetT>
void bench_verify_histogram_even(
  const thrust::device_vector<SampleT>& d_input,
  const std::vector<thrust::device_vector<CounterT>>& opt_hists_d,
  OffsetT num_pixels,
  int num_bins,
  SampleT lower_level,
  SampleT upper_level,
  const char* bench_label)
{
  static_assert(NumActiveChannels >= 1 && NumActiveChannels <= NumChannels);

  std::vector<thrust::device_vector<CounterT>> ref_hists_d(NumActiveChannels);
  bench_ref_even_op<NumChannels, NumActiveChannels, SampleT, CounterT, OffsetT> op{};
  op.d_input     = thrust::raw_pointer_cast(d_input.data());
  op.num_pixels  = num_pixels;
  op.num_bins    = num_bins;
  op.lower_level = lower_level;
  op.upper_level = upper_level;
  for (int c = 0; c < NumActiveChannels; ++c)
  {
    ref_hists_d[c].assign(num_bins, CounterT{0});
    op.d_hist[c] = thrust::raw_pointer_cast(ref_hists_d[c].data());
  }

  thrust::for_each(
    thrust::counting_iterator<OffsetT>(0), thrust::counting_iterator<OffsetT>(num_pixels), op);
  bench_check_cuda(cudaGetLastError(), "ref-even launch");
  bench_check_cuda(cudaDeviceSynchronize(), "ref-even sync");

  const auto opt_hists = bench_snapshot_histograms(opt_hists_d);
  const auto ref_hists = bench_snapshot_histograms(ref_hists_d);
  bench_compare_histograms(opt_hists, ref_hists, bench_label, NumActiveChannels);
}

template <int NumChannels,
          int NumActiveChannels,
          typename SampleT,
          typename CounterT,
          typename OffsetT>
void bench_verify_histogram_range(
  const thrust::device_vector<SampleT>& d_input,
  const std::vector<thrust::device_vector<CounterT>>& opt_hists_d,
  const std::vector<thrust::device_vector<SampleT>>& d_levels_per_channel,
  OffsetT num_pixels,
  const char* bench_label)
{
  static_assert(NumActiveChannels >= 1 && NumActiveChannels <= NumChannels);

  const int num_levels = static_cast<int>(d_levels_per_channel[0].size());
  const int num_bins   = num_levels - 1;

  std::vector<thrust::device_vector<CounterT>> ref_hists_d(NumActiveChannels);
  bench_ref_range_op<NumChannels, NumActiveChannels, SampleT, CounterT, OffsetT> op{};
  op.d_input    = thrust::raw_pointer_cast(d_input.data());
  op.num_pixels = num_pixels;
  op.num_levels = num_levels;
  for (int c = 0; c < NumActiveChannels; ++c)
  {
    ref_hists_d[c].assign(num_bins, CounterT{0});
    op.d_hist[c]   = thrust::raw_pointer_cast(ref_hists_d[c].data());
    op.d_levels[c] = thrust::raw_pointer_cast(d_levels_per_channel[c].data());
  }

  thrust::for_each(
    thrust::counting_iterator<OffsetT>(0), thrust::counting_iterator<OffsetT>(num_pixels), op);
  bench_check_cuda(cudaGetLastError(), "ref-range launch");
  bench_check_cuda(cudaDeviceSynchronize(), "ref-range sync");

  const auto opt_hists = bench_snapshot_histograms(opt_hists_d);
  const auto ref_hists = bench_snapshot_histograms(ref_hists_d);
  bench_compare_histograms(opt_hists, ref_hists, bench_label, NumActiveChannels);
}
