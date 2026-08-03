# Cooperative direct-output low-bin histogram experiment

## Scope

This experiment compares five implementations on an NVIDIA B200:

- `BAS`: established main-overlay baseline;
- `SST`: non-cooperative static-SMEM privatization;
- `CST`: cooperative direct-output static-SMEM privatization;
- `SDY`: non-cooperative dynamic-SMEM privatization;
- `CDY`: cooperative direct-output dynamic-SMEM privatization.

The cooperative kernels zero the output histograms, reset the work-stealing
queue, perform a grid-wide synchronization, run the same static or dynamic
per-block `AgentHistogram` accumulation as the corresponding non-cooperative
kernel, and call the same device-atomic `StoreOutput`. They do not allocate a
per-block global staging slab and do not perform an atomic-free gather.

## Source and machine provenance

- Experiment branch: `pr/histocache/cooperative-direct-output-smem`.
- Raw branch base: `f16c464e64d5786de2e0bf536479e0ef9ae57b30`.
- Main-overlay benchmark source: `0ad3f9967aba832e768f5e3a52bcd5900454d22c` plus the raw experiment's input-generator overlay. The established binaries report a dirty source tree because of that overlay.
- Current `upstream/main` observed during the experiment: `5e1fc1691f9d93ed90dd463fe290c7022015c55f`.
- GPU: NVIDIA B200, compute capability 10.0, 183359 MiB.
- Driver: 580.126.09.
- CUDA compiler: 13.3.33.
- Host compiler: GCC 13.3.0.
- CMake: 4.3.2.
- Build architecture: `100a`.

The comparison against `main` uses the established raw experiment's
main-overlay binaries, not a build from current `upstream/main`. A fresh current
main overlay was attempted but is source-incompatible with the raw branch's
cache-query and atomic-helper APIs. Consequently, `main` in this report means
the exact established baseline above.

## Characterization grid

- Families: EVEN, RANGE, multi-EVEN, multi-RANGE.
- Sample types: I32 and F64.
- Element counts: 1 Mi, 16 Mi, 64 Mi, 256 Mi, 1,073,741,824, and 2,000,000,000.
- Bin counts: 8, 16, 32, 64, 128, 256, and 512.
- Input shapes: all 15 raw-branch shapes.
- Samples per NVBench invocation: 3, with `--min-time 0.02`.
- Counter configurations:
  - c32: 32-bit local/output counters and 32-bit offsets;
  - u64: 32-bit local counters, 64-bit output counters, and 64-bit offsets.

Static kernels have a compile-time capacity of 256 bins and are therefore
structurally invalid at 512. Dynamic variants and main were measured at 512.

The c32 multi-channel forced variants are structurally invalid at
1,073,741,824 and 2,000,000,000 pixels because `pixels * channels` overflows a
32-bit offset. Exactly 208 such forced cells were dropped with no launch tag;
the u64 sweep covers every one of those cells. No u64 cell was dropped.

## Exact build and sweep commands

```bash
CXX=/usr/bin/c++ CUDAHOSTCXX=/usr/bin/c++ \
  /home/shadeform/.local/cmake-venv/bin/cmake \
  --preset cub-benchmark \
  -B build/cooperative-direct-output-cub-benchmark-cmake4 \
  -DCMAKE_CUDA_ARCHITECTURES=100a

/home/shadeform/.local/cmake-venv/bin/cmake \
  --build build/cooperative-direct-output-cub-benchmark-cmake4 \
  --target \
    cub.bench.histogram.even.base \
    cub.bench.histogram.range.base \
    cub.bench.histogram.multi.even.base \
    cub.bench.histogram.multi.range.base \
  -j 8

HIST_BENCH_BUILD=build/cooperative-direct-output-cub-benchmark-cmake4 \
  bash cub/benchmarks/bench/histogram/build_u64_variants.sh
```

```bash
HIST_SWEEP_BRANCH_BIN=build/cooperative-direct-output-cub-benchmark-cmake4/bin \
HIST_SWEEP_MAIN_BIN=/home/shadeform/cccl/autocuda/worktrees/main-baseline/build/cub-benchmark/bin \
HIST_SWEEP_SKIP_HITRATE=1 \
HIST_SWEEP_BINS='8 16 32 64 128 256 512' \
HIST_SWEEP_ELEMENTS='1048576 16777216 67108864 268435456 1073741824 2000000000' \
HIST_SWEEP_SAMPLES='I32 F64' \
HIST_SWEEP_REPEATS=3 \
HIST_SWEEP_MIN_TIME=0.02 \
HIST_SWEEP_TIMEOUT=240 \
bash cub/benchmarks/bench/histogram/run_perbinary_sweep.sh \
  cooperative-direct-output-smem-c32 \
  --algorithms BAS SST CST SDY CDY
```

The initial c32 multi-channel pass exposed an extent-dependent allocation in
the cache-slot metadata query. Slot sizing is extent-independent, so the query
was corrected to probe `min(N, 1 Mi)`. The multi-channel recovery command was:

```bash
/home/shadeform/.local/viz-venv/bin/python3 \
  cub/benchmarks/bench/histogram/histogram_algo_sweep.py \
  --branch-bin-dir build/cooperative-direct-output-cub-benchmark-cmake4/bin \
  --main-bin-dir /home/shadeform/cccl/autocuda/worktrees/main-baseline/build/cub-benchmark/bin \
  --binaries multi_even multi_range \
  --samples I32 F64 \
  --bins 8 16 32 64 128 256 512 \
  --elements 1048576 16777216 67108864 268435456 1073741824 2000000000 \
  --shapes concentrated:1.0 concentrated:0.75 concentrated:0.5 concentrated:0.25 concentrated:0.0 powerlaw:0.75 powerlaw:0.25 hash_synonym stale_resident:0.5 stale_resident:0.25 temporal_phases:0.10 strided_sweep sawtooth poison sawtooth:8192:2654435761:1 \
  --repeats 3 --min-time 0.02 --timeout 240 \
  --algorithms BAS SST CST SDY CDY \
  --out autocuda/results/cooperative-direct-output-smem-c32-resume/multi_full.json
```

```bash
HIST_SWEEP_BRANCH_BIN=build/cooperative-direct-output-cub-benchmark-cmake4/bin \
HIST_SWEEP_MAIN_BIN=/home/shadeform/cccl/autocuda/worktrees/main-baseline/build/cub-benchmark/bin \
HIST_SWEEP_SKIP_HITRATE=1 \
HIST_SWEEP_BINS='8 16 32 64 128 256 512' \
HIST_SWEEP_ELEMENTS='1048576 16777216 67108864 268435456 1073741824 2000000000' \
HIST_SWEEP_SAMPLES='I32 F64' \
HIST_SWEEP_REPEATS=3 \
HIST_SWEEP_MIN_TIME=0.02 \
HIST_SWEEP_TIMEOUT=240 \
bash cub/benchmarks/bench/histogram/run_perbinary_sweep.sh \
  cooperative-direct-output-smem-u64 \
  --binary-suffix .u64 \
  --algorithms BAS SST CST SDY CDY
```

The full u64 run lasted from 2026-08-02 16:18 UTC through 2026-08-03 01:13 UTC.

## Artifact validation

`histogram_cooperative_analysis.py` validated and summarized the results with:

```bash
/home/shadeform/.local/viz-venv/bin/python3 \
  cub/benchmarks/bench/histogram/histogram_cooperative_analysis.py \
  --c32 autocuda/results/cooperative-direct-output-smem-c32/algo_sweep_full.json \
  --u64 autocuda/results/cooperative-direct-output-smem-u64/algo_sweep_full.json \
  --outdir autocuda/results/cooperative-direct-output-smem-analysis
```

It validated 44,400 positive measurements. Expected counts are recorded in
`artifact_validation.json`; the complete cell-level table is
`measurements.csv`.

## Measured results

Ratios below are geometric means. Small means 1 Mi and 16 Mi elements. Large
means 256 Mi, 1,073,741,824, and 2,000,000,000 elements. The 64 Mi size remains
in the raw data but is intentionally excluded from both size buckets.

### u64 complete characterization

| Family | Size | CST/SST | CDY/SDY | SST/main | CST/main | SDY/main | CDY/main |
|---|---:|---:|---:|---:|---:|---:|---:|
| EVEN | small | 0.965× | 0.964× | 1.678× | 1.619× | 1.701× | 1.639× |
| EVEN | large | 1.000× | 1.000× | 2.474× | 2.473× | 2.455× | 2.454× |
| RANGE | small | 0.966× | 0.962× | 1.102× | 1.065× | 1.117× | 1.075× |
| RANGE | large | 1.006× | 1.001× | 1.083× | 1.090× | 1.165× | 1.166× |
| multi-EVEN | small | 0.981× | 0.985× | 1.779× | 1.746× | 1.589× | 1.565× |
| multi-EVEN | large | 1.000× | 1.000× | 2.509× | 2.509× | 1.963× | 1.963× |
| multi-RANGE | small | 0.907× | 0.984× | 1.024× | 0.929× | 1.017× | 1.000× |
| multi-RANGE | large | 1.002× | 1.000× | 1.059× | 1.061× | 1.175× | 1.175× |

Across every u64 family, size, bin, shape, and sample cell, CST is 0.985× SST
and CDY is 0.991× SDY. Against main, the corresponding overall geometric means
are 1.501× for CST and 1.482× for CDY.

The c32 results reproduce the same relationship. The largest c32/u64 aggregate
difference in the table is the set of large multi-channel cells that c32 cannot
represent and u64 adds.

### Sample-type split for cooperative/non-cooperative ratios

| Family | Sample | Size | CST/SST | CDY/SDY |
|---|---|---:|---:|---:|
| EVEN | I32 | small | 0.979× | 0.979× |
| EVEN | F64 | small | 0.951× | 0.948× |
| RANGE | I32 | small | 0.970× | 0.957× |
| RANGE | F64 | small | 0.962× | 0.967× |
| multi-EVEN | I32 | small | 0.975× | 0.986× |
| multi-EVEN | F64 | small | 0.988× | 0.984× |
| multi-RANGE | I32 | small | 0.891× | 0.976× |
| multi-RANGE | F64 | small | 0.923× | 0.992× |
| EVEN | I32/F64 | large | 0.999–1.000× | 1.000× |
| RANGE | I32/F64 | large | 1.003–1.010× | 1.000–1.002× |
| multi-EVEN | I32/F64 | large | 1.000× | 1.000× |
| multi-RANGE | I32/F64 | large | 0.994–1.011× | 1.000× |

### Important exceptions

- Worst CST/SST: 0.774× at multi-RANGE, I32, 1 Mi, 64 bins, `powerlaw:0.75`.
- Worst CDY/SDY: 0.883× at EVEN, F64, 1 Mi, 256 bins, `powerlaw:0.75`.
- Worst CST/main: 0.753× at RANGE, I32, 16 Mi, 8 bins, `temporal_phases:0.10`.
- Worst CDY/main: 0.629× at the same RANGE workload.
- Dynamic multi-EVEN has isolated large F64 regressions near 0.690× main for concentrated low-bin inputs, despite a 1.963× large-workload family geometric mean.
- The largest dynamic wins over main occur at 512 bins: up to 7.836× EVEN, 6.423× RANGE, 6.942× multi-EVEN, and 4.523× multi-RANGE for CDY.

## Conclusions

### Measured

- Cooperative direct output does not improve saturated throughput. At large
  element counts, both cooperative variants are effectively tied with their
  non-cooperative counterparts.
- The extra cooperative launch and grid synchronization produce a repeatable
  small-workload cost. It is usually 2–5%, but static multi-RANGE loses 9.3% in
  aggregate and has individual regressions above 20%.
- Dynamic cooperative direct output is the less risky variant. Its complete-u64
  geometric mean is 0.991× the non-cooperative dynamic kernel, versus 0.985× for
  cooperative static.
- The optimized raw-branch SMEM kernels substantially beat the established main
  baseline on EVEN and multi-EVEN. RANGE and multi-RANGE are more shape-sensitive
  and retain significant low-bin temporal-phase regressions.
- The c32 and u64 single-channel results agree closely, and the complete u64
  multi-channel data confirms the same saturated-throughput result at the
  billion-element sizes.

### Hypotheses

- The small-workload gap is consistent with fixed cooperative-launch and
  grid-synchronization latency, because it vanishes as element count grows while
  the accumulation and output paths remain otherwise identical.
- The unusually large static multi-RANGE penalty may additionally reflect its
  narrower static launch configuration and short search-transform workload,
  where the fixed synchronization cost is a larger fraction of total runtime.
  This experiment did not independently isolate those components.

The measured data does not support replacing the non-cooperative variants with
cooperative direct output for performance. The cooperative design is throughput
neutral at saturation but has a clear latency cost and no observed compensating
large-workload gain.

## Validation

The post-format smoke matrix used all four families, I32/F64, bins 8/256/512,
two shapes, and all five series. Its benchmark-side independent reference check
passed, and the sweep accepted every forced cell only after observing the exact
requested launch tag.

```bash
/home/shadeform/.local/viz-venv/bin/python3 \
  cub/benchmarks/bench/histogram/histogram_algo_sweep.py \
  --branch-bin-dir build/cooperative-direct-output-cub-benchmark-cmake4/bin \
  --main-bin-dir /home/shadeform/cccl/autocuda/worktrees/main-baseline/build/cub-benchmark/bin \
  --binaries even range multi_even multi_range \
  --samples I32 F64 --bins 8 256 512 --elements 1048576 \
  --shapes concentrated:1.0 poison \
  --repeats 1 --min-time 0.001 --timeout 120 \
  --out autocuda/results/cooperative-direct-output-smem-smoke/post_format_smoke.json \
  --algorithms BAS SST CST SDY CDY
```

Core Catch2 validation used a fresh build:

```bash
CXX=/usr/bin/c++ CUDAHOSTCXX=/usr/bin/c++ \
  /home/shadeform/.local/cmake-venv/bin/cmake \
  --preset cub-cpp20 \
  -B build/cooperative-direct-output-cub-test \
  -DCMAKE_CUDA_ARCHITECTURES=100a

/home/shadeform/.local/cmake-venv/bin/cmake \
  --build build/cooperative-direct-output-cub-test \
  --target cub.test.device.histogram -j 8

build/cooperative-direct-output-cub-test/bin/cub.test.device.histogram.lid_0 \
  '[histogram][device]' --reporter compact
build/cooperative-direct-output-cub-test/bin/cub.test.device.histogram.lid_2 \
  '[histogram][device]' --reporter compact

/home/shadeform/.local/cmake-venv/bin/cmake \
  --build build/cooperative-direct-output-cub-test \
  --target cub.test.device.histogram_api -j 8
build/cooperative-direct-output-cub-test/bin/cub.test.device.histogram_api.lid_0 \
  --reporter compact
```

Results: 107,824 assertions passed across 58 core device-histogram test cases;
all 8 API cases passed.

An attempted build that also included `cub.test.device.histogram_env` failed on
two pre-existing raw-branch API-drift issues unrelated to these kernels:

- the env test calls an obsolete `output_atomic_spill::spill(output, bin, value)` signature;
- device-side launch-mode compilation reaches existing host-only debug `getenv`/`fprintf` hooks.

Targeted pre-commit passed all hooks for every changed source and script file.

## Durable artifacts

- `../cooperative-direct-output-smem-c32/algo_sweep_full.json`: merged c32 data.
- `../cooperative-direct-output-smem-c32-resume/multi_full.json`: c32 multi-channel recovery data.
- `../cooperative-direct-output-smem-u64/algo_sweep_full.json`: complete u64 data.
- `measurements.csv`: all 44,400 measurements and ratios.
- `geometric_means.csv`: aggregate geometric means.
- `outliers.csv`: ranked per-family exceptions.
- `analysis_summary.md`: generated concise tables.
- `artifact_validation.json`: exact expected/actual cell counts.
- `figures/`: throughput and speedup graphs for c32/u64 and all four families.
- `logs/`: complete c32, c32 recovery, and u64 sweep logs.
- `checksums.sha256`: SHA-256 checksums for raw and derived artifacts.
