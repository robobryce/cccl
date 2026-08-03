# Cooperative direct-output histogram analysis

Ratios are geometric means. Small workloads contain 1 Mi and 16 Mi elements; large workloads contain 256 Mi, 1,073,741,824, and 2,000,000,000 elements. The 64 Mi size is retained in the raw CSV but excluded from these two buckets.

## Artifact validation

- `c32`: PASS
- `u64`: PASS

## Geometric-mean speedups

### c32

| Family | Size | CST/SST | CDY/SDY | SST/main | CST/main | SDY/main | CDY/main |
|---|---:|---:|---:|---:|---:|---:|---:|
| even | small | 0.965× | 0.964× | 1.678× | 1.619× | 1.699× | 1.637× |
| even | large | 0.999× | 1.000× | 2.479× | 2.477× | 2.458× | 2.457× |
| range | small | 0.969× | 0.956× | 1.100× | 1.066× | 1.120× | 1.071× |
| range | large | 1.005× | 1.000× | 1.081× | 1.086× | 1.168× | 1.168× |
| multi-even | small | 0.981× | 0.984× | 1.782× | 1.749× | 1.592× | 1.566× |
| multi-even | large | 1.000× | 1.000× | 2.498× | 2.497× | 1.958× | 1.958× |
| multi-range | small | 0.908× | 0.986× | 1.022× | 0.928× | 1.015× | 1.001× |
| multi-range | large | 0.998× | 0.999× | 1.056× | 1.055× | 1.179× | 1.178× |

### u64

| Family | Size | CST/SST | CDY/SDY | SST/main | CST/main | SDY/main | CDY/main |
|---|---:|---:|---:|---:|---:|---:|---:|
| even | small | 0.965× | 0.964× | 1.678× | 1.619× | 1.701× | 1.639× |
| even | large | 1.000× | 1.000× | 2.474× | 2.473× | 2.455× | 2.454× |
| range | small | 0.966× | 0.962× | 1.102× | 1.065× | 1.117× | 1.075× |
| range | large | 1.006× | 1.001× | 1.083× | 1.090× | 1.165× | 1.166× |
| multi-even | small | 0.981× | 0.985× | 1.779× | 1.746× | 1.589× | 1.565× |
| multi-even | large | 1.000× | 1.000× | 2.509× | 2.509× | 1.963× | 1.963× |
| multi-range | small | 0.907× | 0.984× | 1.024× | 0.929× | 1.017× | 1.000× |
| multi-range | large | 1.002× | 1.000× | 1.059× | 1.061× | 1.175× | 1.175× |
