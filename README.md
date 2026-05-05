# HCTSA.jl
[![Build Status](https://github.com/brendanjohnharris/HCTSA.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/brendanjohnharris/HCTSA.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/brendanjohnharris/HCTSA.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/brendanjohnharris/HCTSA.jl)

HCTSA.jl brings the [_hctsa_](https://github.com/benfulcher/hctsa) feature library into Julia through [pyhctsa](https://github.com/dynamicsandneuralsystems/pyhctsa) and [TimeseriesFeatures.jl](https://github.com/brendanjohnharris/TimeseriesFeatures.jl).

## Installation
```julia
using Pkg
Pkg.add("HCTSA")
using HCTSA
```
To enable java-dependent features, set the environment variable `JULIA_COPY_STACKS=1` before launching Julia.

## Quick Start
```julia
using HCTSA

# Build operation set from pyhctsa config
ops = HCTSA.build_ops()

# Single time series
x = randn(200)
y = ops(x)

# Multiple time series in columns
X = randn(200, 8)
Y = ops(X)
```


## Advanced usage

### Parallel computation
HCTSA uses [MoreMaps.jl](https://github.com/brendanjohnharris/MoreMaps.jl) charts for parallel execution.

> [!WARNING]
> Threaded execution is currently unavailable due to pyhctsa's use of Python's global interpreter lock (GIL). Parallel execution works best through the `Pmap()` backend.

```julia
using HCTSA
using MoreMaps
using Distributed

# Start workers, then load packages on all workers.
addprocs(4)
@everywhere using HCTSA, MoreMaps, UNicodePlots

ops = HCTSA.build_ops()
X = randn(2000, 16)

# Parallel map across time series using a pmap chart.
Y = ops(X; chart = Chart(Pmap()))
```

### Debug logging
Enable HCTSA debug logs in the current Julia session with `ENV["JULIA_DEBUG"] = "HCTSA"`


## Core API
- `HCTSA.load_config(path=nothing)`: Load pyhctsa YAML config.
- `HCTSA.build_mops(...)`: Build master operations from config.
- `HCTSA.cache_mopops(mops, path)`: Cache mop output-key structure.
- `HCTSA.build_ops(mops, mopops)`: Build scalar operation set.
- `HCTSA.testdata(name)`: Load packaged test datasets.


## Notes
- Some features can differ from pyhctsa due to numerical differences Base.between Julia and Python (especially features that take logarithms of numbers near zero).
- Features that depend on jpype1 are filtered unless JULIA_COPY_STACKS=1.

