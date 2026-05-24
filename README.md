# HCTSA.jl
[![Build Status](https://github.com/brendanjohnharris/HCTSA.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/brendanjohnharris/HCTSA.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/brendanjohnharris/HCTSA.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/brendanjohnharris/HCTSA.jl)
[![code style: runic](https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black)](https://github.com/fredrikekre/Runic.jl)

HCTSA.jl brings the [_hctsa_](https://github.com/benfulcher/hctsa) feature library into Julia through [pyhctsa](https://github.com/dynamicsandneuralsystems/pyhctsa) and [TimeseriesFeatures.jl](https://github.com/brendanjohnharris/TimeseriesFeatures.jl).

## Installation
```julia
using Pkg
Pkg.add("HCTSA")
using HCTSA
```
To enable java-dependent features on Mac and Linux, set the environment variable `JULIA_COPY_STACKS=1` before launching Julia. DO NOT do this on Windows, as it will cause crashes.

## Quick Start
```julia
using HCTSA

# Single time series
x = randn(200)
y = hctsa(x)

# Multiple time series in columns
X = randn(200, 8)
Y = hctsa(X)

# Or select features
ops = hctsa[[:ac_1, :ac_2]]
A = ac(X)
```


## Advanced usage

### Parallel computation
HCTSA uses [MoreMaps.jl](https://github.com/brendanjohnharris/MoreMaps.jl) charts for parallel execution. `MoreMaps.QualityLogger()` is useful for tracking the proportion of good features across time series.

> [!WARNING]
> Threaded execution is currently unavailable due to Python's global interpreter lock (GIL). Parallel execution works best through the `Pmap()` backend.

```julia
using HCTSA
using MoreMaps
using Distributed

# Load required packages on all workers
addprocs(4)
@everywhere using HCTSA, MoreMaps

X = randn(100, 20)
Y = hctsa(X; chart = Chart(Pmap(), QualityLogger())) # Takes a few seconds each iter.
```


### Debug logging
Enable HCTSA debug logs in the current Julia session with `ENV["JULIA_DEBUG"] = "HCTSA"`; this will show errors in feature computation.
To view warnings from pyhctsa, set `ENV["PYTHON_LOG_LEVEL"] = "1"` before launching Julia (or set the `python_log_level` preference).



## Core API
- `HCTSA.load_config(path=nothing)`: Load pyhctsa YAML config.
- `HCTSA.build_mops(...)`: Build master operations from config.
- `HCTSA.cache_mopops(mops, path)`: Cache mop output-key structure.
- `HCTSA.build_ops(mops, mopops)`: Build scalar operation set.
- `HCTSA.testdata(name)`: Load packaged test datasets.


## Notes
- Some features can differ from pyhctsa due to numerical differences Base.between Julia and Python (especially features that take logarithms of numbers near zero).
- Features that depend on jpype1 are filtered unless JULIA_COPY_STACKS=1.

