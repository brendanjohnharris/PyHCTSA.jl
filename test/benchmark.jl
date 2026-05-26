#! /bin/bash
# -*- mode: julia -*-
#=
exec julia -t auto --color=no --startup-file=no "${BASH_SOURCE[0]}" "$@"
=#
using Pkg
Pkg.activate(@__DIR__)
using PyHCTSA
using BenchmarkTools

hctsa = PyHCTSA.build_ops()
pyhctsa = PyHCTSA.calculator.FeatureCalculator()
x = PyHCTSA.testdata(:test)
y = PyHCTSA.as_numpy(x)

@info "Timing Julia implementation..."
a = @benchmark $hctsa($x) samples = 10 seconds = 120 evals = 1

@info "Timing Python implementation..."
b = @benchmark $pyhctsa.extract($y) samples = 10 seconds = 120 evals = 1

@info "All tests complete."
