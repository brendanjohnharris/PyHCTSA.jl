#! /bin/bash
# -*- mode: julia -*-
#=
exec julia -t auto --color=no --startup-file=no "${BASH_SOURCE[0]}" "$@"
=#
using Pkg
Pkg.activate(@__DIR__)
using HCTSA
using BenchmarkTools

hctsa = HCTSA.build_ops()
pyhctsa = HCTSA.calculator.FeatureCalculator()
x = HCTSA.testdata(:test)
y = HCTSA.as_numpy(x)

@info "Timing Julia implementation..."
a = @benchmark $hctsa($x) samples=10 seconds=120 evals=1

@info "Timing Python implementation..."
b = @benchmark $pyhctsa.extract($y) samples=10 seconds=120 evals=1

@info "All tests complete."
