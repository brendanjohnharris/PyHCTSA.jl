#! /bin/bash
# -*- mode: julia -*-
#=
exec julia -t auto --startup-file=no --color=yes "${BASH_SOURCE[0]}" "$@"
=#
using Pkg
Pkg.activate(@__DIR__)
using PyHCTSA
using Distributed
using MoreMaps

begin
    addprocs(8)
    @everywhere using PyHCTSA, MoreMaps
end
begin
    ops = PyHCTSA.build_ops()
    pops = PyHCTSA.calculator.FeatureCalculator()

    X = randn(100, 8)
    a = @timed ops(X; chart = Chart(Pmap()))
    b = @timed ops(X; chart = Chart())

    @assert filter(!isnan, a.value) == filter(!isnan, b.value)
    @assert a.time < b.time
end
