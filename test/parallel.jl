#! /bin/bash
# -*- mode: julia -*-
#=
exec julia -t auto --startup-file=no --color=yes "${BASH_SOURCE[0]}" "$@"
=#
using Pkg
Pkg.activate(@__DIR__)
using HCTSA
using Distributed
using MoreMaps

begin
    addprocs(8)
    @everywhere using HCTSA, MoreMaps
end
begin
    ops = HCTSA.build_ops()
    pops = HCTSA.calculator.FeatureCalculator()

    X = randn(100, 8)
    a = @timed ops(X; chart = Chart(Pmap()))
    b = @timed ops(X; chart = Chart())

    @assert filter(!isnan, a.value) == filter(!isnan, b.value)
    @assert a.time < b.time
end
