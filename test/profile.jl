#! /bin/bash
# -*- mode: julia -*-
#=
exec julia -t auto --startup-file=no --color=yes "${BASH_SOURCE[0]}" "$@"
=#
using Pkg
Pkg.activate(@__DIR__)
using HCTSA
using MoreMaps

begin
    ops = HCTSA.build_ops()
    pops = HCTSA.calculator.FeatureCalculator()

    x = HCTSA.testdata(:test)
    ops(x; chart = Chart())
    @profview ops(x; chart = Chart())

    Profile.clear()
    @profile ops(x; chart = Chart())
    Profile.print()
end
