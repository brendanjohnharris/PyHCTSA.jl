#! /bin/bash
# -*- mode: julia -*-
#=
exec julia -t auto --startup-file=no --color=yes "${BASH_SOURCE[0]}" "$@"
=#
using Pkg
Pkg.activate(@__DIR__)
using PyHCTSA
using MoreMaps

begin
    ops = PyHCTSA.build_ops()
    pops = PyHCTSA.calculator.FeatureCalculator()

    x = PyHCTSA.testdata(:test)
    ops(x; chart = Chart())
    @profview ops(x; chart = Chart())

    Profile.clear()
    @profile ops(x; chart = Chart())
    Profile.print()
end
