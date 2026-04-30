#! /bin/bash
# -*- mode: julia -*-
#=
exec julia -t auto --color=no --startup-file=no "${BASH_SOURCE[0]}" "$@"
=#
using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")))
using HCTSA
using Statistics
ENV["JULIA_DEBUG"] = "HCTSA"   # or "all"

hctsa = HCTSA.build_ops()

for test in HCTSA.testnames[[1, 6]]
    @info "-------------------------------------------------------------------------------"
    @info "Testset: $test"
    data = HCTSA.testdata(test)
    out = hctsa(data)
    Q = mean(!isnan, out)
    @info "Quality of $test: $Q"
end

@info "All tests complete."
