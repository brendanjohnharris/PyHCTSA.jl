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
end
begin
    # * Create/clear log file
    logpath = joinpath(@__DIR__, "crash_test.log")
    open(logpath, "w") do io
        write(io, "")
    end

    # * Test feature by feature for crash
    for i in 1:length(ops)
        @info "Testing feature $i / $(length(ops))"
        # Write status to log file in case of crash
        f = ops[[i]]
        open(logpath, "a") do io
            write(io, "Testing feature $i / $(length(ops)) ($(only(getnames(f))))\n")
        end
        f(x)
        open(logpath, "a") do io
            write(io, "Feature $i / $(length(ops)) passed!\n")
        end
    end
end
