#! /bin/bash
# -*- mode: julia -*-
#=
exec julia -t auto --startup-file=no --color=yes "${BASH_SOURCE[0]}" "$@"
=#
using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")))
using HCTSA
using TimeseriesFeatures
using Printf

mops = HCTSA.build_mops()
x = randn(10000)

# Warm up the Python side so JIT/import overhead doesn't pollute the first mop's time.
@info "Warming up..."
first(mops)(x)

n = length(mops)
names = Vector{String}(undef, n)
times = Vector{Float64}(undef, n)

@info "Timing $n mop$(n == 1 ? "" : "s")..."
for (i, mop) in enumerate(collect(mops))
    nm = String(getname(mop))
    names[i] = nm
    t0 = time_ns()
    try
        mop(x)
    catch e
        @debug "Error running $nm" exception = (e, catch_backtrace())
    end
    times[i] = (time_ns() - t0) / 1.0e9
    @printf("[%4d/%d] %-60s %.4fs\n", i, n, nm, times[i])
end

# Sort descending by time
order = sortperm(times; rev = true)
names_sorted = names[order]
times_sorted = times[order]

outpath = joinpath(@__DIR__, "mop_times.csv")
open(outpath, "w") do io
    println(io, "name,time_seconds")
    for (nm, t) in zip(names_sorted, times_sorted)
        println(io, "$(nm),$(t)")
    end
end

@info "Wrote $outpath"
