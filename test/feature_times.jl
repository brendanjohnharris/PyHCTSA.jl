#! /bin/bash
# -*- mode: julia -*-
#=
exec julia -t auto --startup-file=no --color=yes "${BASH_SOURCE[0]}" "$@"
=#
using PyHCTSA
using TimeseriesBase
using MoreMaps
using Statistics
using PythonCall
using DelimitedFiles

begin # * Parameters
    lengths = logrange(50, 50000, length = 100) .|> round .|> Int |> reverse |> Dim{:length}
    X = map(cumsum ∘ randn, lengths)
    features = PyHCTSA.build_mops()
    features(X[end], Union{Py, PyException}) # warm up (precompile)
end

begin # * Time
    times = map(Chart(Iterators.product, QualityLogger(; nlogs = 1000), Sequential()), Feat(getnames(features)), X) do name, x
        try
            f = features[name]
            t = @timed f(x)
            return t.time
        catch e
            @warn "Error computing feature $name for length $(length(x))" exception = (e, catch_backtrace())
            return NaN
        end
    end
    times = reverse(times, dims = 2)
end

begin # * Save timings to CSV
    # Wide table: one row per feature, one column per series length.
    # First column is the feature name, header row is the length values.
    outfile = joinpath(@__DIR__, "feature_times.csv")
    header = reshape(["feature"; string.(collect(lookup(times, 2)))], 1, :)
    table = hcat(string.(getnames(features)), Matrix(times))   # features x lengths, in Feat order
    open(outfile, "w") do io
        writedlm(io, header, ',')
        writedlm(io, table, ',')
    end
    @info "Wrote $(size(table, 1)) x $(length(lookup(times, 2))) timing table to $outfile"
end
