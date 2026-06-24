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

begin # * Fit a model to each feature's time curve. Power-law model
    ℓ = log10.(collect(lookup(times, 2)))                      # predictor: log series length
    names = getnames(features)
    curves = collect(eachslice(times; dims = :Feat))  # one time curve per feature

    fits = map(names, curves) do name, t
        y = log10.(collect(t))
        keep = isfinite.(y)                           # drop failed (NaN) features
        sum(keep) < 2 && return (; name, a = NaN, b = NaN, r2 = NaN)
        A = [ones(sum(keep)) ℓ[keep]]                 # design matrix [1  logN]
        c = A \ y[keep]                               # least squares
        ŷ = A * c
        r2 = 1 - sum(abs2, y[keep] .- ŷ) / sum(abs2, y[keep] .- mean(y[keep]))
        (; name, a = 10^c[1], b = c[2], r2)
    end

    b = DimArray(getindex.(fits, :b), Feat(names))    # complexity exponent per feature
end

begin # * Report features with the worst time complexity and the worst 'intercept'
    topn = 10

    # Share of total runtime at the LONGEST series length: this is the time you would
    # actually save (as a % of all features' summed time at that length) by dropping a
    # feature. Uses the measured times directly, not the fitted power law.
    tlong = collect(times[:, end])                       # measured time at longest N, per feature
    total_long = sum(t -> isfinite(t) ? t : 0.0, tlong)
    longtime = Dict(getnames(features) .=> tlong)         # name -> seconds at longest N
    pctsaved = Dict(getnames(features) .=> 100 .* tlong ./ total_long)

    goodfit = filter(f -> f.r2 > 0.5, fits)             # only features with a good power-law fit
    worst_complexity = sort(goodfit; by = f -> f.b, rev = true)[1:min(topn, end)]
    worst_intercept = sort(goodfit; by = f -> f.a, rev = true)[1:min(topn, end)]
    worst_cost = sort(fits; by = f -> pctsaved[f.name], rev = true)[1:min(topn, end)]

    @info "Worst time complexity (largest exponent b = dlog t / dlog N)"
    foreach(worst_complexity) do f
        println(
            rpad(f.name, 40), " b=", round(f.b; digits = 2),
            "   (a=", round(f.a; sigdigits = 2), ", r²=", round(f.r2; digits = 2),

        )
    end

    @info "Worst intercept (largest prefactor a = baseline cost)"
    foreach(worst_intercept) do f
        println(
            rpad(f.name, 40), " a=", round(f.a; sigdigits = 2),
            "   (b=", round(f.b; digits = 2), ", r²=", round(f.r2; digits = 2),

        )
    end

    @info "Biggest time saving if removed (% of total runtime at longest N = $(lookup(times, 2)[end]))"
    foreach(worst_cost) do f
        println(
            rpad(f.name, 40), " saves ", round(pctsaved[f.name]; digits = 2), "%",
            "   (", round(1.0e3 * longtime[f.name]; digits = 1), " ms",
            ", b=", round(f.b; digits = 2), ")"
        )
    end
end
