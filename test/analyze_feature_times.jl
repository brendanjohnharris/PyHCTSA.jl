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

begin # * Load
    csvfile = joinpath(@__DIR__, "feature_times.csv")
    raw, header = readdlm(csvfile, ',', Any; header = true)   # Any: col 1 is names, rest floats
    names = Symbol.(raw[:, 1])                           # feature names (row order)
    lengths = parse.(Int, vec(header)[2:end])            # series lengths (column order)
    T = Float64.(raw[:, 2:end])                          # times (s), features x lengths
    times = FeatureArray(T, names, :length => lengths)   # Feat x :length, getnames(times) = names
    features = times                                     # getnames(features) returns feature names
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

begin # * Calculate how much time would be saved by dropping the slowest 5% of features
    frac = 0.1
    nfeat = length(tlong)
    ndrop = ceil(Int, frac * nfeat)

    order = sortperm(tlong; rev = true)                  # slowest first (NaN sorts last)
    dropped = order[1:ndrop]
    saved = sum(t -> isfinite(t) ? t : 0.0, tlong[dropped])
    pct = 100 * saved / total_long

    @info "Dropping the slowest $frac of features ($ndrop of $nfeat) at longest N = $(lookup(times, 2)[end])"
    println(
        "  saves ", round(pct; digits = 1), "% of total runtime",
        " (", round(saved; digits = 2), " s of ", round(total_long; digits = 2), " s)"
    )
end

begin # * Save the names for a hctsa_90 feature set containing only the fastest 90% of features
    frac = 0.1
    nfeat = length(tlong)
    ndrop = ceil(Int, frac * nfeat)

    order = sortperm(tlong; rev = true)                  # slowest first (NaN sorts last)
    keep = order[(ndrop + 1):end]                      # keep the fastest 90%
    kept_names = getnames(features)[keep]

    outfile = joinpath(@__DIR__, "..", "assets", "hctsa90.txt")   # packaged as an asset
    open(outfile, "w") do io
        foreach(name -> println(io, name), kept_names)
    end
    @info "Wrote $(length(kept_names)) feature names to $outfile"
end
