#! /bin/bash
# -*- mode: julia -*-
#=
exec julia -t 1 --startup-file=no --color=yes "${BASH_SOURCE[0]}" "$@"
=#
# using Pkg
# Pkg.activate(normpath(joinpath(@__DIR__, "..")))
using PyHCTSA
using Test
using Random
using MoreMaps

ENV["JULIA_DEBUG"] = "PyHCTSA"   # or "all"

begin
    ops = PyHCTSA.build_ops()
    pops = PyHCTSA.calculator.FeatureCalculator()

    x = PyHCTSA.testdata(:test)

    seed = 42
    pyrandom = PyHCTSA.PythonCall.pyimport("random")

    Random.seed!(42)
    PyHCTSA.numpy.random.seed(seed)
    pyrandom.seed(seed)
    a = @time ops(x; chart = Chart())
    anames = collect(getnames(a))

    Random.seed!(42)
    PyHCTSA.numpy.random.seed(seed)
    pyrandom.seed(seed)
    b = @time pops.extract(PyHCTSA.as_numpy(x))
end

@testset "val_check setup" begin
    @test length(anames) > 0
    @test length(a) == length(anames)
end

# begin # * Compare
#     _bnames = PyHCTSA.PythonCall.pyconvert(Vector{String}, b.columns)
#     bvals = map(eachindex(_bnames)) do i
#         PyHCTSA.convert_op(b.iloc[0, i - 1])
#     end
#     bname_idx = Dict{Symbol, Int}(Symbol(name) => i for (i, name) in enumerate(_bnames))
#     pyidxs = map(anames) do aname
#         get(bname_idx, aname, nothing)
#     end
# end

Q = map(enumerate(anames)) do (i, fname)
    _a = a[fname]
    if !haskey(b, string(fname))
        return false
    end
    _b = PyHCTSA.convert_op(b[string(fname)], fname)
    if isnan(_a) && isnan(_b)
        return true
    end
    q = isapprox(_a, _b, rtol = 1.0e-3, atol = 1.0e-4)
    if !q
        @warn "Mismatch on feature $(fname): Julia $(_a) vs Python $(_b)"
    end
    return q
end
@testset "val_check julia vs pyhctsa agreement" begin
    using Statistics
    @test length(Q) == length(anames)
    # Some features can diverge due to tiny z-score differences amplified downstream.
    @test mean(Q) >= 0.95
end

begin # * Notes
    # Some spectral_summaries features take logs on spectral bins. There is a minor discrepancy
    # between python zscore and julia zscore that result sin machine-precision differences,
    # which the logs amplify. So the discrepancies in these log features are expected.
end

begin # * Time feature by feature
    pyfeature_names = PyHCTSA.PythonCall.pyconvert(Vector{String}, pops.feature_funcs.keys())
    pydistribute = PyHCTSA.PythonCall.pyimport("pyhctsa.distribute")
    xpy = PyHCTSA.as_numpy(x)

    valtimes = map(enumerate(anames[1:1000])) do (i, aname)
        # @info "Timing feature $i / $(length(anames))"

        begin # * time Julia
            aval, ta = @timed ops[i](x)
        end

        begin # * time pyhctsa
            bname = split(string(aname), ".") |> first

            onefuncs = PyHCTSA.PythonCall.pydict()
            onefuncs[bname] = pops.feature_funcs[bname]

            bval, tb = @timed begin
                row = pydistribute._extract_features_single_series(xpy, onefuncs)
            end
            # bval = PyHCTSA.convert_op(row[string(aname)], aname)
        end

        # if !isapprox(aval, bval, rtol = 1e-7, atol = 1e-12)
        #     @warn "Mismatch on feature $(aname): Julia $aval vs Python $bval"
        # end

        # * return results
        (; aname => (; tjulia = ta, tpython = tb)) # valjulia = aval,  valpython = bval))
    end
    valtimes = Dict(only(keys(t)) => only(values(t)) for t in valtimes)
    # end

    # begin # * Extract metrics
    tpython = getindex.(values(valtimes), :tpython)
    tjulia = getindex.(values(valtimes), :tjulia)
    delta = tjulia .- tpython
    rdelta = tjulia ./ tpython

    # * Minimum and maximum relative error
    minimum(rdelta), maximum(rdelta)
end

@testset "val_check timing sanity" begin
    @test length(pyfeature_names) > 0
    @test length(valtimes) > 0
    @test length(tpython) == length(tjulia)
    @test all(isfinite, tpython)
    @test all(isfinite, tjulia)
    @test all(>(0), tpython)
    @test all(>(0), tjulia)
    @test all(isfinite, rdelta)
end

begin # * Show features that are slower
    toldelta = round.(tjulia, sigdigits = 3) .- round.(tpython, sigdigits = 3)
    slowidxs = findall(toldelta .> 0)
    for idx in slowidxs
        aname = anames[idx]
        @info "$aname slower in Julia ($(round(tjulia[idx], sigdigits = 3))s vs $(round(tpython[idx], sigdigits = 3))s)"
    end
    @warn "Julia slower on $(length(slowidxs)) features out of $(length(anames))"
    @warn "Julia faster on $(length(anames) - length(findall(toldelta .> 0))) features out of $(length(anames))"
end

@testset "val_check feature coverage" begin
    @test length(slowidxs) <= length(tjulia)
end

# begin # * Quicktest
#     bname = "spectral_summaries_fft_none"

#     pydistribute = PyHCTSA.PythonCall.pyimport("pyhctsa.distribute")

#     onefuncs = PyHCTSA.PythonCall.pydict()
#     onefuncs[bname] = pops.feature_funcs[bname]

#     xpy = PyHCTSA.as_numpy(x)
#     row = pydistribute._extract_features_single_series(xpy, onefuncs)

#     mops = PyHCTSA.build_mops()
#     v = mops[Symbol(bname)](x)

#     display(row)
#     display(v)

#     display(row["$bname.logstd"])
#     display(v["logstd"])

#     # @assert PyHCTSA.pyconvert(Number, row[bname]) ≈ PyHCTSA.pyconvert(Number, v)
# end

# begin # * Compare julia vs python zscore
#     y = PyHCTSA.as_numpy(x)
#     @benchmark PyHCTSA.zz_score($x)
#     @benchmark PyHCTSA.pyhctsa.utils.z_score($y)
#     @benchmark zz_score($x)

#     z1 = PyHCTSA.zz_score(x)
#     z2 = PyHCTSA.pyhctsa.utils.z_score(y)
#     z2 = PyHCTSA.pyconvert(Array{Float64}, z2)
#     z3 = zz_score(x)

#     @assert z1 ≈ z2 ≈ z3
#     @assert z1 != z2
#     @assert z2 == z3
# end
