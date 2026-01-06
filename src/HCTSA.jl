module HCTSA
using DimensionalData
using Reexport
using TimeseriesFeatures
using LinearAlgebra
using JSON
using PythonCall
import PythonCall: pynew, pycopy!
import Statistics: mean, std, cov

const pyhctsa = pynew()
const pyOperations = pynew()
const calculator = pynew()
const featurebindings = pynew()
const utils = pynew()
const jpype = pynew()
const yaml = pynew()
const numpy = pynew()
const pkgutil = pynew()
const numbers = pynew()

function __init__()
    pycopy!(pyhctsa, pyimport("pyhctsa"))
    pycopy!(pyOperations, pyimport("pyhctsa.Operations"))
    pycopy!(calculator, pyimport("pyhctsa.FeatureCalculator.calculator"))
    pycopy!(utils, pyimport("pyhctsa.Utilities.utils"))
    pycopy!(jpype, pyimport("jpype"))
    pycopy!(yaml, pyimport("yaml"))
    pycopy!(numpy, pyimport("numpy"))
    pycopy!(numbers, pyimport("numbers"))

    # * Load all Operation submodules
    pycopy!(pkgutil, pyimport("pkgutil"))
    for modinfo in pkgutil.iter_modules(pyOperations.__path__)
        pyimport("pyhctsa.Operations.$(modinfo.name)")
    end
end

@reexport using TimeseriesFeatures

function py2dict(pd::Py)
    if pyisinstance(pd, pybuiltins.dict)
        return Dict(PythonCall.pyconvert(String, k) => py2dict(v)
                    for (k, v) in pd.items())
    elseif pyisinstance(pd, pybuiltins.list)
        return [py2dict(item) for item in pd]
    else
        # Base types: let PythonCall auto-convert
        return pyconvert(Any, pd)
    end
end
py2dict(pd::PyDict) = py2dict(Py(pd))
py2dict(x) = x

function description(mop::Dict)
    mop = filter(p -> p.first ∈ ("hctsa_name", "dependencies", "base_name"), mop)
    description = JSON.json(mop)
end
keywords(mop::Dict) = mop["labels"]
function normalize_config(x::AbstractVector)
    pylist(x)
end
normalize_config(x) = identity(x)
function mop_func(module_name::String, func_name::String, config::Dict)
    config = filter(p -> p.first ∉ ("zscore", "abs"), config)
    config = [Symbol(k) => normalize_config(v)
              for (k, v) in config if k ∉ ("zscore", "abs")]
    op_module = getproperty(HCTSA.pyOperations, module_name)
    op_func = getproperty(op_module, func_name)
    function f(x)
        try
            return op_func(x; config...)
        catch e
            @warn "Error computing feature $module_name:$func_name" exception=(e,
                                                                               catch_backtrace())
            return e
        end
    end
end
function name(mop::Dict)
    if haskey(mop, "base_name")
        return mop["base_name"] |> String
    else
        throw(ArgumentError("mop missing 'base_name' key: $mop"))
    end
end
iszscore(config::Dict) = haskey(config, "zscore") && config["zscore"] == true
isabs(config::Dict) = haskey(config, "abs") && config["abs"] == true
iszscore(x) = false
isabs(x) = false

"""
    flatten_config(config::Dict)

Takes a config dict where values may be lists, and returns an iterator of dicts
with all combinations (Cartesian product). Scalar values are treated as single-element lists.

Example:
    config = Dict("d" => [1.0, 0.5], "D" => [3, 5], "zscore" => true)
    # yields 4 dicts:
    # Dict("d" => 1.0, "D" => 3, "zscore" => true)
    # Dict("d" => 1.0, "D" => 5, "zscore" => true)
    # Dict("d" => 0.5, "D" => 3, "zscore" => true)
    # Dict("d" => 0.5, "D" => 5, "zscore" => true)
"""
function flatten_config(config::Dict)
    keys = collect(Base.keys(config))
    # Wrap scalars in a vector, keep vectors as-is
    values = [v isa Vector ? v : [v] for v in [config[k] for k in keys]]
    # Cartesian product of all value combinations
    (Dict(zip(keys, combo)) for combo in Iterators.product(values...))
end

"""
    format_param_value(val)

Format a parameter value for use in feature labels, matching pyhctsa's _format_param_value.
- Negative numbers get 'm' prefix (e.g., -1 -> "m1")
- Floats between 0-1 use '0p' format (e.g., 0.5 -> "0p5")
- Other floats use 'p' for decimal point (e.g., 1.5 -> "1p5")
- Lists show as "start_end" if contiguous range, else joined with "_"
"""
function format_param_value(val::Number)
    if val < 0
        return "m" * format_param_value(-val)
    elseif val == floor(val)
        return string(Int(val))
    elseif 0 < val < 1
        # "0p" + digits after decimal, strip trailing zeros
        return "0p" * rstrip(split(string(val), ".")[2], '0')
    else
        # Replace '.' with 'p', strip trailing zeros and 'p'
        s = replace(string(val), "." => "p")
        return rstrip(rstrip(s, '0'), 'p')
    end
end

function format_param_value(val::AbstractVector)
    # Check if contiguous integer range
    if length(val) > 1 && all(x -> x isa Number, val)
        diffs = diff(val)
        if all(==(1), diffs)
            return format_param_value(first(val)) * "_" * format_param_value(last(val))
        end
    end
    return join(format_param_value.(val), "_")
end

format_param_value(val) = string(val)

"""
    name_config(config::Dict, ordered_args=nothing)

Generate the parameter suffix for a feature name, matching pyhctsa's naming convention.
If ordered_args is provided, parameters are ordered accordingly.
Excludes "zscore" and "abs" keys from the name.
"""
function name_config(config::Dict, mopconfig::Dict)
    # Filter out zscore and abs
    params = filter(p -> p.first ∉ ("zscore", "abs"), config)
    isempty(params) && return ""

    if haskey(mopconfig, "ordered_args") && !isnothing(mopconfig["ordered_args"]) &&
       !isempty(mopconfig["ordered_args"])
        # Use ordered_args to determine parameter order
        parts = [format_param_value(params[arg])
                 for arg in mopconfig["ordered_args"] if haskey(params, arg)]
        if isempty(parts) && !isempty(mopconfig["ordered_args"])
            @warn "ordered_args references keys not in config" ordered_args=mopconfig["ordered_args"] params
        end
    else
        # Default: "key{value}" format for each param
        parts = ["$(k)$(format_param_value(v))" for (k, v) in params]
    end

    return "_" * join(parts, "_")
end

as_numpy(x) = Py(collect(x)).to_numpy(; copy = false)

# * Preprocessing features
using Statistics
z_score(𝐱::AbstractVector) = (𝐱 .- mean(𝐱)) ./ (std(𝐱))
zz_score(𝐱::AbstractVector) = (z_score ∘ z_score)(𝐱)
const numpy_identity = Feature(as_numpy ∘ Identity, :numpy_identity,
                               "𝐱 → 𝐱", ["normalization"])
const numpy_zscore = Feature(as_numpy ∘ zz_score, :numpy_zscore,
                             "𝐱 → (𝐱 - μ(𝐱))/σ(𝐱)",
                             ["normalization"])
const numpy_abs = Feature(as_numpy ∘ Base.BroadcastFunction(abs), :abs, "𝐱 → |𝐱|",
                          ["normalization"])
const numpy_abs_zscore = Feature(as_numpy ∘ Base.BroadcastFunction(abs) ∘
                                 zz_score,
                                 :numpy_abs_zscore,
                                 "𝐱 → |(𝐱 - μ(𝐱))/σ(𝐱)|",
                                 ["normalization"])

"Builds the set of mops defined by an entry in the yaml file. Note that each entry can specify more than one mop, based on the length of the config entry"
function build_mops(module_name::String, func_name::String, mopconfig::Dict)
    configs = mopconfig["configs"]
    configs = map(flatten_config, configs) |> Iterators.flatten
    features = map(configs) do config
        if iszscore(config) && isabs(config)
            super = numpy_abs_zscore
        elseif iszscore(config)
            super = numpy_zscore
        elseif isabs(config)
            super = numpy_abs
        else
            super = numpy_identity
        end
        pop!(config, "zscore", nothing)
        pop!(config, "abs", nothing)
        fullname = name(mopconfig) * name_config(config, mopconfig)
        SuperFeature(mop_func(module_name, func_name, config), Symbol(fullname),
                     description(mopconfig),
                     keywords(mopconfig), Feature(super))
    end
    return features |> SuperFeatureSet
end

function build_mops(module_name::String, mopconfigs::Dict)
    map(collect(mopconfigs)) do (func_name, mopconfig)
        build_mops(module_name, func_name, mopconfig)
    end |> Iterators.flatten |> collect |> SuperFeatureSet
end

function build_mops(mopconfigs::Dict)
    map(collect(mopconfigs)) do (module_name, module_configs)
        build_mops(module_name, module_configs)
    end |> Iterators.flatten |> collect |> SuperFeatureSet
end
function load_config(path = nothing)
    if isnothing(path)
        # pkg_dir = pyconvert(String, HCTSA.pyhctsa.__path__[0])  # root of pyhctsa package
        # path = joinpath(pkg_dir, "Configurations", "hctsa.yaml")
        path = joinpath(@__DIR__, "hctsa.yaml")
    end
    config = HCTSA.yaml.safe_load(read(path, String)) |> HCTSA.py2dict
    return config
end
function build_mops(module_name::String, args...)
    config = load_config(args...)
    build_mops(module_name, config[module_name])
end
function build_mops()
    config = load_config()
    build_mops(config)
end

function convert_op(mopval)
    if pyisinstance(mopval, HCTSA.numpy.ndarray)
        return pyconvert(Float64, mopval |> only)
    elseif pyisinstance(mopval, pybuiltins.Exception)
        @warn exception = (mopval, catch_backtrace())
        return NaN
    else
        return pyconvert(Float64, mopval)
    end
end
convert_op(x::Number) = convert(Float64, x)
function get_op(mopval, opname, mop)
    if pyisinstance(mopval, pybuiltins.Exception)
        @warn exception = (y, catch_backtrace())
        return NaN
    elseif pyisinstance(mopval, numbers.Number) || pyisinstance(mopval, numpy.ndarray)
        @warn "$(getname(mop)) output $mopval is not subscriptable; maybe it errored and returned NaN?"
        return NaN
    elseif haskey(mopval, opname)
        return convert_op(mopval[opname])
    else
        @warn "$(getname(mop)) output has no key $opname in: $mopval"
        return NaN
    end
end
get_op(opname, mop) = mopval -> get_op(mopval, opname, mop)

function cache_mopops(mops = build_mops(), path = joinpath(@__DIR__, "mopops.json"))
    x = rand(5000)
    y = mops(x, Union{Py, PyException})
    mopops = map(y) do mopval
        if pyisinstance(mopval, numbers.Number) || pyisinstance(mopval, numpy.ndarray)
            return []
        else
            return pyconvert(Vector{String}, mopval.keys())
        end
    end
    # base_names = map(mops |> collect) do mop
    #     JSON.parse(getdescription(mop))["base_name"]
    # end
    mopops = Dict(getnames(mops) .=> mopops)
    open(path, "w") do io
        JSON.print(io, mopops)
    end
    return mopops
end
function load_mopops(path = joinpath(@__DIR__, "mopops.json"))
    open(path, "r") do io
        mopops = JSON.parse(read(io, String))
        return mopops
    end
end

function build_ops(mops = build_mops(), mopops = load_mopops())
    return map(mops |> collect) do mop
               description = getdescription(mop)
               keywords = getkeywords(mop)
               name = getname(mop)
               #    base_name = JSON.parse(getdescription(mop))["base_name"]
               opnames = mopops[String(getname(mop))]
               if isempty(opnames)
                   return [SuperFeature(convert_op ∘ (getmethod ∘ getfeature)(mop), name,
                                        description,
                                        keywords, getsuper(mop))]
                   #    return [SuperFeature(convert_op, name, description, keywords, mop)]
               else
                   return map(opnames) do opname
                       SuperFeature(get_op(opname, mop),
                                    Symbol("$(name)_$opname"),
                                    description, keywords, mop)
                   end
               end
           end |> Iterators.flatten |> collect |> Vector{SuperFeature} |> SuperFeatureSet
end

include("Artifacts.jl")

# function build_ops()
#     mops = HCTSA.build_mops()
#     x = rand(5000)
#     ops = map(collect(mops)) do mop
#         y = mop(x)
#         description = getdescription(mop)
#         keywords = getkeywords(mop)
#         name = getname(mop)
#         base_name = JSON.parse(getdescription(mop))["base_name"]
#         if isa(y, PyException)
#             return nothing # Inconclusive
#         elseif pyisinstance(y, pybuiltins.dict)
#             return map(y.keys()) do opname
#                        SuperFeature(get_op(opname, mop),
#                                     Symbol("$(name)_$opname"),
#                                     description, keywords, mop)
#                    end |> collect |> FeatureSet
#         else # mop is an op
#             return [SuperFeature(convert_op, name, description, keywords, mop)] |>
#                    FeatureSet
#         end
#     end
#     ops = filter(!isnothing, ops)
#     ops = Iterators.flatten(ops) .|> SuperFeature
#     return FeatureSet(ops)
# end

# const pyOperations = build_ops()

# include("metadata.jl")
# include("testdata.jl")

# nancheck(𝐱::AbstractVector) = length(𝐱) < 3 || any(isnan, 𝐱) || any(isinf, 𝐱)

# function __catchaMouse16(𝐱::AbstractVector, fName::Symbol)::Float64
#     nancheck(𝐱) && return NaN
#     _ccall(fName, Cdouble)(𝐱)
# end

# """
#     _catchaMouse16(𝐱::AbstractArray{Float64}, fName::Symbol)
#     _catchaMouse16(fName::Symbol, 𝐱::AbstractArray{Float64})
# Evaluate the feature `fName` on the single time series `𝐱`. See `HCTSA.featuredescriptions` for a summary of the 22 available time series features. Time series with NaN or Inf values will produce NaN feature values.

# # Examples
# ```julia
# 𝐱 = HCTSA.testdata[:test]
# HCTSA._catchaMouse16(𝐱, :AC_nl_035)
# ```
# """
# function _catchaMouse16(𝐱::AbstractVector, fName::Symbol)::Float64
#     __catchaMouse16(𝐱, fName)
# end
# function _catchaMouse16(X::AbstractMatrix, fName::Symbol)::Matrix{Float64}
#     mapslices(𝐱 -> __catchaMouse16(𝐱, fName), X, dims = [1])
# end

# const features = map(featurenames) do name
#     function feature(𝐱::AbstractVector{<:Real})::Float64
#         nancheck(𝐱) && return NaN
#         _ccall(name, Cdouble)(convert(Vector{Float64}, 𝐱))
#     end
# end

# """
# The set of HCTSA features without a preliminary z-score
# """
# catchaMouse16_raw = FeatureSet(features, featurenames, featuredescriptions, featurekeywords)

# """
#     catchaMouse16(𝐱::Vector)
#     catchaMouse16(X::Array)
#     catchaMouse16[featurename::Symbol](X::Array)
# Evaluate all features for a time series vector `𝐱` or the columns of an array `X`.
# `catchaMouse16` is a FeatureSet, which means it can be indexed by feature names (as symbols) to return a subset of the available features.
# `getnames(catchaMouse16)`, `getkeywords(catchaMouse16)` and `getdescriptions(catchaMouse16)` will also return feature names, keywords and descriptions respectively.
# Features are returned in a `FeatureArray`, in which array rows are annotated by feature names. A `FeatureArray` can be converted to a regular array with `Array(F)`.

# # Examples
# ```julia
# 𝐱 = HCTSA.testdata[:test]
# 𝐟 = catchaMouse16(𝐱)

# X = randn(100, 10)
# F = catchaMouse16(X)
# F = catchaMouse16[:AC_nl_035](X)
# ```
# """
# catchaMouse16 = SuperFeatureSet(features, featurenames, featuredescriptions,
#                                 featurekeywords, zᶠ)
# export catchaMouse16

# for f in featurenames
#     eval(quote
#              $f = catchaMouse16[$(Meta.quot(f))]
#              export $f
#          end)
# end

# """
#     AC_nl_035(x::AbstractVector{Union{Float64, Int}}) # For example
# An alternative to `catchaMouse16(:AC_nl_035](x)`.
# All features, such as `AC_nl_035`, are exported as Features and can be evaluated by calling their names.

# # Examples
# ```julia
# 𝐱 = HCTSA.testdata[:test]
# f = AC_nl_035(𝐱)
# ```
# """
# AC_nl_035

# """
#     c16
# The HCTSA feature set with shortened names; see [`catchaMouse16`](@ref).
# """
# c16 = SuperFeatureSet(features, short_featurenames, featuredescriptions, featurekeywords,
#                       zᶠ)
# export c16

end
