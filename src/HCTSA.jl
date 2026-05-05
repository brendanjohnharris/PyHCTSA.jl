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

const PREPROCESS_CONFIG_KEYS = Set(["zscore", "abs"])
DEFAULT_CHART() = Chart(ProgressLogger())

function __init__()
    pycopy!(pyhctsa, pyimport("pyhctsa"))
    pycopy!(pyOperations, pyimport("pyhctsa.operations"))
    pycopy!(calculator, pyimport("pyhctsa.calculator"))
    pycopy!(utils, pyimport("pyhctsa.utils"))
    pycopy!(jpype, pyimport("jpype"))
    pycopy!(yaml, pyimport("yaml"))
    pycopy!(numpy, pyimport("numpy"))
    pycopy!(numbers, pyimport("numbers"))

    # * Load all Operation submodules
    pycopy!(pkgutil, pyimport("pkgutil"))
    for modinfo in pkgutil.iter_modules(pyOperations.__path__)
        pyimport("pyhctsa.operations.$(modinfo.name)")
    end

    # * Kill python messages
    logging = pyimport("logging")
    logging.disable(logging.CRITICAL)
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

function _metadata_dependencies(mop::Dict)
    deps = haskey(mop, "dependencies") ?
           get(mop, "dependencies", String[]) :
           get(mop, "depedencies", String[])
    if isnothing(deps)
        return String[]
    elseif deps isa AbstractVector
        return String.(deps)
    elseif deps isa AbstractString
        return [String(deps)]
    else
        return String[]
    end
end

function description(mop::Dict)
    md = Dict{String, Any}()
    haskey(mop, "base_name") && (md["base_name"] = String(mop["base_name"]))
    if haskey(mop, "legacy_name") && !isnothing(mop["legacy_name"])
        md["legacy_name"] = String(mop["legacy_name"])
    end
    md["dependencies"] = _metadata_dependencies(mop)
    return JSON.json(md)
end

function keywords(mop::Dict)
    if haskey(mop, "labels") && mop["labels"] isa AbstractVector
        return String.(mop["labels"])
    end
    return String[]
end

_keystring(k) = k isa AbstractString ? String(k) : string(k)

function normalize_config(x::AbstractVector)
    pylist(x)
end
normalize_config(x) = identity(x)
function mop_func(module_name::String, func_name::String, config::Dict)
    config = filter(p -> _keystring(p.first) ∉ PREPROCESS_CONFIG_KEYS, config)
    config = [Symbol(k) => normalize_config(v)
              for (k, v) in config if _keystring(k) ∉ PREPROCESS_CONFIG_KEYS]
    op_module = getproperty(HCTSA.pyOperations, module_name)
    op_func = getproperty(op_module, func_name)
    function f(x)
        try
            return PythonCall.GIL.@lock op_func(x; config...)
        catch e
            @debug "Error computing feature $module_name:$func_name" exception=(e,
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
function isabs(config::Dict)
    haskey(config, "abs") && config["abs"] == true
end
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
- Boolean parameters return their key when true and are omitted when false.
- Negative numbers get 'm' prefix (e.g., -1 -> "m1")
- Floats between 0-1 use '0p' format (e.g., 0.5 -> "0p5")
- Other floats use 'p' for decimal point (e.g., 1.5 -> "1p5")
- Lists show as "start_end" if contiguous range, else joined with "_"
"""
function format_param_value(val::Bool, key::Union{Nothing, String} = nothing)
    return val ? (isnothing(key) ? "" : key) : ""
end

function format_param_value(val::Number, key::Union{Nothing, String} = nothing)
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

function format_param_value(val::AbstractVector, key::Union{Nothing, String} = nothing)
    # Check if contiguous integer range
    if length(val) > 1 && all(x -> x isa Number, val)
        diffs = diff(val)
        if all(==(1), diffs)
            return format_param_value(first(val)) * "_to_" * format_param_value(last(val))
        end
    end
    return join(format_param_value.(val), "_")
end

format_param_value(val, key::Union{Nothing, String} = nothing) = string(val)

"""
    name_config(config::Dict, mopconfig::Dict; do_zscore::Bool=true, do_absval::Bool=false)

Generate the parameter suffix for a feature name, matching pyhctsa's naming convention.
If ordered_args is provided, parameters are ordered accordingly.
Excludes preprocessing keys from the name.
"""
function name_config(config::Dict, mopconfig::Dict; do_zscore::Bool = true,
                     do_absval::Bool = false)
    # Filter out preprocessing keys
    params = filter(p -> _keystring(p.first) ∉ PREPROCESS_CONFIG_KEYS, config)
    params_by_string = Dict(_keystring(k) => v for (k, v) in params)
    parts = String[]

    if haskey(mopconfig, "ordered_args") && !isnothing(mopconfig["ordered_args"]) &&
       !isempty(mopconfig["ordered_args"])
        # Use ordered_args to determine parameter order
        for arg in mopconfig["ordered_args"]
            arg_str = _keystring(arg)
            if haskey(params_by_string, arg_str)
                formatted = format_param_value(params_by_string[arg_str], arg_str)
                !isempty(formatted) && push!(parts, formatted)
            end
        end
        if isempty(parts) && !isempty(mopconfig["ordered_args"])
            @warn "ordered_args references keys not in config" ordered_args=mopconfig["ordered_args"] params
        end
    else
        # Default: "key{value}" format for each param
        for (k, v) in params
            key = String(k)
            formatted = format_param_value(v, key)
            if isempty(formatted)
                continue
            end
            if v isa Bool
                push!(parts, formatted)
            else
                push!(parts, "$(k)$(formatted)")
            end
        end
    end

    label_suffix = isempty(parts) ? "" : "_" * join(parts, "_")
    if !do_zscore
        label_suffix *= "_raw"
    end
    if do_absval
        label_suffix *= "_abs"
    end
    return label_suffix
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

function java_filter(ops)
    copystacks = get(ENV, "JULIA_COPY_STACKS", 0)
    if !(copystacks isa Integer) && !(copystacks isa Bool)
        copystacks = Meta.parse(copystacks)
    end
    if convert(Int64, copystacks) == 1
        return ops
    else
        deps = map(getdescriptions(ops)) do desc
            desc_dict = JSON.parse(desc)
            if haskey(desc_dict, "dependencies")
                return desc_dict["dependencies"]
            elseif haskey(desc_dict, "depedencies")
                return desc_dict["depedencies"]
            else
                return String[]
            end
        end
        java_ops = map(deps) do dep
            depvec = dep isa AbstractVector ? String.(dep) : String[]
            return any(==("jpype1"), depvec)
        end
        if any(java_ops)
            @info "Filtering out $(sum(java_ops)) features with Java dependencies (JULIA_COPY_STACKS=$(copystacks)). Please set JULIA_COPY_STACKS=1 to include these features."
        end
        return ops[.!java_ops]
    end
end

"Builds the set of mops defined by an entry in the yaml file. Note that each entry can specify more than one mop, based on the length of the config entry"
function build_mops(module_name::String, func_name::String, mopconfig::Dict)
    configs = mopconfig["configs"]
    configs = map(flatten_config, configs) |> Iterators.flatten
    features = map(configs) do config
        do_zscore = iszscore(config)
        do_absval = isabs(config)
        if do_zscore && do_absval
            super = numpy_abs_zscore
        elseif do_zscore
            super = numpy_zscore
        elseif do_absval
            super = numpy_abs
        else
            super = numpy_identity
        end
        pop!(config, "zscore", nothing)
        pop!(config, "abs", nothing)
        fullname = name(mopconfig) *
                   name_config(config, mopconfig; do_zscore = do_zscore,
                               do_absval = do_absval)
        SuperFeature(mop_func(module_name, func_name, config), Symbol(fullname),
                     description(mopconfig),
                     keywords(mopconfig), Feature(super))
    end
    mops = features |> SuperFeatureSet |> java_filter
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

function _default_config_path()
    pyhctsa_file = pyconvert(String, HCTSA.pyhctsa.__file__)
    cfg_path = normpath(joinpath(dirname(pyhctsa_file), "configurations", "hctsa.yaml"))
    isfile(cfg_path) ||
        throw(ArgumentError("pyhctsa configuration file not found at $cfg_path"))
    return cfg_path
end

function load_config(path = nothing)
    if isnothing(path)
        path = _default_config_path()
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

function convert_op(mopval, mopname = "")
    if pyhasattr(mopval, "ndim") && pyhasattr(mopval, "size") && pyhasattr(mopval, "item")
        mopval = mopval.item()
    end

    if pyisinstance(mopval, pybuiltins.Exception)
        @debug "Feature $mopname returned a Python exception" exception=(mopval,
                                                                         catch_backtrace())
        return NaN
    elseif pyisinstance(mopval, pybuiltins.str)
        msg = try
            pyconvert(String, mopval)
        catch
            string(mopval)
        end
        @debug "Feature $mopname returned Python string output: $msg"
        return NaN
    elseif pyisinstance(mopval, numbers.Number)
        return pyconvert(Float64, mopval)
    else
        try
            return pyconvert(Float64, mopval)
        catch e
            @debug "Failed to convert $mopname feature output to Float64. This probably means the `mopops.json` has incorrect fields for this mop:" exception=(e,
                                                                                                                                                               catch_backtrace())
            @debug mopval
            return NaN
        end
    end
end
function convert_op(E::PyException, mopname = "")
    @debug "PyException in feature $mopname" exception=(E, catch_backtrace())
    return NaN
end
convert_op(x::Number, args...) = convert(Float64, x)

function _mop_output_keys(mopval)
    if pyisinstance(mopval, pybuiltins.dict)
        return pyconvert(Vector{String}, pybuiltins.list(mopval.keys()))
    end
    return String[]
end

function get_op(mopval, opname, mop)
    if mopval isa PyException || pyisinstance(mopval, pybuiltins.Exception)
        @debug "Feature $(getname(mop)).$opname returned an exception when extracting dict key" exception=(mopval,
                                                                                                           catch_backtrace())
        return NaN
    elseif pyisinstance(mopval, pybuiltins.dict)
        if haskey(mopval, opname)
            return convert_op(mopval[opname], opname)
        elseif haskey(mopval, "out")
            return convert_op(mopval["out"], opname)
        end
        @warn "$(getname(mop)) output has no key $opname. Valid keys: $(_mop_output_keys(mopval))"
        return NaN
    elseif pyisinstance(mopval, numbers.Number)
        return pyconvert(Float64, mopval)
    elseif pyisinstance(mopval, numpy.ndarray)
        @warn "$(getname(mop)) output $mopval is not subscriptable; maybe it errored and returned NaN?"
        return NaN
    else
        @warn "$(getname(mop)) output has unsupported container type for key extraction"
        return NaN
    end
end
get_op(opname, mop) = mopval -> get_op(mopval, opname, mop)

function mop_quality(y)
    Q = map(y) do mopval
        if mopval isa PyException || pyisinstance(mopval, pybuiltins.Exception)
            return false
        elseif pyisinstance(mopval, pybuiltins.str)
            return false
        elseif pyisinstance(mopval, numbers.Number)
            return true
        elseif pyisinstance(mopval, numpy.ndarray)
            return true
        elseif pyisinstance(mopval, pybuiltins.dict)
            return true
        else
            return false
        end
    end
end

function cache_mopops(mops = build_mops(),
                      path = joinpath(@__DIR__, "../assets/mopops.json"))
    x = rand(5000)
    y = mops(x, Union{Py, PyException})
    Q1 = mop_quality(y)
    mopops = map(y) do mopval
        if mopval isa PyException || pyisinstance(mopval, pybuiltins.Exception)
            return String[]
        elseif pyisinstance(mopval, numbers.Number) || pyisinstance(mopval, numpy.ndarray)
            return []
        else
            return _mop_output_keys(mopval)
        end
    end

    x = testdata(:test)
    y = mops(x, Union{Py, PyException})
    Q2 = mop_quality(y)
    test_mopops = map(y) do mopval
        if mopval isa PyException || pyisinstance(mopval, pybuiltins.Exception)
            return String[]
        elseif pyisinstance(mopval, numbers.Number) || pyisinstance(mopval, numpy.ndarray)
            return []
        else
            return _mop_output_keys(mopval)
        end
    end

    # * Join test mopops
    mopops = map(zip(mopops, test_mopops)) do (mopop, test_mopop)
        if isempty(mopop)
            test_mopop
        else
            mopop
        end
    end

    # base_names = map(mops |> collect) do mop
    #     JSON.parse(getdescription(mop))["base_name"]
    # end
    mopops = Dict(getnames(mops) .=> mopops)
    open(path, "w") do io
        JSON.print(io, mopops, 2)
    end
    Q = Q1 .| Q2 # False here means we didn't successfully capture the 'output mode' of the mop, so we can't reliably use it for inferring ops
    if !(all(Q))
        @debug "Some mops did not return valid outputs during caching, so their mopops may be incomplete or inaccurate. Consider running mopops caching with a different test input or checking for errors in mop execution."
    end
    return mopops, Q
end
function load_mopops(path = joinpath(@__DIR__, "../assets/mopops.json"))
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
               opnames = get(mopops, String(getname(mop)), String[])
               if isempty(opnames)
                   return [
                       SuperFeature(x -> convert_op(((getmethod ∘ getfeature)(mop))(x),
                                                    name), name,
                                    description,
                                    keywords, getsuper(mop))]
                   #    return [SuperFeature(convert_op, name, description, keywords, mop)]
               else
                   return map(opnames) do opname
                       SuperFeature(get_op(opname, mop),
                                    Symbol("$(name).$opname"),
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
