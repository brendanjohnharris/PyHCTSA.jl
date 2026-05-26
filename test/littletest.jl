using PyHCTSA
using JSON
using PythonCall

begin
    mops = PyHCTSA.build_mops()
    x = rand(5000)
    F = @time mops(x, Union{Py, PyException})
    idxs = map(Base.Fix2(isa, PyException), F)
    Q = sum(idxs)
end

begin
    X = randn(1000, 20)
    Operations = PyHCTSA.build_ops()
    @time Operations(X)
end

begin
    fs = PyHCTSA.calculator.FeatureCalculator()
    a = @time fs.extract(PyHCTSA.as_numpy(X'))
    a = pyconvert(Array{Float64}, a.values)
end

begin # ! How to generate ops from mops?
    ops = PyHCTSA.Operations
    Y = ops(rand(2000))
end

begin
    function convert_op(mopval)
        if pyisinstance(mopval, PyHCTSA.numpy.ndarray)
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
        elseif haskey(mopval, opname)
            return convert_op(mopval[opname])
        else
            @warn "$(getname(mop)) output has no key $opname in: $mopval"
            return NaN
        end
    end
    get_op(opname, mop) = mopval -> get_op(mopval, opname, mop)

    mopops = PyHCTSA.load_mopops()
    ops = map(mops |> collect) do mop
        description = getdescription(mop)
        keywords = getkeywords(mop)
        name = getname(mop)
        base_name = JSON.parse(getdescription(mop))["base_name"]
        opnames = mopops[base_name]
        if isempty(opnames)
            return [SuperFeature(convert_op, name, description, keywords, mop)] |>
                FeatureSet
        else
            return map(opnames) do opname
                SuperFeature(
                    get_op(opname, mop),
                    Symbol("$(name)_$opname"),
                    description, keywords, mop
                )
            end |> collect |> FeatureSet
        end
    end |> Iterators.Flatten |> collect |> FeatureSet
    Y = ops(rand(2000))
end

# begin
#     function convert_op(mopval)
#         if pyisinstance(mopval, PyHCTSA.numpy.ndarray)
#             return pyconvert(Float64, mopval |> only)
#         elseif pyisinstance(mopval, pybuiltins.Exception)
#             @warn exception = (mopval, catch_backtrace())
#             return NaN
#         else
#             return pyconvert(Float64, mopval)
#         end
#     end
#     convert_op(x::Number) = convert(Float64, x)
#     function get_op(mopval, opname, mop)
#         if pyisinstance(mopval, pybuiltins.Exception)
#             @warn exception = (y, catch_backtrace())
#             return NaN
#         elseif haskey(mopval, opname)
#             return convert_op(mopval[opname])
#         else
#             @warn "$(getname(mop)) output has no key $opname in: $mopval"
#             return NaN
#         end
#     end
#     get_op(opname, mop) = mopval -> get_op(mopval, opname, mop)

#     mopops = PyHCTSA.load_mopops()
#     ops = map(mops |> collect) do mop
#               description = getdescription(mop)
#               keywords = getkeywords(mop)
#               name = getname(mop)
#               base_name = JSON.parse(getdescription(mop))["base_name"]
#               opnames = mopops[base_name]
#               if isempty(opnames)
#                   return [SuperFeature(convert_op, name, description, keywords, mop)] |>
#                          FeatureSet
#               else
#                   return map(opnames) do opname
#                              SuperFeature(get_op(opname, mop),
#                                           Symbol("$(name)_$opname"),
#                                           description, keywords, mop)
#                          end |> collect |> FeatureSet
#               end
#           end |> Iterators.Flatten |> collect |> FeatureSet
#     Y = ops(rand(2000))
# end
# f = PyHCTSA.calculator.FeatureCalculator()

# config = f.config |> PyHCTSA.py2dict
# feature_funcs = f.feature_funcs |> PyHCTSA.py2dict

# * Now we get each 'Master Operation' from the config, extract the func from feature_funcs,
#   and build a Feature. If the config says zscore, we use a SuperFeature with the zscore
