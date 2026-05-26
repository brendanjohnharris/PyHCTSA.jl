using Test
using TestItems
using TestItemRunner

@run_package_tests

@testsnippet Setup begin
    using PyHCTSA
    using JSON
    using PythonCall
    using TimeseriesFeatures

    "Return a stable small module/function pair from the pyhctsa config."
    function first_config_entry(config)
        module_name = "distribution"
        if !haskey(config, module_name)
            module_name = first(sort(collect(keys(config))))
        end
        func_name = first(sort(collect(keys(config[module_name]))))
        return module_name, func_name
    end

    "Create a small stable subset of a feature set."
    function first_subset(fs, n = 12)
        all = collect(fs)
        nkeep = min(n, length(all))
        return SuperFeatureSet(all[1:nkeep])
    end
end

@testitem "pyhctsa config loads" setup = [Setup] begin
    config = PyHCTSA.load_config()
    @test config isa Dict
    @test !isempty(config)
    @test haskey(config, "distribution")
end

@testitem "build_mops" setup = [Setup] begin
    mops = PyHCTSA.build_mops("distribution")
    @test length(mops) > 0

    x = randn(1024)
    y = mops(x, Any)
    @test length(y) == length(mops)
end

@testitem "build_ops from cached mop outputs" setup = [Setup] begin
    all_mops = PyHCTSA.build_mops("distribution")
    n = min(12, length(all_mops))
    subset = SuperFeatureSet(collect(all_mops)[1:n])

    tmp, io = mktemp()
    close(io)
    mopops, Q = PyHCTSA.cache_mopops(subset, tmp)
    @test mopops isa Dict
    @test length(mopops) == n

    ops = PyHCTSA.build_ops(subset, mopops)
    @test length(ops) >= n

    out = ops(randn(1024))
    @test length(out) == length(ops)
end

@testitem "metadata normalization" setup = [Setup] begin
    d = PyHCTSA.description(
        Dict(
            "base_name" => "foo",
            "legacy_name" => "bar",
            "dependencies" => ["numpy", "scipy"]
        )
    )
    parsed = JSON.parse(d)
    @test parsed["base_name"] == "foo"
    @test parsed["legacy_name"] == "bar"
    @test parsed["dependencies"] == ["numpy", "scipy"]
end

@testitem "py2dict recursively converts Python containers" setup = [Setup] begin
    pd = PythonCall.pydict()
    pd["a"] = 1
    pd["b"] = PythonCall.pylist([2, 3])
    inner = PythonCall.pydict()
    inner["c"] = "x"
    pd["inner"] = inner

    d = PyHCTSA.py2dict(pd)
    @test d isa Dict
    @test d["a"] == 1
    @test d["b"] == [2, 3]
    @test d["inner"]["c"] == "x"
end

@testitem "metadata dependency normalization variants" setup = [Setup] begin
    d1 = JSON.parse(
        PyHCTSA.description(
            Dict(
                "base_name" => "f",
                "dependencies" => ["a", "b"]
            )
        )
    )
    @test d1["dependencies"] == ["a", "b"]

    d2 = JSON.parse(PyHCTSA.description(Dict("base_name" => "f", "depedencies" => ["jpype1"])))
    @test d2["dependencies"] == ["jpype1"]

    d3 = JSON.parse(
        PyHCTSA.description(
            Dict(
                "base_name" => "f",
                "dependencies" => "statsmodels"
            )
        )
    )
    @test d3["dependencies"] == ["statsmodels"]

    d4 = JSON.parse(PyHCTSA.description(Dict("base_name" => "f", "dependencies" => nothing)))
    @test d4["dependencies"] == String[]
end

@testitem "keywords returns labels or empty" setup = [Setup] begin
    @test PyHCTSA.keywords(Dict("labels" => ["distribution", "location"])) ==
        ["distribution", "location"]
    @test PyHCTSA.keywords(Dict("labels" => "not-a-vector")) == String[]
    @test PyHCTSA.keywords(Dict("base_name" => "x")) == String[]
end

@testitem "name throws without base_name" setup = [Setup] begin
    @test_throws ArgumentError PyHCTSA.name(Dict("legacy_name" => "x"))
end

@testitem "flatten_config cartesian product" setup = [Setup] begin
    cfg = Dict("d" => [1.0, 0.5], "D" => [3, 5], "zscore" => true)
    flat = collect(PyHCTSA.flatten_config(cfg))
    @test length(flat) == 4
    @test all(haskey.(flat, Ref("d")))
    @test all(haskey.(flat, Ref("D")))
    @test all(haskey.(flat, Ref("zscore")))

    scalar_cfg = Dict("tau" => 1, "zscore" => true)
    scalar_flat = collect(PyHCTSA.flatten_config(scalar_cfg))
    @test length(scalar_flat) == 1
    @test scalar_flat[1]["tau"] == 1
end

@testitem "format_param_value semantics" setup = [Setup] begin
    @test PyHCTSA.format_param_value(true, "zscore") == "zscore"
    @test PyHCTSA.format_param_value(false, "zscore") == ""
    @test PyHCTSA.format_param_value(-1) == "m1"
    @test PyHCTSA.format_param_value(0.5) == "0p5"
    @test PyHCTSA.format_param_value(2.75) == "2p75"
    @test PyHCTSA.format_param_value([1, 2, 3]) == "1_to_3"
    @test PyHCTSA.format_param_value([1, 3, 5]) == "1_3_5"
end

@testitem "name_config ordered and unordered modes" setup = [Setup] begin
    ordered_cfg = Dict("d" => 0.5, "D" => 3, "zscore" => true, "abs" => false)
    mopcfg_ordered = Dict("ordered_args" => ["d", "D"], "base_name" => "pol_var")
    s1 = PyHCTSA.name_config(ordered_cfg, mopcfg_ordered; do_zscore = true, do_absval = false)
    @test s1 == "_0p5_3"

    unordered_cfg = Dict("tau" => 3, "what_method" => "ols", "zscore" => false)
    mopcfg_unordered = Dict("base_name" => "partial_autocorr")
    s2 = PyHCTSA.name_config(
        unordered_cfg, mopcfg_unordered; do_zscore = false,
        do_absval = true
    )
    @test startswith(s2, "_")
    @test occursin("tau3", s2)
    @test occursin("what_methodols", s2)
    @test endswith(s2, "_raw_abs")
end

@testitem "zscore and abs flags" setup = [Setup] begin
    @test PyHCTSA.iszscore(Dict("zscore" => true))
    @test !PyHCTSA.iszscore(Dict("zscore" => false))
    @test !PyHCTSA.iszscore("not-a-dict")

    @test PyHCTSA.isabs(Dict("abs" => true))
    @test !PyHCTSA.isabs(Dict("abs" => false))
    @test !PyHCTSA.isabs(1)
end

@testitem "default config path resolves" setup = [Setup] begin
    cfg_path = PyHCTSA._default_config_path()
    @test isfile(cfg_path)

    cfg = PyHCTSA.load_config(cfg_path)
    @test cfg isa Dict
    @test !isempty(cfg)
end

@testitem "build_mops overload consistency" setup = [Setup] begin
    config = PyHCTSA.load_config()
    module_name, func_name = first_config_entry(config)
    mopconfig = config[module_name][func_name]

    by_triplet = PyHCTSA.build_mops(module_name, func_name, mopconfig)
    by_module_dict = PyHCTSA.build_mops(module_name, Dict(func_name => mopconfig))

    @test length(by_triplet) > 0
    @test length(by_triplet) == length(by_module_dict)
end

@testitem "convert_op and get_op behavior" setup = [Setup] begin
    mops = PyHCTSA.build_mops("distribution")
    mop = collect(mops)[1]

    @test PyHCTSA.convert_op(1) == 1.0
    @test PyHCTSA.convert_op(1.25) == 1.25
    @test isnan(PyHCTSA.convert_op(Py("bad-output"), "bad-output"))

    d1 = PythonCall.pydict()
    d1["answer"] = 4.0
    @test PyHCTSA.get_op(d1, "answer", mop) == 4.0

    d2 = PythonCall.pydict()
    d2["out"] = 5.0
    @test PyHCTSA.get_op(d2, "missing", mop) == 5.0

    d3 = PythonCall.pydict()
    d3["other"] = 6.0
    @test isnan(PyHCTSA.get_op(d3, "missing", mop))
end

@testitem "mop_quality classification" setup = [Setup] begin
    d = PythonCall.pydict()
    d["x"] = 1
    vals = Any[1.0, Py(2), Py("bad"), d]
    q = PyHCTSA.mop_quality(vals)
    @test q == [true, true, false, true]
end

@testitem "java_filter handles env parsing" setup = [Setup] begin
    config = PyHCTSA.load_config()
    @test haskey(config, "correlation")
    @test haskey(config["correlation"], "add_noise")

    old = get(ENV, "JULIA_COPY_STACKS", nothing)
    try
        ENV["JULIA_COPY_STACKS"] = "0"
        filtered = PyHCTSA.build_mops(
            "correlation", "add_noise",
            config["correlation"]["add_noise"]
        )

        ENV["JULIA_COPY_STACKS"] = "1"
        unfiltered = PyHCTSA.build_mops(
            "correlation", "add_noise",
            config["correlation"]["add_noise"]
        )

        @test length(unfiltered) >= length(filtered)
        @test length(unfiltered) > 0
    finally
        if isnothing(old)
            pop!(ENV, "JULIA_COPY_STACKS", nothing)
        else
            ENV["JULIA_COPY_STACKS"] = old
        end
    end
end

@testitem "cache_mopops and load_mopops round-trip" setup = [Setup] begin
    all_mops = PyHCTSA.build_mops("distribution")
    subset = first_subset(all_mops, 8)

    tmp, io = mktemp()
    close(io)

    mopops, Q = PyHCTSA.cache_mopops(subset, tmp)
    @test mopops isa Dict
    @test length(mopops) == length(subset)
    @test Q isa AbstractVector{Bool}
    @test length(Q) == length(subset)

    loaded = PyHCTSA.load_mopops(tmp)
    @test loaded isa JSON.Object
    @test sort(string.(keys(loaded))) == sort(string.(keys(mopops)))
end

@testitem "build_ops works with subset and mopops" setup = [Setup] begin
    all_mops = PyHCTSA.build_mops("distribution")
    subset = first_subset(all_mops, 10)

    tmp, io = mktemp()
    close(io)
    mopops, _ = PyHCTSA.cache_mopops(subset, tmp)

    ops = PyHCTSA.build_ops(subset, mopops)
    @test length(ops) >= length(subset)

    out_vec = ops(randn(1024))
    @test length(out_vec) == length(ops)

    out_mat = ops(randn(1024, 3))
    @test size(out_mat, 1) == length(ops)
end

@testitem "testdata API" setup = [Setup] begin
    for name in PyHCTSA.testnames
        x = PyHCTSA.testdata(name)
        @test x isa Vector{Float64}
        @test !isempty(x)
    end

    @test_throws ArgumentError PyHCTSA.testdata(:does_not_exist)
end

@testitem "Compare outputs with pyhctsa" setup = [Setup] begin
    include("val_check.jl")
end

# @testitem "Cast example" setup=[Setup] begin
#     using Asciicast
#     cast_readme(PyHCTSA)
# end
