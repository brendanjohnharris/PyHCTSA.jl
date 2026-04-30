using Test
using TestItems
using TestItemRunner

@run_package_tests

@testsnippet Setup begin
    using HCTSA
    using JSON

    "Return a stable small module/function pair from the pyhctsa config."
    function first_config_entry(config)
        module_name = "distribution"
        if !haskey(config, module_name)
            module_name = first(sort(collect(keys(config))))
        end
        func_name = first(sort(collect(keys(config[module_name]))))
        return module_name, func_name
    end
end

@testitem "pyhctsa config loads" setup=[Setup] begin
    config = HCTSA.load_config()
    @test config isa Dict
    @test !isempty(config)
    @test haskey(config, "distribution")
end

@testitem "config key validation" setup=[Setup] begin
    config = HCTSA.load_config()
    module_name, func_name = first_config_entry(config)

    @test_throws ArgumentError HCTSA._validate_config_keys!(module_name, func_name,
                                                            Dict("doAbs" => true))
    @test HCTSA._validate_config_keys!(module_name, func_name,
                                       Dict("D" => 3, "abs" => true, "zscore" => false)) isa
          Dict
end

@testitem "build_mops smoke" setup=[Setup] begin
    mops = HCTSA.build_mops("distribution")
    @test length(mops) > 0

    x = randn(1024)
    y = mops(x, Any)
    @test length(y) == length(mops)
end

@testitem "build_ops from cached mop outputs" setup=[Setup] begin
    all_mops = HCTSA.build_mops("distribution")
    n = min(12, length(all_mops))
    subset = SuperFeatureSet(collect(all_mops)[1:n])

    tmp, io = mktemp()
    close(io)
    mopops = HCTSA.cache_mopops(subset, tmp)
    @test mopops isa Dict
    @test length(mopops) == n

    ops = HCTSA.build_ops(subset, mopops)
    @test length(ops) >= n

    out = ops(randn(1024))
    @test length(out) == length(ops)
end

@testitem "metadata normalization" setup=[Setup] begin
    d = HCTSA.description(Dict("base_name" => "foo",
                               "legacy_name" => "bar",
                               "dependencies" => ["numpy", "scipy"]))
    parsed = JSON.parse(d)
    @test parsed["base_name"] == "foo"
    @test parsed["legacy_name"] == "bar"
    @test parsed["dependencies"] == ["numpy", "scipy"]
end
