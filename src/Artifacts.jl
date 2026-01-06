using DelimitedFiles
using Pkg.Artifacts

artifact_toml = joinpath(pkgdir(HCTSA), "Artifacts.toml")
test_hash = artifact_hash("testdata", artifact_toml)

if test_hash == nothing || !artifact_exists(test_hash)
    test_hash = create_artifact() do artifact_dir
        test_url_base = "https://raw.githubusercontent.com/brendanjohnharris/catch22/8c76cabb12b9005990a357fd907c37ce8858510d/testData" # v0.4.0
        testfiles = [
            "test.txt",
            "test2.txt",
            "testInf.txt",
            "testInfMinus.txt",
            "testNaN.txt",
            "testShort.txt",
            "testSinusoid.txt"
        ]
        [download("$(test_url_base)/$f", joinpath(artifact_dir, f)) for f in testfiles]
    end
    bind_artifact!(artifact_toml, "testdata", test_hash)
end

test_datadir = artifact_path(test_hash)

const testnames = [:test
                   :test2
                   :testInf
                   :testInfMinus
                   :testNaN
                   :testShort
                   :testSinusoid]

function _loaddata(x)
    reduce(vcat,
           readdlm(normpath(joinpath(test_datadir, String(x) * ".txt")), ' ', Float64,
                   '\n'))
end
function _loadoutput(x)
    file = normpath(joinpath(test_datadir, String(x) * "_output.txt"))
    if isfile(file)
        out = readdlm(file, ',', comments = true)
        return Dict([Symbol(x[2][2:end]) => x[1] for x in eachrow(out)])
    else
        return nothing
    end
end

"""
    testdata(name::Symbol)

Load test data by name. Data is loaded lazily (only when requested).
Available datasets: $testnames

# Examples
```julia
data = testdata(:test)
data = testdata(:testSinusoid)
```
"""
function testdata(name::Symbol)
    if name ∉ testnames
        throw(ArgumentError("Unknown test dataset: $name. Available: $testnames"))
    end
    return _loaddata(name)
end
