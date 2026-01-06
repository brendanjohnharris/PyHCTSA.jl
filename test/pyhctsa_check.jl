using HCTSA
using PythonCall
using DataFrames

begin
    ops = HCTSA.build_ops()
    pops = HCTSA.calculator.FeatureCalculator()

    x = HCTSA.testdata(:test)
    a = @time ops(x)
    b = @time pops.extract(HCTSA.as_numpy(x)) |> PyTable |> DataFrame
end
begin # * Compare
    anames = map(string, getnames(a))
    _bnames = pyconvert(Vector{String}, b.columns)
    bnames = replace.(bnames, '.' => "_")
    bnames = map(bnames) do bname
        if bname[(end - 1):end] == "_0" && bname ∉ anames
            return bname[1:(end - 2)]
        else
            return bname
        end
    end
    idxs = indexin(bnames, anames)
end
Q = map(enumerate(idxs)) do (i, idx)
    if isnothing(idx)
        return NaN
    end
    _a = a[idx]
    _b = b[!, i] |> only
    if isnan(_a) && isnan(_b)
        return true
    end
    q = _a ≈ _b
    if !q
        @warn "Mismatch on feature $(anames[idx]) ($idx -> $_a), $(_bnames[i]) ($i -> $_b)"
    end
    return q
end
