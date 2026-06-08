### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# ╔═╡ ab32a4c8-ef62-47ca-99a5-4e9f5a72f898
begin
	using Revise
	eval(:(import Pkg; Pkg.develop(["KrylovKit"])))
	using KrylovKit
end

# ╔═╡ 54c0aa98-fd35-11f0-8dc8-4fb8474a532c
using ApproxFun, Integrals, VectorInterface, LinearAlgebra, SpecialFunctions

# ╔═╡ 291983e2-bb22-47f1-a745-a3849ff1f7bd
begin
	using VectorInterface: scale!, add!

	VectorInterface.scalartype(::Type{<:Fun{S, T}}) where {S, T} = T

	VectorInterface.scale!(f::Fun, α::Number) = (scale!(coefficients(f), α); f)
	VectorInterface.scale!!(f::Fun, α::Number) = scale!(f, α)
	VectorInterface.scale(f::Fun, α::Number) = scale!(copy(f), α)

	function VectorInterface.add!(f::Fun, g::Fun, α::Number, β::Number)
		cf, cg = coefficients(f), coefficients(g)
		nf, ng = length(cf), length(cg)
		if nf < ng
			add!(cf, view(cg, 1:nf), α, β)
			resize!(cf, length(cg))
			add!(view(cf, (nf + 1):ng), view(cg, (nf + 1):ng), α, false)
		else
			add!(view(cf, 1:ng), cg, α, β)
			scale!(view(cf, (ng + 1):nf), β)
		end
		return f
	end
	VectorInterface.add!!(f::Fun, g::Fun, α::Number, β::Number) = add!(f, g, α, β)
	VectorInterface.add(f::Fun, g::Fun, α::Number, β::Number) = add!(copy(f), g, α, β)
end

# ╔═╡ 56d20146-8bc1-4bbe-a581-ab0301e89be7
begin
	eval(:(import Pkg; Pkg.add(; url = "https://github.com/Eliassj/ARS.jl")))
	using ARS
end

# ╔═╡ de2c7e75-e8b1-4d58-871a-1a779f0c0358
using BenchmarkTools

# ╔═╡ 1256d1b5-c862-404b-9949-564ff55c2339
using DifferentiationInterface, ForwardDiff

# ╔═╡ 99d6765c-dfdb-4dad-840c-f7e99faaf199
using DifferentiationInterface: Constant

# ╔═╡ 1512ed7f-9660-4a4c-90f8-9b7d97ac2fa8
using LogDensityProblems, LogDensityProblemsAD, DynamicHMC, DynamicHMC.Diagnostics, Mooncake

# ╔═╡ ad4470f3-0f08-422c-be29-062aabacda77
using WGLMakie

# ╔═╡ b85d1fee-d2a9-429e-87bc-066c01ece42e
using CairoMakie

# ╔═╡ 1b9964f6-8d8a-4731-b7f5-dc42b25a0c19
using MathTeXEngine

# ╔═╡ ceecbb7f-a41d-4882-81eb-dd6306c78378
using SkewLinearAlgebra

# ╔═╡ 7e14cebd-c78e-490b-a5c3-db1f6c2e2754
using Combinatorics: combinations

# ╔═╡ b08fbcf5-5486-4d87-a4fa-44d8fe6f782c
using Random

# ╔═╡ 0b444cd0-ca0e-4109-8fc6-23bb7c499751
using OhMyThreads, FHist, Statistics, Distributions

# ╔═╡ b0c54a2c-abe0-4ce8-9f50-61d25119d872
using RandomMatrices

# ╔═╡ f604b534-86c4-4371-9f71-38f6846c84e7
using ForwardDiff: derivative

# ╔═╡ 9e7b838e-7e94-4a83-8404-faa921fb1b5d
begin
	eval(:(import Pkg; Pkg.add(; url = "https://github.com/simeonschaub/FredholmDeterminants.jl")))
	using FredholmDeterminants: TracyWidom, FredholmDeterminants
end

# ╔═╡ f0d08e48-1b50-4e11-8cb8-9908ef5ca115
using BayesDensityHistSmoother: HistSmoother, BayesDensityHistSmoother as BD

# ╔═╡ f064608d-a0e4-4d21-a102-e492237d7d19
using FastGaussQuadrature

# ╔═╡ e35e1679-971e-470e-9bf5-1efd6283485b
N = 10

# ╔═╡ a9b3177c-b619-49a2-b839-5e93742a795b
function skew_dot(f::Fun{<:Hermite}, g::Fun{<:Hermite})
	f′, g′ = Derivative() * f, Derivative() * g
	inner(f, g) = sum(CartesianIndices((ncoefficients.((f, g)))); init = 0.0) do I
		i, j = Tuple(I)
		m, n = i - 1, j - 1
		isodd(m + n) && return 0.0
		(-1)^(m ÷ 2 + n ÷ 2) * exp2((m + n - 1) / 2) * gamma((m + n + 1) / 2) * coefficient(f, i) * coefficient(g, j)
	end
	return (inner(f, g′) - inner(f′, g))
end

# ╔═╡ 99664353-1ca2-406c-9fc9-c9e0335bb76e
# ╠═╡ disabled = true
#=╠═╡
function skew_dot′(f::Fun{<:Hermite}, g::Fun{<:Hermite})
	f′, g′ = Derivative() * f, Derivative() * g
	inner(f, g) = sum(1:minimum(ncoefficients.((f, g))); init = 0.0) do i
		n = i - 1
		exp2(n) * factorial(n) * coefficient(f, i) * coefficient(g, i)
	end
	return √π * (inner(f, g′) - inner(f′, g))
end
  ╠═╡ =#

# ╔═╡ ec4fec4a-8c22-450a-9721-c4b21f5957f6
w(x) = exp(-x^2)

# ╔═╡ bfa16f35-e8ff-4605-a4bc-df385438a019
function skew_dot′(f, g)
	f′, g′ = Derivative() * f, Derivative() * g
	integrand((x,), p) = w(x)^2 * (f(x) * g′(x) - f′(x) * g(x))
	prob = IntegralProblem(integrand, (-Inf, Inf))
	sol = solve(prob, HCubatureJL(); reltol = 1e-6, abstol = 1e-6)
	return sol.u
end

# ╔═╡ 80242e98-449b-41ab-aaf1-6e054dffdbb1
function norm(f::Fun{<:Hermite})
	nrm_sqr = √π * sum(enumerate(coefficients(f))) do (i, c)
		n = i - 1
		exp2(n) * factorial(n) * c^2
	end
	return √nrm_sqr
end

# ╔═╡ a0356246-980e-4725-bdb1-f067ca5d773c
function standard_inner(f::Fun{<:Hermite}, g::Fun{<:Hermite})
	return √π * sum(1:minimum(ncoefficients.((f, g))); init = 0.0) do i
		n = i - 1
		exp2(n) * factorial(n) * coefficient(f, i) * coefficient(g, i)
	end
end

# ╔═╡ 390dea24-024e-4c10-8ec1-991c3682ab6a
let x = Fun(Hermite())
	global A(f) = SymplecticFormVec(InnerProductVec(x * f[][], f[].dotf), f.skewf)
end

# ╔═╡ 31f01e7f-48a0-4105-8379-217ead9d7031
begin
	v₀ = SymplecticFormVec(InnerProductVec(Fun(Hermite(), [1.0]), standard_inner), (f, g) -> skew_dot(f[], g[]))
	itr = ArnoldiIterator(A, v₀, ModifiedSymplecticGramSchmidt2(ESR2))
	fact = initialize(itr)
	for _ in 1:(2N - 1)
		expand!(itr, fact)
	end
	H = [rayleighquotient(fact); zeros(1, 2N - 1) normres(fact)]
	W = getindex.(getindex.(basis(fact).basis))
end

# ╔═╡ a4b08932-9ea9-4d25-a052-4060e7bd5bee
domain = -4.9375:.125:4.9375 #quantile.(Normal(0, 1 / √2), .005:.01:.995) #-2.875:.25:2.875

# ╔═╡ 641e4213-4017-4694-b185-ed6b25a36ba4
function randPfPPseq!(K, n_max = size(K, 1) ÷ 2)
    n = size(K, 1) ÷ 2
    𝓘 = Vector{Int}(undef, n_max)
    count = 0

    # Track original block indices through pivoting
    perm = Vector(1:n)

    @inbounds for j in 1:n
        # ── Partial pivoting: pick block k ≥ j with largest |K[2k-1,2k]| ──
        _, pivot = findmax(k -> abs(K[2k-1, 2k]), j:n)
        pivot += j - 1
        if pivot != j
            i1, i2, k1, k2 = 2j-1, 2j, 2pivot-1, 2pivot
            m = 2n
            for col in 1:m                           # swap rows
                K[i1, col], K[k1, col] = K[k1, col], K[i1, col]
                K[i2, col], K[k2, col] = K[k2, col], K[i2, col]
            end
            for row in 1:m                           # swap cols
                K[row, i1], K[row, k1] = K[row, k1], K[row, i1]
                K[row, i2], K[row, k2] = K[row, k2], K[row, i2]
            end
            perm[j], perm[pivot] = perm[pivot], perm[j]
        end

        # ── Bernoulli sampling with prob Pf(K[j,j]) ──
        r = K[2j-1, 2j]
        if rand() < r
            count += 1
            𝓘[count] = perm[j]
        else
            r -= 1                                   # Pfaffian "K[j,j] -= 1"
        end

        (count == n_max || j == n) && break

        # ── Rank-2 Schur complement update (allocation-free) ──
        #   K[q,q] -= K[q,p] * (1/r)[0 -1;1 0] * K[p,q]
        #          =  (1/r)(u₁v₂' − u₂v₁')   via two BLAS.ger! calls
        rinv = inv(r)
        i1, i2 = 2j-1, 2j
        qr  = 2j+1:2n
        Kqq = @view K[qr, qr]
        u₁  = @view K[qr, i1]               # column views (no alloc)
        u₂  = @view K[qr, i2]
        v₁  = @view K[i1, qr]               # row views (strided, no alloc)
        v₂  = @view K[i2, qr]
        BLAS.ger!(-rinv, u₂, v₁, Kqq)       # Kqq -= (1/r) u₂ v₁'
        BLAS.ger!( rinv, u₁, v₂, Kqq)       # Kqq += (1/r) u₁ v₂'
    end

    return sort!(resize!(𝓘, count))
end

# ╔═╡ 817eaab2-703d-4605-805b-4efc86f8bbb7
WGLMakie.activate!()

# ╔═╡ dc2bcab5-14cd-4919-a395-c7bdae7d1764
# ╠═╡ disabled = true
#=╠═╡
data_GibbsMALA′ = randPfPPGibbsMALA(N, 10000, K; backend = AutoMooncakeForward())
  ╠═╡ =#

# ╔═╡ f7b4b8d8-6607-4dda-8e2d-00714ed7c913
# ╠═╡ disabled = true
#=╠═╡
randPfPPGibbsMALA(N, 10000, K; backend = AutoMooncake())
  ╠═╡ =#

# ╔═╡ dab7adf3-8500-44d5-9f39-2f6efb73e136
function skewadjugate(S::SkewHermitian{<:Real})
	n = size(S, 1)
	(isodd(n) || n == 0) && return SkewHermitian(zeros(T, n, n))
	eigvals, eigvecs = eigen(S)

	T = real(eltype(eigvals))
	dadj = similar(eigvals, T)
    prod = one(T)
    for i = 1:2:length(eigvals)
		d = imag(eigvals[i])
        dadj[i] = prod
        dadj[i + 1] = -prod
        prod *= d
    end
    prod = one(T)
    for i = (length(eigvals) - 1):-2:1
        dadj[i] *= prod
        dadj[i + 1] *= prod
        prod *= imag(eigvals[i])
    end

	return SkewHermitian{T, Matrix{T}}(real.(det(eigvecs) .* eigvecs * Diagonal(im .* dadj) * eigvecs'))
end

# ╔═╡ bf652d8f-c51f-463e-b386-02606346e25e
begin
	Mooncake.@is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof(pfaffian), AbstractMatrix{<:LinearAlgebra.BlasReal}}
	function Mooncake.rrule!!(::Mooncake.CoDual{typeof(pfaffian)}, x::Mooncake.CoDual{<:AbstractMatrix{<:LinearAlgebra.BlasReal}})
		function pfaffian_adjoint(dy)
			x.dx .+= dy .* skewadjugate(SkewHermitian(x.x)) ./ 2
		    return Mooncake.NoRData(), Mooncake.NoRData()
		end
		return Mooncake.CoDual(pfaffian(x.x), Mooncake.NoFData()), pfaffian_adjoint
	end
	Mooncake.@is_primitive Mooncake.DefaultCtx Mooncake.ReverseMode Tuple{typeof(logabspfaffian), AbstractMatrix{<:LinearAlgebra.BlasReal}}
	function Mooncake.rrule!!(::Mooncake.CoDual{typeof(logabspfaffian)}, x::Mooncake.CoDual{<:AbstractMatrix{<:LinearAlgebra.BlasReal}})
		logabs, sgn = logabspfaffian(x.x)
		function logabspfaffian_adjoint(dy)
			# dy[1] is the cotangent of log|pf(A)|; sign is discrete so dy[2] is ignored
			x.dx .+= dy[1] .* skewadjugate(SkewHermitian(x.x)) ./ (2 * sgn * exp(logabs))
		    return Mooncake.NoRData(), Mooncake.NoRData()
		end
		return Mooncake.CoDual((logabs, sgn), Mooncake.NoFData()), logabspfaffian_adjoint
	end
	ignore(x) = x
	Mooncake.@zero_adjoint Mooncake.DefaultCtx Tuple{typeof(ignore), Any}
end

# ╔═╡ be1925cf-d859-47b9-ac08-8f962e4ceb70
set_texfont_family!(FontFamily(merge(FontFamily("NewComputerModern").fonts, Dict(:bold => "TeXGyreHerosMakie/TeXGyreHerosMakie-Bold.otf")); special_chars = MathTeXEngine._symbol_to_new_computer_modern))
# set_texfont_family!()

# ╔═╡ c09eeec0-0958-4fd1-87ce-756a5f1def12
CairoMakie.activate!()

# ╔═╡ 69fae8b8-56ba-401b-9895-0b5d1346e86f
let N = 6 #2N
	color = repeat(Makie.wong_colors(), fld1(N, 7))[1:N]
	fig = Figure(; size = (500, 500))

	ax = Axis(fig[1, 1]; limits = ((-3, 3), (-3, 3)))
	hidexdecorations!(ax; grid = false)
	x = range(-3, 3; length = 255)
	for i in 1:N
		lines!(ax, x, W[i].(x))
		#scatter!(ax, domain, W[i].(domain))
	end
	axislegend(ax, [
		[LineElement(; color)]#, MarkerElement(; color, marker = :circle)]
		for (N, color) in zip(0:(N - 1), color)
	], map(N -> L"p_{%$N}(x)", 0:(N - 1)))

	ax2 = Axis(fig[2, 1]; limits = ((-3, 3), (-.1, 1.2)), xlabel = L"x")
	lines!(ax2, x, map(w, x); color = :gray, label = L"w(x)")
	#scatter!(ax2, domain, map(w, domain); color = :gray, label = L"w(x)")
	axislegend(ax2; merge = true)
	rowsize!(fig.layout, 2, Relative(0.2))

	Label(fig[0, 1], L"$\beta = 4$\textbf{ Skew-Orthogonal Polynomials w.r.t. }$w(x) = e^{-x^2}$"; tellwidth = false, font = ax.titlefont, fontsize = 18)

	save("gse_sop.pdf", fig)

	fig
end

# ╔═╡ a2ceb55d-1c72-490e-858f-4795d8e1c3cb
# ╠═╡ disabled = true
#=╠═╡
begin
	# see https://www.brandeis.edu/mathematics/people/adlerpublications/22.skeworthogpol+randommat.pdf Eq (2.8)
	f = Matrix{Float64}(undef, 2length(domain), 2length(domain))

	w⁻¹∂ₓw = (Derivative() - Fun(Hermite(), [0, 1]))
    S₄(x, y) = (w(x) * w(y)) * sum(0:(N - 1)) do m
        (W[2m + 1](x) * (w⁻¹∂ₓw * W[2m + 2])(y) - W[2m + 2](x) * (w⁻¹∂ₓw * W[2m + 1])(y))
    end
    D₄(x, y) = (w(x) * w(y)) * sum(0:(N - 1)) do m
        ((w⁻¹∂ₓw * W[2m + 1])(x) * (w⁻¹∂ₓw * W[2m + 2])(y) - (w⁻¹∂ₓw * W[2m + 2])(x) * (w⁻¹∂ₓw * W[2m + 1])(y))
    end
	function I₄(x, y)
		integrand((y′,), p) = S₄(x, y′)
		prob = IntegralProblem(integrand, (x, y))
		sol = solve(prob, HCubatureJL(); reltol = 1e-12, abstol = 1e-12)
		return -sol.u
	end

    for (i, x) in enumerate(domain), (j, y) in enumerate(domain)
        f[2i - 1, 2j - 1] = S₄(x, y)
        f[2i, 2j - 1] = D₄(x, y)
        f[2i - 1, 2j] = I₄(x, y)
        f[2i, 2j] = S₄(y, x)
    end
	f .*= 0.125
	K = JMatrix(2length(domain)) * f
	tr(f), eigvals(f), K[77:82, 77:82]
end
  ╠═╡ =#

# ╔═╡ 5b22deaa-6087-4c76-b530-04e594a5e705
let W = W, N = N
	# see https://www.brandeis.edu/mathematics/people/adlerpublications/22.skeworthogpol+randommat.pdf Eq (2.8)
	global f = Matrix{Float64}(undef, 2length(domain), 2length(domain))

	∂ₓW = Ref(Derivative()) .* W
    global S₄(x, y) = (w(x) * w(y)) * sum(0:(N - 1)) do m
         ∂ₓW[2m + 2](x) * W[2m + 1](y) - ∂ₓW[2m + 1](x) * W[2m + 2](y)
    end
    global D₄(x, y) = (w(x) * w(y)) * sum(0:(N - 1)) do m
         -∂ₓW[2m + 2](x) * ∂ₓW[2m + 1](y) + ∂ₓW[2m + 1](x) * ∂ₓW[2m + 2](y)
    end
	global I₄(x, y) = (w(x) * w(y)) * sum(0:(N - 1)) do m
         W[2m + 2](x) * W[2m + 1](y) - W[2m + 1](x) * W[2m + 2](y)
    end

    for (i, x) in enumerate(domain), (j, y) in enumerate(domain)
		Δ = step(domain) # √((binedges[i + 1] - binedges[i]) * (binedges[j + 1] - binedges[j]))
        f[2i - 1, 2j - 1] = S₄(x, y) * Δ
        f[2i, 2j - 1] = I₄(x, y) * Δ
        f[2i - 1, 2j] = D₄(x, y) * Δ
        f[2i, 2j] = S₄(y, x) * Δ
    end
	global K = JMatrix(2length(domain)) * f
end

# ╔═╡ 85435207-3685-45fb-937d-396eeb19e18e
let
	obj = ARS.Objective(x -> log(S₄(x, x)))
	#sam = ARS.ARSampler(obj, [range(-5, 5; length = 201);], (-Inf, Inf))
	sam = ARS.ARSampler(obj, [range(-5, 5; length = 501);], (-Inf, Inf))
	fig = hist(ARS.sample!(sam, 1000000); normalization = :pdf, bins = 100)
	lines!(-7..7, s -> S₄(s, s) / N)
	fig
end

# ╔═╡ c49b6a97-56fd-4a50-ab47-63a384c61073
function sample_diag(; w = 2.0)
	s₀ = 2randn()
	log_u = log(S₄(s₀, s₀)) + log(rand())

	l = s₀ - w * rand()
	r = l + w

	while log(S₄(l, l)) > log_u
		l -= w
	end
	while log(S₄(r, r)) > log_u
		r += w
	end

	while true
		s = l + rand() * (r - l)
		if log(S₄(s, s)) >= log_u
			return s
		elseif s < s₀
			l = s
		else
			r = s
		end
	end
end

# ╔═╡ be347205-af4f-49ef-9b8d-037613d94d1b
let
	fig = hist([sample_diag() for _ in 1:1000000]; normalization = :pdf, bins = 100)
	lines!(-7..7, s -> S₄(s, s) / N)
	fig
end

# ╔═╡ 0515ab0f-bfde-4a0d-988d-1dcb36e17478
begin
	struct RPCholCache{T, S}
		n::Int
		sam::S
		S::Vector{T}
		L::Matrix{T}
		k::Matrix{T}
	end
	function RPCholCache(n::Int)
		obj = ARS.Objective(x -> log(S₄(x, x)))
		sam = ARS.ARSampler(obj, [range(-5, 5; length = 501);], (-Inf, Inf))
		return RPCholCache(n, sam, Vector{Float64}(undef, n),
						   Matrix{Float64}(undef, 2n - 2, 2n - 2),
						   Matrix{Float64}(undef, 2n - 2, 2),
						  )
	end
end

# ╔═╡ a6addf6f-278c-4f7d-8063-a79b5193dba7
let
	fig = surface(domain, domain, K[1:2:end, 2:2:end]; axis = (; type = Axis3, clip = false))
	#=x, y = -3.0888478567047395, 2.1648702740511316
	@show S₄(domain[15], domain[57]) * step(domain) S₄(domain[16], domain[58]) * step(domain)
	scatter!([[Point3f(x, y, @show S₄(x, y) * step(domain))]; Point3f.(getindex.(Ref(domain), [15, 16]), getindex.(Ref(domain), [57, 58]), @show getindex.(Ref(K), 2 .* [15, 16] .- 1, 2 .* [57, 58]))]; color = [:red, :pink, :magenta])=#
	lines!(Point3f.(domain, domain, S₄.(domain, domain) .* step(domain)); color = :red, linewidth = 3)
	fig
end

# ╔═╡ ffea356e-d9f5-48e4-b79b-6fc7c61c03ca
let
	fig = surface(domain, domain, K[1:2:end, 1:2:end]; axis = (; type = Axis3, clip = false))
	y, x = -3.0888478567047395, 2.1648702740511316
	scatter!([[Point3f(x, y, I₄(x, y) * step(domain))]; Point3f.(getindex.(Ref(domain), [57, 58]), getindex.(Ref(domain), [15, 16]), .-getindex.(Ref(K), 2 .* [15, 16] .- 1, 2 .* [57, 58] .- 1))]; color = [:red, :pink, :magenta])
	fig
end

# ╔═╡ 57e76596-23e0-42e9-8322-a415030c3186
I₄(100., -100.)

# ╔═╡ 6d81d722-9f4a-4328-9ad7-df4dc738ce33
surface(domain, domain, -K[2:2:end, 2:2:end]; axis = (; type = Axis3))

# ╔═╡ af8adeda-3f36-4a70-9090-849a8bef35a3
surface(domain, domain, [dot(K[(2i - 1):2i, 2j - 1], JMatrix(2), K[(2i - 1):2i, 2j]) / K[2i - 1, 2i] for i in eachindex(domain), j in eachindex(domain)]; axis = (; type = Axis3))

# ╔═╡ bd1f7abc-716c-4d9a-97f7-2b03f7e5876b
surface(domain, domain, [K[2j - 1, 2j] - dot(K[(2i - 1):2i, 2j - 1], JMatrix(2), K[(2i - 1):2i, 2j]) / K[2i - 1, 2i] for i in eachindex(domain), j in eachindex(domain)]; axis = (; type = Axis3))

# ╔═╡ f1645e25-2def-4fb3-861b-cb2a6801a250
sum([K[2j - 1, 2j] - dot(K[(2i - 1):2i, 2j - 1], JMatrix(2), K[(2i - 1):2i, 2j])  / K[2i - 1, 2i] for i in eachindex(domain), j in eachindex(domain)]; dims = 2)

# ╔═╡ f5a11c98-a7b8-45d6-8f2b-4a880911a073
sum([K[2j - 1, 2j] / (N - 1) - dot(K[(2i - 1):2i, 2j - 1], JMatrix(2), K[(2i - 1):2i, 2j]) / (N * (N - 1)) / (K[2i - 1, 2i] / N) for i in eachindex(domain), j in eachindex(domain)]; dims = 2)

# ╔═╡ 2a32b857-3803-478e-9425-b3cb45aefd13
surface(domain, domain, [det(K[(2i - 1):2i, (2j - 1):2j]) for i in eachindex(domain), j in eachindex(domain)]; axis = (; type = Axis3))

# ╔═╡ 5f99c038-67b9-4b7d-893b-90279597f7c1
begin
	function fill_kernel!(K, X, Y, m = length(X), n = length(Y))
		for i in 1:m, j in 1:n
			x, y = X[i], Y[j]
			K[2i - 1, 2j - 1] = I₄(x, y)
			K[2i, 2j - 1] = -S₄(x, y)
			K[2i - 1, 2j] = S₄(y, x)
			K[2i, 2j] = -D₄(x, y)
	    end
		return K
	end
	function evaluate_conditioned_pfpp!(k, L, S, s, i = length(S))
		fill_kernel!(k, S, s, i, 1)
		_k = view(k, 1:2i, :)
		ldiv!(LowerTriangular(view(L, 1:2i, 1:2i)), _k)
		kₛₛ = S₄(s, s)
		d = kₛₛ - dot(view(_k, :, 1), JMatrix(2i), view(_k, :, 2))
		return d, kₛₛ
	end
	function update_cholesky!(L, k, sqrt_d, i = size(k, 1) ÷ 2)
		for j in 1:(i - 1)
			k[2j - 1, 1], k[2j, 1] = -k[2j, 1], k[2j - 1, 1]
			k[2j - 1, 2], k[2j, 2] = -k[2j, 2], k[2j - 1, 2]
		end
		L[(2i - 1):2i, 1:(2i - 2)] .= view(k, 1:(2i - 2), :)'
		L[2i - 1, 2i - 1] = L[2i, 2i] = sqrt_d
		return L
	end
end

# ╔═╡ 6a9f7aed-f542-4494-9072-d7530b85d030
function randPfPPRPChol!((; n, sam, S, L, k)::RPCholCache{T}; max_iters = 1000) where {T}
    i = 0
	L .= zero(T)

    for _ in 1:max_iters
		s = ARS.sample!(sam, 1)[]
		d, kₛₛ = evaluate_conditioned_pfpp!(k, L, S, s, i)
		p = d / 2kₛₛ
		0 ≤ p ≤ 1 || @warn i, d, kₛₛ, d / 2kₛₛ

        if rand() < p
            i += 1
            S[i] = s
			i == n && return sort!(S)
			update_cholesky!(L, k, √d, i)
		end
    end
	@warn "Exceeded max_iters = $max_iters"
	return sort(view(S, 1:i))
end

# ╔═╡ d4ccb607-ff22-4718-8271-dde92d790da3
@btime randPfPPRPChol!($(RPCholCache(N)))

# ╔═╡ df695fff-9770-4ed5-8702-719db2fefc6f
data_rpchol = let c = RPCholCache(N)
	stack(_ -> randPfPPRPChol!(c), 1:1000)
end

# ╔═╡ ff34a5d9-6df4-4804-957b-f9442934ebf9
function randPfPPRPCholSlice!((; n, sam, S, L, k)::RPCholCache{T}; max_iters = 1000, w = 1.0) where {T}
    i = 0
	L .= zero(T)

    g(s, i) = log(max(evaluate_conditioned_pfpp!(k, L, S, s, i)[1], 0.0))

	while true
		s₀ = √2 * randn()
		log_u = g(s₀, i) + log(rand())
		log_u == -Inf && continue

		l = s₀ - w * rand()
		r = l + w

		while g(l, i) > log_u
			l -= w
		end
		while g(r, i) > log_u
			r += w
		end

		while true
			s = l + rand() * (r - l)
			if (log_d = g(s, i)) >= log_u
	            i += 1
	            S[i] = s
				i == n && return sort!(S)
				update_cholesky!(L, k, exp(log_d / 2), i)
				break
			elseif s < s₀
				l = s
			else
				r = s
			end
		end
    end
end

# ╔═╡ 8e9dec92-5175-4e94-b0dd-8b32887464e5
@btime randPfPPRPCholSlice!($(RPCholCache(N)))

# ╔═╡ e983f0ad-93b2-40dc-947b-86637c30da50
function randPfPPGibbs(S₀, T; w = 1.0)
	N = length(S₀)
	S = Matrix{Float64}(undef, N, T + 1)
	L = zeros(2N - 2, 2N - 2)
	k = Matrix{Float64}(undef, 2N - 2, 2)
	for j in 1:(N - 1)
		s = S₀[j]
		d, _ = evaluate_conditioned_pfpp!(k, L, S₀, s, j - 1)
		update_cholesky!(L, k, √d, j)
		S[j, 1] = s
	end
	S[N, 1] = S₀[N]

	for t in 2:(T + 1)
		@views S[:, t] .= S[:, t - 1]
		for i in N:-1:1
			for j in i:(N - 1)
				s = S[j + 1, t]
				d, _ = evaluate_conditioned_pfpp!(k, L, view(S, :, t), s, j - 1)
				update_cholesky!(L, k, √d, j)
				S[j, t] = s
			end

    		g(s) = log(max(evaluate_conditioned_pfpp!(k, L, view(S, :, t), s, N - 1)[1], 0.0))

			s₀ = S[i, t - 1]
			log_u = g(s₀) + log(rand())

			l = s₀ - w * rand()
			r = l + w

			while g(l) > log_u
				l -= w
			end
			while g(r) > log_u
				r += w
			end

			while true
				s = l + rand() * (r - l)
				if g(s) >= log_u
					S[end, t] = s
					break
				elseif s < s₀
					l = s
				else
					r = s
				end
			end
		end
	end
	reverse!(@view S[:, 2:2:end]; dims = 1)
	return S
end

# ╔═╡ 92a81d80-5a3b-4869-b514-d4f86966431d
gibbs_data = tmapreduce(hcat, 1:16) do _
	randPfPPGibbs(range(-3, 3; length = N), 10000)
end

# ╔═╡ d409eccf-1971-40f7-8b16-4dfaedc0792c
begin
	struct LogLikelihoodForwardDiff{T}
		S::Matrix{T}
		L::Matrix{T}
		k::Matrix{ForwardDiff.Dual{ForwardDiff.Tag{LogLikelihoodForwardDiff{T}, Float64}, Float64, 1}}
	end
	struct LogLikelihoodOther{T}
		S::Matrix{T}
		L::Matrix{T}
		k::Matrix{T}
	end
	LogLikelihood{T} = Union{LogLikelihoodForwardDiff{T}, LogLikelihoodOther{T}}
	LogLikelihood(S::Matrix{T}, L::Matrix{T}, ::AutoForwardDiff) where {T} = LogLikelihoodForwardDiff(S, L, Matrix{ForwardDiff.Dual{ForwardDiff.Tag{LogLikelihoodForwardDiff{T}, Float64}, Float64, 1}}(undef, size(L, 1), 2))
	LogLikelihood(S::Matrix{T}, L::Matrix{T}, ::Any) where {T} = LogLikelihoodOther(S, L, Matrix{T}(undef, size(L, 1), 2))

	function ((; S, L, k)::LogLikelihood)(s, t)
		d, _ = evaluate_conditioned_pfpp!(k, L, view(S, :, t), s, size(S, 1) - 1)
		return log(max(d, 0.0))
	end
end

# ╔═╡ 28157a28-fc41-4c2d-aa4f-eae711b77b66
function randPfPPGibbsMALA(S₀, T; ε = 0.57, backend = AutoForwardDiff())
	N = length(S₀)
	S = Matrix{Float64}(undef, N, T + 1)
	L = zeros(2N - 2, 2N - 2)
	k = Matrix{Float64}(undef, 2N - 2, 2)
	for j in 1:(N - 1)
		s = S₀[j]
		d, _ = evaluate_conditioned_pfpp!(k, L, S₀, s, j - 1)
		update_cholesky!(L, k, √d, j)
		S[j, 1] = s
	end
	S[N, 1] = S₀[N]

	g = LogLikelihood(S, L, backend)
	prep = prepare_derivative(g, backend, 0.0, Constant(1))

	for t in 2:(T + 1)
		@views S[:, t] .= S[:, t - 1]
		for i in N:-1:1
			for j in i:(N - 1)
				s = S[j + 1, t]
				d, _ = evaluate_conditioned_pfpp!(k, L, view(S, :, t), s, j - 1)
				update_cholesky!(L, k, √d, j)
				S[j, t] = s
			end

			sᵢ = S[i, t - 1]
			𝓁ᵢ, ∇𝓁ᵢ = value_and_derivative(g, prep, backend, sᵢ, Constant(t))

			z = randn()
			s′ = sᵢ + (ε^2 / 2) * ∇𝓁ᵢ + ε * z

			𝓁′, ∇𝓁′ = value_and_derivative(g, prep, backend, s′, Constant(t))

		    μ  = sᵢ + (ε^2 / 2) * ∇𝓁ᵢ
		    μ′ = s′ + (ε^2 / 2) * ∇𝓁′

			log_α = 𝓁′ - 𝓁ᵢ - ((sᵢ - μ′)^2 - (s′ - μ)^2) / 2ε^2

			if log(rand()) < log_α
		        S[end, t] = s′
			else
		        S[end, t] = sᵢ
		    end
		end
	end
	reverse!(@view S[:, 2:2:end]; dims = 1)
	return S
end

# ╔═╡ e01f28ee-240d-4eea-ba29-ba90603f69ef
begin
	Random.seed!(1)
	data_GibbsMALA = tmapreduce(hcat, 1:16) do _
		randPfPPGibbsMALA(range(-3, 3; length = N), 10000)
	end
end

# ╔═╡ b3ef3789-26c9-45e9-be14-a3cf36c7a8e1
begin
	struct PfPPProblem{T}
		K::Matrix{T}
	end
	PfPPProblem(N) = PfPPProblem(Matrix{Float64}(undef, 2N, 2N))
	function LogDensityProblems.logdensity((; K)::PfPPProblem, S)
		fill_kernel!(K, S, S)
		return logabspfaffian(K)[1]
	end
	LogDensityProblems.dimension((; K)::PfPPProblem) = size(K, 1) ÷ 2
end

# ╔═╡ 59e8fa13-4e28-49b7-b78e-424d9b7a765f
∇P = ADgradient(AutoMooncake(), PfPPProblem(N); x = zeros(N))

# ╔═╡ ce812a23-9084-4671-b3bc-9f5693165186
tr(f)

# ╔═╡ f188a104-c693-4005-bd5d-766bf056ed53
eigen(K, JMatrix(2length(domain))).values

# ╔═╡ 72d85e49-3294-4568-b833-dda51fc38653
h = randPfPPseq!(copy(K))

# ╔═╡ b545d6b1-b3cc-4268-b8dc-409b8e1bddf2
data_hmc = mcmc_with_warmup(Random.default_rng(), ∇P, 1000; initialization = (; q = domain[h]), reporter = LogProgressReport(; time_interval_s = 10.0))

# ╔═╡ f2c0ebc8-73e1-4468-8335-ca6d02e9543d
binedges = -5:0.125:5 #clamp.(quantile.(Normal(0, 1 / √2), 0:.01:1), -10, 10)

# ╔═╡ 0ed0bc4a-0ad1-4716-9eb2-4a06e7a58ede
begin
	binedges′ = (√2 * binedges .- 2√N) .* (2N)^(1 / 6)
	domain′ = (√2 * domain .- 2√N) .* (2N)^(1 / 6)
end

# ╔═╡ 3e4b06a0-96a5-432e-b70a-3c222cbabfe4
begin
	hists1 = [Hist1D(; counttype = Int, binedges = binedges′) for _ in 1:N, _ in 1:50]
	@tasks for _ in 1:10000
		@local kernel = similar(K)
		for i in 1:50
			copyto!(kernel, K)
			h = randPfPPseq!(kernel, N)
			if length(h) == N
				atomic_push!.(@view(hists1[:, i]), (√2 * domain[reverse(h)] .- 2√N) .* (2N)^(1 / 6))
			end
		end
	end
	hists1_mean = map(1:N) do i
		c = stack(bincounts.(@view(hists1[i, :])))
		m = mean(c; dims = 2)
		Hist1D(; binedges = binedges′, bincounts = vec(m))
	end
	hists1_errors = map(1:N) do i
		c = stack(bincounts.(normalize.(@view(hists1[i, :]))))
		m = mean(c; dims = 2)
		s = std(c; dims = 2)
		Vec3f.(domain′, vec(m), vec(s))
	end
end

# ╔═╡ 31a32ac1-c4ef-4f5f-b350-0f7ed9ca839d
begin
	hists2 = [Hist1D(; counttype = Int, binedges = binedges′) for _ in 1:N, _ in 1:50]
	@tasks for _ in 1:10000
		for i in 1:50
			λ = (√(2N) * eigvals(rand(GaussianHermite(4), N))[1:2:end] .- 2√N) .* (2N)^(1 / 6)
			atomic_push!.(@view(hists2[:, i]), reverse(λ))
		end
	end
	hists2_mean = map(1:N) do i
		c = stack(bincounts.(@view(hists2[i, :])))
		m = mean(c; dims = 2)
		Hist1D(; binedges = binedges′, bincounts = vec(m))
	end
	hists2_errors = map(1:N) do i
		c = stack(bincounts.(normalize.(@view(hists2[i, :]))))
		m = mean(c; dims = 2)
		s = std(c; dims = 2)
		Vec3f.(domain′, vec(m), vec(s))
	end
end

# ╔═╡ fc5925a1-4b9a-4d74-a290-43a6ff147d8e
xlims = extrema(bincenters(hists2_mean[1])[bincounts(hists2_mean[1]) .> 0]) .+ (-1, 1)

# ╔═╡ 6ab62af9-d3a3-48f0-b4e1-74f19a0de1e2
begin
	_log10(x) = x < 0 ? -log(floatmax()) : log10(x)
	Makie.inverse_transform(::typeof(_log10)) = Makie.inverse_transform(log10)
	Makie.defaultlimits(::typeof(_log10)) = Makie.defaultlimits(log10)
	Makie.defined_interval(::typeof(_log10)) = Makie.defined_interval(log10)
	Makie.get_ticks(::Makie.Automatic, ::typeof(_log10), any_formatter, vmin, vmax) = Makie.get_ticks(Makie.Automatic(), log10, any_formatter, vmin, vmax)
end

# ╔═╡ 2fb8d3a9-4afa-41d4-9c2d-a883565b8a71
let
	fig = Figure(; size = (650, 650))
	ax = Axis(fig[1, 1]; yscale = _log10, limits = (extrema(domain′), (1e-5, 5)))
	tightlimits!(ax)
	for i in 1:N
		xlims = extrema(bincenters(hists2_mean[i])[bincounts(hists2_mean[i]) .> 0]) .+ (-1, 1)
		ax′ = Axis(fig[fld1(i + 1, 2), mod1(i + 1, 2)]; limits = (xlims, (0, 1.1 * maximum(bincounts(normalize(hists2_mean[i]))))))
		tightlimits!(ax′)

		for ax in [ax, ax′]
			stairs!(ax, normalize(hists1_mean[i]); color = Cycled(i))
			errorbars!(ax, hists1_errors[i] .- 0.25 * Vec3f(.15, 0, 0); color = Cycled(i))
			stairs!(ax, normalize(hists2_mean[i]); linestyle = :dash, linewidth = 2, color = Cycled(i))
			errorbars!(ax, hists2_errors[i] .+ 0.25 * Vec3f(.15, 0, 0); color = Cycled(i))
		end
	end
	Legend(fig[:, 3],
		[
			[
				[LineElement(; color = :gray25), LineElement(; color = :gray25, points = Point2f[(0.35, 0.2), (0.35, .8)])],
				[LineElement(; color = :gray25, linestyle = :dash), LineElement(; color = :gray25, points = Point2f[(0.65, 0.2), (0.65, .8)])],
			],
			[PolyElement(; color, strokecolor = :transparent) for color in Cycled.(1:N)],
		],
		[
			["PfPP", "GSE"],
			string.(1:N),
		],
		["Source", "Row"],
	)
	fig
end

# ╔═╡ 3805f0d6-cd08-4fac-871d-6adfe36aead8
# ╠═╡ disabled = true
#=╠═╡
data = let
	data = Vector{Float64}(undef, 50000)
	@tasks for i in eachindex(data)
		@local c = RPCholCache(N)
		data[i] = randPfPPRPChol!(c)[end]
	end
	data
end
  ╠═╡ =#

# ╔═╡ f0813f13-dab7-47c9-8603-e885bcc4d451
# ╠═╡ disabled = true
#=╠═╡
data = vec(maximum(gibbs_data; dims = 1))
  ╠═╡ =#

# ╔═╡ 0e9b6628-bb12-4db3-b2fe-e93151b685f9
data = vec(maximum(data_GibbsMALA; dims = 1))

# ╔═╡ 778f3aed-0685-4422-a782-72927dd99d35
# ╠═╡ disabled = true
#=╠═╡
data = vec(maximum(data_hmc.posterior_matrix; dims = 1))
  ╠═╡ =#

# ╔═╡ 288cae69-47ab-44ca-8134-c5809347fee3
begin
	data_GibbsMALA_sorted = sort(data_GibbsMALA; dims = 1, rev = true)
	hists3 = [Hist1D(@view(data_GibbsMALA_sorted[i, 10001 * (j - 1) .+ (1001:10001)]); counttype = Int, binedges) for i in 1:N, j in 1:16]
	hists3_mean = map(1:N) do i
		c = stack(bincounts.(@view(hists3[i, :])))
		m = mean(c; dims = 2)
		Hist1D(; binedges, bincounts = vec(m))
	end
	hists3_errors = map(1:N) do i
		c = stack(bincounts.(normalize.(@view(hists3[i, :]))))
		m = mean(c; dims = 2)
		s = std(c; dims = 2)
		Vec3f.(domain, vec(m), vec(s))
	end
end

# ╔═╡ d242b2ca-264d-42e8-8368-f55dbac3a37b
begin
	gibbs_data_sorted = sort(gibbs_data; dims = 1, rev = true)
	hists4 = [Hist1D(@view(gibbs_data_sorted[i, 10001 * (j - 1) .+ (1001:10001)]); counttype = Int, binedges) for i in 1:N, j in 1:16]
	hists4_mean = map(1:N) do i
		c = stack(bincounts.(@view(hists4[i, :])))
		m = mean(c; dims = 2)
		Hist1D(; binedges, bincounts = vec(m))
	end
	hists4_errors = map(1:N) do i
		c = stack(bincounts.(normalize.(@view(hists4[i, :]))))
		m = mean(c; dims = 2)
		s = std(c; dims = 2)
		Vec3f.(domain, vec(m), vec(s))
	end
end

# ╔═╡ 1e269a47-8f8d-48b9-97e9-5164c2b1e045
# ╠═╡ disabled = true
#=╠═╡
fit_gibbs_mala = tmap(1:N) do i
	#histsmoother = HistSmoother(vec(stack(j -> data_GibbsMALA_sorted[i, 10001 * (j - 1) .+ (101:10001)], 1:16)))
	histsmoother = HistSmoother(data_GibbsMALA_sorted[i, 101:1100])

	BD.varinf(histsmoother)[1]
end
  ╠═╡ =#

# ╔═╡ fe5e43cf-fde2-4535-b914-a10a14e17f46
fit_gibbs_mala = let histsmoother = HistSmoother((√2 * data_GibbsMALA_sorted[1, 101:1100] .- 2√N) .* (2N)^(1 / 6))
	[BD.sample(histsmoother, 10100)]
end

# ╔═╡ 439390ab-4e4b-4a01-9806-5c06b331dbc8
let
	fig = Figure(; size = (500, 350))
	ax = Axis(fig[1, 1]; limits = ((-5, 1), (0, 1.1 * maximum(bincounts(normalize(hists2_mean[1]))))))

	#hist!(ax, data; bins = binedges, normalization = :pdf, color = :gray)
	#plot!(ax, normalize(hists3_mean[1]); color = :lightgray)
	#errorbars!(ax, hists3_errors[1]; color = :gray, linewidth = 2)	

	stairs!(ax, normalize(hists1_mean[1]))
	errorbars!(ax, hists1_errors[1] .- step(binedges′) * Vec3f(.15, 0, 0); color = Cycled(1), linewidth = 2)
	stairs!(ax, normalize(hists2_mean[1]); color = :red, linewidth = 2, linestyle = :dash)
	errorbars!(ax, hists2_errors[1] .+ step(binedges′) * Vec3f(.15, 0, 0); color = :red, linewidth = 2)

	x = 0:(length(domain) - 1)
	y = map(x) do k
		pfaffian(skewhermitian!(JMatrix(2(length(domain) - k)) - K[(2k + 1):2length(domain), (2k + 1):2length(domain)])) / step(binedges)
	end
	stairs!(ax, binedges′, diff([0.0; y; 1.0]) ./ (√2 * (2N)^(1 / 6)); color = :yellow, linewidth = 2, linestyle = :dot)

	x = range(-5, 1; length = 257)
	lines!(ax, x, pdf.(TracyWidom{4}(), x); color = Cycled(3), linewidth = 2)

	#plot!(ax, fit_gibbs_mala[1]; color = :gray, strokecolor = :gray, strokewidth = 2, linestyle = :dashdot)

	axislegend(ax, [
		[LineElement(; color = Cycled(1)), LineElement(; color = Cycled(1), points = Point2f[(0.35, 0.2), (0.35, .8)])],
		[LineElement(; color = :red, linestyle = :dash), LineElement(; color = :red, points = Point2f[(0.65, 0.2), (0.65, .8)])],
		[PolyElement(; color = Cycled(1)), PolyElement(; color = :red, points = Point2f[(1, 0), (0, 0), (1, 1)]), LineElement(; color = :yellow, linestyle = :dot, linewidth = 2)],
		LineElement(; color = Cycled(3)),
	], ["PfPP", "GSE eigmax", "Fredholm Pf", "Tracy Widom"])
	Label(fig[0, 1], "Maximum GSE eigenvalues (N = $N) sampled\nfrom PfPP and directly"; tellwidth = false, font = ax.titlefont, fontsize = 18)

	save("gse_hist.pdf", fig)

	fig
end

# ╔═╡ d83c9bdb-c1b4-44aa-9273-0e9e9b244009
fit_gibbs = tmap(1:N) do i
	#histsmoother = HistSmoother(vec(stack(j -> gibbs_data_sorted[i, 10001 * (j - 1) .+ (101:10001)], 1:16)))
	histsmoother = HistSmoother((√2 * gibbs_data_sorted[i, 101:1100] .- 2√N) .* (2N)^(1 / 6))

	BD.varinf(histsmoother)[1]
end

# ╔═╡ 1de56646-dc61-4e3a-864b-127053f29d24
let
	fig = Figure()
	ax = Axis(fig[1, 1]; limits = ((2, 5), (0, 1.1 * maximum(bincounts(normalize(hists2_mean[1]))))))

	#hist!(ax, data; bins = binedges, normalization = :pdf, color = :gray)
	plot!(ax, normalize(hists4_mean[1]); color = :lightgray)
	errorbars!(ax, hists4_errors[1]; color = :gray, linewidth = 2)	

	stairs!(ax, normalize(hists1_mean[1]))
	errorbars!(ax, hists1_errors[1] .- step(binedges) * Vec3f(.25, 0, 0); color = Cycled(1), linewidth = 2)
	stairs!(ax, normalize(hists2_mean[1]); color = :red, linewidth = 2, linestyle = :dash)
	errorbars!(ax, hists2_errors[1] .+ step(binedges) * Vec3f(.25, 0, 0); color = :red, linewidth = 2)

	x = 0:(length(domain) - 1)
	y = map(x) do k
		pfaffian(skewhermitian!(JMatrix(2(length(domain) - k)) - K[(2k + 1):2length(domain), (2k + 1):2length(domain)])) / step(binedges)
	end
	stairs!(ax, binedges, diff([0.0; y; 1.0]); color = :yellow, linewidth = 2, linestyle = :dot)

	x = range(first(domain), last(domain); length = 257)
	lines!(ax, (2√N .+ x ./ (2N)^(1 / 6)) / √2, pdf.(TracyWidom{4}(), x) .* √2 * (2N)^(1 / 6); color = Cycled(3), linewidth = 2)

	plot!(ax, fit_gibbs[1]; color = :gray, strokecolor = :gray, strokewidth = 2, linestyle = :dashdot)

	axislegend(ax, [
		[LineElement(; color = Cycled(1)), LineElement(; color = Cycled(1), points = Point2f[(0.35, 0.2), (0.35, .8)])],
		[LineElement(; color = :red, linestyle = :dash), LineElement(; color = :red, points = Point2f[(0.65, 0.2), (0.65, .8)])],
		[PolyElement(; color = Cycled(1)), PolyElement(; color = :red, points = Point2f[(1, 0), (0, 0), (1, 1)]), LineElement(; color = :yellow, linestyle = :dot, linewidth = 2)],
		LineElement(; color = Cycled(3)),
	], ["PfPP", "GSE Eigmax", "Fredholm Pf", "Tracy Widom"])
	Label(fig[0, 1], "Distribution of maximum GSE Eigenvalues (N = $N) sampled from PfPP and directly"; tellwidth = false, font = ax.titlefont)

	#save("gse_hist.pdf", fig)

	fig
end

# ╔═╡ bbb3b411-1f08-471e-b99c-3f36927e31b2
let
	fig = Figure(; size = (650, 650))
	ax = Axis(fig[1, 1]; yscale = _log10, limits = ((-5, 5), (1e-5, 5)))
	tightlimits!(ax)
	for i in 1:N
		xlims = support(BD.model(fit_gibbs[i]))
		ax′ = Axis(fig[fld1(i + 1, 2), mod1(i + 1, 2)]; limits = (xlims, (0, 2)))
		tightlimits!(ax′)

		for ax in [ax, ax′]
			plot!(ax, fit_gibbs[i]; color = Cycled(i), strokecolor = Cycled(i))
			#plot!(ax, posterior_samples_gibbs[i]; color = Cycled(i), strokecolor = Cycled(i))
		end
	end
	Legend(fig[:, 3],
		   [PolyElement(; color, strokecolor = :transparent) for color in Cycled.(1:N)],
		   ["EV $i" for i in 1:N],
	)
	fig
end

# ╔═╡ eca631ec-38fa-4740-9dd1-96df841f752b
data_hmc_sorted = sort(data_hmc.posterior_matrix; dims = 1, rev = true)

# ╔═╡ 89bda349-3a7a-46b3-ba57-62677a0c5e76
fit_hmc = tmap(1:N) do i
	histsmoother = HistSmoother((√2 * data_hmc_sorted[i, :] .- 2√N) .* (2N)^(1 / 6))
	BD.varinf(histsmoother)[1]
end

# ╔═╡ 3ce601a3-2c80-4a15-80fc-8f2091cc9d0a
let
	fig = Figure()
	ax = Axis(fig[1, 1]; limits = ((2, 5), (0, 1.1 * maximum(bincounts(normalize(hists2_mean[1]))))))
	
	x = 0:(length(domain) - 1)
	y = map(x) do k
		pfaffian(skewhermitian!(JMatrix(2(length(domain) - k)) - K[(2k + 1):2length(domain), (2k + 1):2length(domain)])) / step(binedges)
	end
	stairs!(ax, binedges, diff([0.0; y; 1.0]); color = Cycled(4), linewidth = 2, label = "Fredholm Pf")
	
	plot!(ax, fit_hmc[1]; strokewidth = 2, linestyle = :dash)
	
	plot!(ax, fit_gibbs[1]; color = Cycled(2), strokecolor = Cycled(2), strokewidth = 2, linestyle = :dot)
	
	plot!(ax, fit_gibbs_mala[1]; color = Cycled(3), strokecolor = Cycled(3), strokewidth = 2, linestyle = :dashdot)

	x = range(first(domain), last(domain); length = 257)
	#lines!(ax, (2√N .+ x ./ (2N)^(1 / 6)) / √2, pdf.(TracyWidom{4}(), x) .* √2 * (2N)^(1 / 6); color = Cycled(3), linewidth = 2)


	#=axislegend(ax, [
		[LineElement(; color = Cycled(1)), LineElement(; color = Cycled(1), points = Point2f[(0.35, 0.2), (0.35, .8)])],
		[LineElement(; color = :red, linestyle = :dash), LineElement(; color = :red, points = Point2f[(0.65, 0.2), (0.65, .8)])],
		[PolyElement(; color = Cycled(1)), PolyElement(; color = :red, points = Point2f[(1, 0), (0, 0), (1, 1)]), LineElement(; color = :yellow, linestyle = :dot, linewidth = 2)],
		LineElement(; color = Cycled(3)),
	], ["PfPP", "GSE Eigmax", "Fredholm Pf", "Tracy Widom"])=#
	Label(fig[0, 1], "Distribution of maximum GSE Eigenvalues (N = $N) sampled from PfPP and directly"; tellwidth = false, font = ax.titlefont)

	#save("gse_hist.pdf", fig)

	fig
end

# ╔═╡ b2362bbd-67c5-4913-8d8b-37ea6a12f49b
let
	fig = Figure(; size = (650, 650))
	ax = Axis(fig[1, 1]; yscale = _log10, limits = ((-5, 5), (1e-5, 5)))
	tightlimits!(ax)
	for i in 1:N
		xlims = support(BD.model(fit_hmc[i]))
		ax′ = Axis(fig[fld1(i + 1, 2), mod1(i + 1, 2)]; limits = (xlims, (0, 2)))
		tightlimits!(ax′)

		for ax in [ax, ax′]
			plot!(ax, fit_hmc[i]; color = Cycled(i), strokecolor = Cycled(i))
		end
	end
	Legend(fig[:, 3],
		   [PolyElement(; color, strokecolor = :transparent) for color in Cycled.(1:N)],
		   ["EV $i" for i in 1:N],
	)
	fig
end

# ╔═╡ 606c23f6-05fd-4162-b1de-735d050eafef
let chain = gibbs_data
	fig = Figure(; size = (650, 650))
	for i in 1:N
		ax = current_axis()
		m, n = fld1(i, 2), mod1(i, 2)
		lines(fig[m, n], chain[i, :]; axis = (; limits = ((0, size(chain, 2)), (-5, 5))), linewidth = 0.5)
		ax′ = current_axis()
		tightlimits!(ax′)
		ax !== nothing && linkaxes!(ax, ax′)
		m == cld(N, 2) || hidexdecorations!(ax′; grid = false)
		n == 1 || hideydecorations!(ax′; grid = false)
	end
	fig
end

# ╔═╡ 5e45b1c4-a317-4c33-95af-3cc4acc6367a
let chain = data_GibbsMALA
	fig = Figure(; size = (650, 650))
	for i in 1:N
		ax = current_axis()
		m, n = fld1(i, 2), mod1(i, 2)
		lines(fig[m, n], chain[i, :]; axis = (; limits = ((0, size(chain, 2)), (-5, 5))), linewidth = 0.5)
		ax′ = current_axis()
		tightlimits!(ax′)
		ax !== nothing && linkaxes!(ax, ax′)
		m == cld(N, 2) || hidexdecorations!(ax′; grid = false)
		n == 1 || hideydecorations!(ax′; grid = false)
	end
	fig
end

# ╔═╡ 2acb6e61-4558-4e14-95d2-10e98ebc12cb
nodes, weights = let (x, w) = gausslegendre(25)
	@. w *= -4 / (x^2 + 2x - 3)
	@. x = 2atanh((x + 1) / 2)
	x, w
end

# ╔═╡ 5922cf28-e9ef-4d0d-bbae-367318657522
function Kₛ_pushforward(s, nodes, weights)
    Kₛ, dKₛ = DifferentiationInterface.value_and_derivative(AutoForwardDiff(), s) do s
		n = length(nodes)
		K = Matrix{eltype(s)}(undef, 2n, 2n)
		for i in 1:n, j in 1:n
			x, y = s + nodes[i], s + nodes[j]
			w = √(weights[i] * weights[j])
			K[2i - 1, 2j - 1] = I₄(x, y) * w
			K[2i, 2j - 1] = -S₄(x, y) * w
			K[2i - 1, 2j] = S₄(y, x) * w
			K[2i, 2j] = -D₄(x, y) * w
	    end
		return K
    end
    return Kₛ, dKₛ
end

# ╔═╡ 2d40242e-5dd5-42b3-9013-6011d9eb6f23
function eigmax_pdf(s)
	Kₛ, dKₛ = Kₛ_pushforward(s, nodes, weights)
	A = skewhermitian!(JMatrix(2length(nodes)) - Kₛ)
	return -pfaffian(A) * tr(A \ dKₛ) / 2
end

# ╔═╡ 8d97564e-98b5-4fbe-8467-4c6132e22914
begin
	vals3 = Matrix{Float64}(undef, N, 158416)
	@tasks for i in 1:158416
		λ = √N * eigvals(rand(GaussianHermite(4), N))[1:2:end]
		@view(vals3[:, i]) .= reverse(λ)
	end
	vals3
end

# ╔═╡ a80df87d-07b4-453f-8a58-93b86b87c52f
fit_gse = tmap(1:N) do i
	histsmoother = HistSmoother((√2 * vals3[i, 1:1000] .- 2√N) .* (2N)^(1 / 6))
	BD.varinf(histsmoother)[1]
end

# ╔═╡ 03b05d6c-8f49-4a16-872b-3e2062366bfa
fit_rpchol = tmap(1:N) do i
	histsmoother = HistSmoother((√2 * data_rpchol[N - i + 1, :] .- 2√N) .* (2N)^(1 / 6))
	BD.varinf(histsmoother)[1]
end

# ╔═╡ fe654ee0-162a-4ab8-abec-cdfd38a7c1d2
CairoMakie.activate!()

# ╔═╡ 4a8919e4-449c-415e-97bc-d2c09d041da7
let
	fig = Figure(; size = (500, 600))

	s = range(-5, 1; length = 257)
	pdf = eigmax_pdf.((2√N .+ s ./ (2N)^(1 / 6)) ./ √2) ./ (√2 * (2N)^(1 / 6))
	local p1, p2
	for (i, fit) in enumerate([fit_gse[1], fit_rpchol[1], fit_gibbs[1], fit_gibbs_mala[1], fit_hmc[1]])
		ax = Axis(fig[fld1(i, 2), mod1(i, 2)];
			title = ["Sampled Directly", "RPCholesky", "Slice-within-Gibbs", "MALA-within-Gibbs", "HMC"][i],
			limits = ((-5, 1), (0, 0.65)), yticks = 0:0.2:0.6
		)
		
		p1 = lines!(ax, s, pdf; color = Cycled(1), linewidth = 2, linestyle = :dash)
	
		p2 = plot!(ax, fit; color = Cycled(2), strokecolor = Cycled(2), strokewidth = 2, linestyle = :dot)
	end

	Legend(fig[3, 2], [p1, p2], ["Fredholm Pf", "Estimated Density"], "Legend"; tellwidth = false)
	Label(fig[0, :], "Maximum GSE eigenvalues (N = $N) sampled\ncontinuously"; tellwidth = false, font = current_axis().titlefont, fontsize = 18)
	#Label(fig[3, :], "* Only 1000 samples were used for HMC since samples took longer to generate"; tellwidth = false, fontsize = 10, halign = :right)

	save("gse_continuous.pdf", fig)

	fig
end

# ╔═╡ a4de208b-2bfb-4235-a2a9-e409a5383cb1
let
	fig = Figure(; size = (500, 600))

	s = range(-5, 1; length = 257)
	pdf = eigmax_pdf.((2√N .+ s ./ (2N)^(1 / 6)) ./ √2) ./ (√2 * (2N)^(1 / 6))
	local p1, p2
	for (i, fit) in enumerate([fit_gse[1], fit_rpchol[1], fit_gibbs[1], fit_gibbs_mala[1], fit_hmc[1]])
		ax = Axis(fig[fld1(i, 2), mod1(i, 2)];
			title = ["Sampled Directly", "RPCholesky", "Slice-within-Gibbs", "MALA-within-Gibbs", "HMC"][i],
			limits = ((-5, 1), (1e-6, 1)), yscale = log10,
		)
		
		p1 = lines!(ax, s, pdf; color = Cycled(1), linewidth = 2, linestyle = :dash)
	
		p2 = plot!(ax, fit; color = Cycled(2), strokecolor = Cycled(2), strokewidth = 2, linestyle = :dot)
	end

	Legend(fig[3, 2], [p1, p2], ["Fredholm Pf", "Estimated Density"], "Legend"; tellwidth = false)
	Label(fig[0, :], "Distribution of maximum GSE Eigenvalues (N = $N)\nsampled continuously"; tellwidth = false, font = current_axis().titlefont)

	fig
end

# ╔═╡ 4d35d207-a4b8-43c6-8fba-7beb82456750
eigen(f)

# ╔═╡ a4b2315b-3b60-41b1-ae34-441c9365fc40
Y′ = let (λ, Q) = eigen(f; sortby = abs)
	real.(Q[:, isapprox.(λ, 1; rtol = 1e-4)])
end

# ╔═╡ 40b78395-9ee6-47f8-b6df-80053c6b532e
let (λ, Q) = eigen(f; sortby = abs)
	global Z′ = inv(Q)[isapprox.(λ, 1; rtol = 1e-4), :]
	Q * Diagonal(λ) / Q, f, Y′ * Z′
end

# ╔═╡ ec024d67-3212-47ed-89c2-7bff802a9786
let
	Z′ = Y′ \ f
	#randPfPPproj(Y′, Z′)
	f, Y′ * Z′
end

# ╔═╡ 0ab944fa-4bdb-4057-ad90-ae170fc815a7
let (U, S, V) = svd(f)
	Q1 = qr(U[81:82, 1:2N]').Q[:, 3:end]
	Q2 = qr(V[81:82, 1:2N]').Q[:, 3:end]
	U[81:82, :], (U[:, 1:2N] * Q1)[81:82, :], V[81:82, :], (V[:, 1:2N] * Q2)[81:82, :], tr(U[:, 1:2N] * Q1 * Diagonal(S[3:2N]) * Q2' * V[:, 1:2N]')
end

# ╔═╡ fedb8d40-b3ca-496a-a283-f6fac1221efb
function sr(A::AbstractMatrix{T}, alg = ModifiedSymplecticGramSchmidtIR()) where {T}
	S = SymplecticBasis(Vector{T}[])
	R = similar(A, size(A, 2), size(A, 2))
	for i in axes(A, 2)
		x, β = skeworthonormalize!!(A[:, i], S, view(R, 1:(i - 1), i), alg)
		R[i, i] = β
		push!(S.basis, x)
	end
	return (; S = stack(S), R = UpperTriangular(R))
end

# ╔═╡ 6199d1b0-b438-4e31-ad22-6cb462766e86
import Serialization

# ╔═╡ 0f47c966-4195-4c13-9379-42cf428a12e2
# ╠═╡ disabled = true
#=╠═╡
Serialization.serialize("kernel_gse.bin", K)
  ╠═╡ =#

# ╔═╡ 254bc018-1632-4960-a70d-80f908c1533d
function randPfPPproj(Y, Z)
    two_N, two_n = size(Y)
    N = div(two_N, 2)
    n = div(two_n, 2)
    
    𝓘 = Vector{Int}(undef, n)
    
    @views for k in 1:n
        # 1. Compute marginal probabilities for each item
		p = dot.(eachrow(Y)[1:2:end], eachcol(Z)[1:2:end])
        
        # Normalize and sample (max is used to scrub tiny negative floating point errors)
        p = max.(p, 0.0)
        𝓘[k] = rand(Distributions.Categorical(p ./ (n - k + 1)))
        i = 𝓘[k]
		
		# 2. QR/LQ-based Deflation
        # Instead of generic nullspace(), we use the structure of Y_i and Z_i
        # LQ on Y_i gives us a 2x2L matrix and a unitary QY. 
        # The last (2n-2) rows of QY are the nullspace of Y_i.
        QY = qr(Y[2i-1:2i, :]').Q
        
        # QR on Z_i gives us a 2Rx2 matrix and a unitary QZ.
        # The last (2n-2) columns of QZ are the nullspace of Z_i'.
        QZ = qr(Z[:, 2i-1:2i]).Q
        
        # 3. Extract the bases (the "2:end" equivalent)
        # For rank-2 reduction, we take everything from index 3 onwards
        NY = QY[:, 3:end]
        NZ = QZ[:, 3:end]
        
        # 4. Update
        # We still need the oblique correction (C) because Y and Z 
        # aren't necessarily orthogonal to each other, just internally structured.
        C = NZ' * NY
        Y = Y * NY
        Z = (C \ NZ') * Z
    end
    
    return sort!(𝓘)
end

# ╔═╡ 7da45d2c-1038-46ba-bbec-99a60fd5ad18
randPfPPproj(real.(Y′), real.(Z′))

# ╔═╡ 0fd9d389-534b-47ad-b79e-44a78e7c4a5a
real.(Z′) * real.(Y′)

# ╔═╡ bd09a846-afe4-44a3-93b9-0c9678f096aa
begin
	hists5 = [Hist1D(; counttype = Int, binedges = binedges′) for _ in 1:N, _ in 1:50]
	let
		Y, Z = let (_Q, R) = sr(Y′)
			Q = _Q[:, 1:2N] |> real
			Q, -JMatrix(20) * Q' * JMatrix(160)
		end#real.(Y′), real.(Z′)
		@tasks for _ in 1:10000
			for i in 1:50
				h = randPfPPproj(Y, Z)
				atomic_push!.(@view(hists5[:, i]), (√2 * domain[reverse(h)] .- 2√N) .* (2N)^(1 / 6))
			end
		end
	end
	hists5_mean = map(1:N) do i
		c = stack(bincounts.(@view(hists5[i, :])))
		m = mean(c; dims = 2)
		Hist1D(; binedges = binedges′, bincounts = vec(m))
	end
	hists5_errors = map(1:N) do i
		c = stack(bincounts.(normalize.(@view(hists5[i, :]))))
		m = mean(c; dims = 2)
		s = std(c; dims = 2)
		Vec3f.(domain′, vec(m), vec(s))
	end
end

# ╔═╡ 4d9d74bf-d9e3-4490-84f1-79011ad27d1d
let (hists1_mean, hists1_errors) = (hists5_mean, hists5_errors)
	fig = Figure(; size = (500, 330))
	ax = Axis(fig[1, 1]; limits = ((-5, 1), (0, 1.1 * maximum(bincounts(normalize(hists2_mean[1]))))))

	#hist!(ax, data; bins = binedges, normalization = :pdf, color = :gray)
	#plot!(ax, normalize(hists3_mean[1]); color = :lightgray)
	#errorbars!(ax, hists3_errors[1]; color = :gray, linewidth = 2)	

	stairs!(ax, normalize(hists1_mean[1]))
	errorbars!(ax, hists1_errors[1] .- step(binedges′) * Vec3f(.15, 0, 0); color = Cycled(1), linewidth = 2)
	stairs!(ax, normalize(hists2_mean[1]); color = :red, linewidth = 2, linestyle = :dash)
	errorbars!(ax, hists2_errors[1] .+ step(binedges′) * Vec3f(.15, 0, 0); color = :red, linewidth = 2)

	x = 0:(length(domain) - 1)
	y = map(x) do k
		pfaffian(skewhermitian!(JMatrix(2(length(domain) - k)) - K[(2k + 1):2length(domain), (2k + 1):2length(domain)])) / step(binedges)
	end
	stairs!(ax, binedges′, diff([0.0; y; 1.0]) ./ (√2 * (2N)^(1 / 6)); color = :yellow, linewidth = 2, linestyle = :dot)

	x = range(-5, 1; length = 257)
	lines!(ax, x, pdf.(TracyWidom{4}(), x); color = Cycled(3), linewidth = 2)

	#plot!(ax, fit_gibbs_mala[1]; color = :gray, strokecolor = :gray, strokewidth = 2, linestyle = :dashdot)

	axislegend(ax, [
		[LineElement(; color = Cycled(1)), LineElement(; color = Cycled(1), points = Point2f[(0.35, 0.2), (0.35, .8)])],
		[LineElement(; color = :red, linestyle = :dash), LineElement(; color = :red, points = Point2f[(0.65, 0.2), (0.65, .8)])],
		[PolyElement(; color = Cycled(1)), PolyElement(; color = :red, points = Point2f[(1, 0), (0, 0), (1, 1)]), LineElement(; color = :yellow, linestyle = :dot, linewidth = 2)],
		LineElement(; color = Cycled(3)),
	], ["PfPP", "GSE eigmax", "Fredholm Pf", "Tracy Widom"])
	Label(fig[0, 1], "Maximum GSE eigenvalues (N = $N) sampled from PfPP and directly"; tellwidth = false, font = ax.titlefont)

	save("gse_hist.pdf", fig)

	fig
end

# ╔═╡ e3338264-be74-440b-b099-0f0d12daf383
let (_Q, R) = sr(Y′)
	Q = _Q[:, 1:2N]
	Q' * JMatrix(160) * Q, (R * Z′) * JMatrix(160) * (R * Z′)'
	Q, (-JMatrix(20) * real(R * Z′) * JMatrix(160))'
end

# ╔═╡ 798a73e9-1d84-4f7e-b2d1-9c89faea4c15
let (_Q, R) = sr(Y′)
	Q = _Q[:, 1:2N] |> real
	randPfPPproj(Q, -JMatrix(20) * Q' * JMatrix(160))
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
ARS = "5744327c-528e-406a-a16f-961c23f1e1d8"
ApproxFun = "28f2ccd6-bb30-5033-b560-165f7b14dc2f"
BayesDensityHistSmoother = "4ec5b509-5e49-418c-a128-3e7a20b6d93e"
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
Combinatorics = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
DifferentiationInterface = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
DynamicHMC = "bbc10e6e-7c05-544b-b16e-64fede858acb"
FHist = "68837c9b-b678-4cd5-9925-8a54edc8f695"
FastGaussQuadrature = "442a2c76-b920-505d-bb47-c5924d526838"
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
FredholmDeterminants = "807c80a6-c809-4266-8359-c14a54b3d3b7"
Integrals = "de52edbc-65ea-441a-8357-d3a637375a31"
KrylovKit = "0b1a1467-8014-51b9-945f-bf0ae24f4b77"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LogDensityProblems = "6fdf6af0-433a-55f7-b3ed-c6c6e0b8df7c"
LogDensityProblemsAD = "996a588d-648d-4e1f-a8f0-a84b347e47b1"
MathTeXEngine = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
OhMyThreads = "67456a42-1dca-4109-a031-0a68de7e3ad5"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
RandomMatrices = "2576dda1-a324-5b11-aa66-c48ed7e3c618"
Revise = "295af30f-e4ad-537b-8983-00126c2a3abe"
Serialization = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
SkewLinearAlgebra = "5c889d49-8c60-4500-9d10-5d3a22e2f4b9"
SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
VectorInterface = "409d34a3-91d5-4945-b6ec-7529ddf182d8"
WGLMakie = "276b4fcb-3e11-5398-bf8b-a0c2d153d008"

[compat]
ApproxFun = "~0.13.28"
BayesDensityHistSmoother = "~0.2.0"
BenchmarkTools = "~1.6.3"
CairoMakie = "~0.15.9"
Combinatorics = "~1.1.0"
DifferentiationInterface = "~0.7.16"
Distributions = "~0.25.124"
DynamicHMC = "~3.6.0"
FHist = "~0.11.17"
FastGaussQuadrature = "~1.1.0"
ForwardDiff = "~1.3.3"
Integrals = "~5.4.0"
KrylovKit = "~0.10.2"
LogDensityProblems = "~2.2.0"
LogDensityProblemsAD = "~1.13.1"
MathTeXEngine = "~0.6.7"
Mooncake = "~0.5.17"
OhMyThreads = "~0.8.5"
RandomMatrices = "~0.5.6"
Revise = "~3.14.2"
SkewLinearAlgebra = "~1.1.0"
SpecialFunctions = "~2.7.2"
VectorInterface = "~0.5.0"
WGLMakie = "~0.13.9"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.6"
manifest_format = "2.0"
project_hash = "af5fd7666ab333a1b11a7fdbfa797315e9c825a3"

[[deps.ADTypes]]
git-tree-sha1 = "f7304359109c768cf32dc5fa2d371565bb63b68a"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "1.21.0"

    [deps.ADTypes.extensions]
    ADTypesChainRulesCoreExt = "ChainRulesCore"
    ADTypesConstructionBaseExt = "ConstructionBase"
    ADTypesEnzymeCoreExt = "EnzymeCore"

    [deps.ADTypes.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"

[[deps.ARS]]
deps = ["Compat", "DifferentiationInterface", "DocStringExtensions", "ForwardDiff", "Random", "SpecialFunctions", "StatsBase"]
git-tree-sha1 = "853b75089a44d9109a3a64729fef4e20ff46b66b"
repo-rev = "main"
repo-url = "https://github.com/Eliassj/ARS.jl"
uuid = "5744327c-528e-406a-a16f-961c23f1e1d8"
version = "1.0.2-DEV"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "Dates", "InverseFunctions", "MacroTools"]
git-tree-sha1 = "2eeb2c9bef11013efc6f8f97f32ee59b146b09fb"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.44"

    [deps.Accessors.extensions]
    AxisKeysExt = "AxisKeys"
    IntervalSetsExt = "IntervalSets"
    LinearAlgebraExt = "LinearAlgebra"
    StaticArraysExt = "StaticArrays"
    StructArraysExt = "StructArrays"
    TestExt = "Test"
    UnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "0761717147821d696c9470a7a86364b2fbd22fd8"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.5.2"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AdaptivePredicates]]
git-tree-sha1 = "7e651ea8d262d2d74ce75fdf47c4d63c07dba7a6"
uuid = "35492f91-a3bd-45ad-95db-fcad7dcfedb7"
version = "1.2.0"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.Animations]]
deps = ["Colors"]
git-tree-sha1 = "e092fa223bf66a3c41f9c022bd074d916dc303e7"
uuid = "27a7e980-b3e6-11e9-2bcd-0b925532e340"
version = "0.4.2"

[[deps.ApproxFun]]
deps = ["AbstractFFTs", "ApproxFunBase", "ApproxFunFourier", "ApproxFunOrthogonalPolynomials", "ApproxFunSingularities", "Calculus", "DomainSets", "FastTransforms", "LinearAlgebra", "RecipesBase", "Reexport", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "aaa2506645c1b7e1e7fdcf6182852de53d7a1bab"
uuid = "28f2ccd6-bb30-5033-b560-165f7b14dc2f"
version = "0.13.28"

    [deps.ApproxFun.extensions]
    ApproxFunDualNumbersExt = "DualNumbers"

    [deps.ApproxFun.weakdeps]
    DualNumbers = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"

[[deps.ApproxFunBase]]
deps = ["AbstractFFTs", "BandedMatrices", "BlockArrays", "BlockBandedMatrices", "Calculus", "Combinatorics", "DSP", "DomainSets", "FFTW", "FillArrays", "InfiniteArrays", "IntervalSets", "LazyArrays", "LinearAlgebra", "LowRankMatrices", "SparseArrays", "SpecialFunctions", "StaticArrays", "Statistics"]
git-tree-sha1 = "6019585657a7f2a1c2e51b5e7bfe126721c1265b"
uuid = "fbd15aa5-315a-5a7d-a8a4-24992e37be05"
version = "0.9.33"

    [deps.ApproxFunBase.extensions]
    ApproxFunBaseDualNumbersExt = "DualNumbers"
    ApproxFunBaseTestExt = "Test"

    [deps.ApproxFunBase.weakdeps]
    DualNumbers = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.ApproxFunFourier]]
deps = ["AbstractFFTs", "ApproxFunBase", "BandedMatrices", "DomainSets", "FFTW", "FastTransforms", "InfiniteArrays", "IntervalSets", "LinearAlgebra", "Reexport", "StaticArrays"]
git-tree-sha1 = "0cb3a0f77266ae5e13f5ac155dc2082b7aea47c9"
uuid = "59844689-9c9d-51bf-9583-5b794ec66d30"
version = "0.3.31"

[[deps.ApproxFunOrthogonalPolynomials]]
deps = ["ApproxFunBase", "BandedMatrices", "BlockArrays", "BlockBandedMatrices", "DomainSets", "FastGaussQuadrature", "FastTransforms", "FillArrays", "HalfIntegers", "IntervalSets", "LinearAlgebra", "OddEvenIntegers", "Reexport", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "a9e92363a191b5e0407f1e93ca9aad25a96bb130"
uuid = "b70543e2-c0d9-56b8-a290-0d4d6d4de211"
version = "0.6.62"
weakdeps = ["Polynomials", "Static"]

    [deps.ApproxFunOrthogonalPolynomials.extensions]
    ApproxFunOrthogonalPolynomialsPolynomialsExt = "Polynomials"
    ApproxFunOrthogonalPolynomialsStaticExt = "Static"

[[deps.ApproxFunSingularities]]
deps = ["ApproxFunBase", "ApproxFunOrthogonalPolynomials", "BlockBandedMatrices", "DomainSets", "HalfIntegers", "IntervalSets", "LinearAlgebra", "OddEvenIntegers", "Reexport", "SpecialFunctions"]
git-tree-sha1 = "2589e9b66749b21dc138e14b3ae8e7fe790b0491"
uuid = "f8fcb915-6b99-5be2-b79a-d6dbef8e6e7e"
version = "0.3.22"
weakdeps = ["StaticArrays"]

    [deps.ApproxFunSingularities.extensions]
    ApproxFunSingularitiesStaticArraysExt = "StaticArrays"

[[deps.ArgCheck]]
git-tree-sha1 = "f9e9a66c9b7be1ad7372bbd9b062d9230c30c5ce"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.5.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "d57bd3762d308bded22c3b82d033bff85f6195c6"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.4.0"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "54f895554d05c83e3dd59f6a396671dae8999573"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.24.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceAMDGPUExt = "AMDGPU"
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = ["CUDSS", "CUDA"]
    ArrayInterfaceChainRulesCoreExt = "ChainRulesCore"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceMetalExt = "Metal"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceSparseArraysExt = "SparseArrays"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.ArrayLayouts]]
deps = ["FillArrays", "LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "e0b47732a192dd59b9d079a06d04235e2f833963"
uuid = "4c555306-a7a7-4459-81d9-ec55ddd5c99a"
version = "1.12.2"
weakdeps = ["SparseArrays"]

    [deps.ArrayLayouts.extensions]
    ArrayLayoutsSparseArraysExt = "SparseArrays"

[[deps.Arrow]]
deps = ["ArrowTypes", "BitIntegers", "CodecLz4", "CodecZstd", "ConcurrentUtilities", "DataAPI", "Dates", "EnumX", "Mmap", "PooledArrays", "SentinelArrays", "StringViews", "Tables", "TimeZones", "TranscodingStreams", "UUIDs"]
git-tree-sha1 = "4a69a3eadc1f7da78d950d1ef270c3a62c1f7e01"
uuid = "69666777-d1a9-59fb-9406-91d4454c9d45"
version = "2.8.1"

[[deps.ArrowTypes]]
deps = ["Sockets", "UUIDs"]
git-tree-sha1 = "404265cd8128a2515a81d5eae16de90fdef05101"
uuid = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
version = "2.3.0"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Automa]]
deps = ["PrecompileTools", "SIMD", "TranscodingStreams"]
git-tree-sha1 = "a8f503e8e1a5f583fbef15a8440c8c7e32185df2"
uuid = "67c07d97-cdcb-5c2c-af73-a7f9c32a568b"
version = "1.1.0"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "4126b08903b777c88edf1754288144a0492c05ad"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.8"

[[deps.BSplineKit]]
deps = ["ArrayLayouts", "BandedMatrices", "FastGaussQuadrature", "ForwardDiff", "LinearAlgebra", "PrecompileTools", "Random", "Reexport", "SparseArrays", "Static", "StaticArrays", "StaticArraysCore", "StatsAPI"]
git-tree-sha1 = "02d491054afeb89b7f34331701e4474eb0b904f7"
uuid = "093aae92-e908-43d7-9660-e50ee39d5a0a"
version = "0.19.2"

[[deps.BandedMatrices]]
deps = ["ArrayLayouts", "FillArrays", "LinearAlgebra", "PrecompileTools"]
git-tree-sha1 = "02fa77c70ba84361b9bc9ff28523bd9d78519265"
uuid = "aae01518-5342-5314-be14-df237901396f"
version = "1.11.0"

    [deps.BandedMatrices.extensions]
    BandedMatricesSparseArraysExt = "SparseArrays"
    CliqueTreesExt = "CliqueTrees"

    [deps.BandedMatrices.weakdeps]
    CliqueTrees = "60701a23-6482-424a-84db-faee86b9b1f8"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.BangBang]]
deps = ["Accessors", "ConstructionBase", "InitialValues", "LinearAlgebra"]
git-tree-sha1 = "cceb62468025be98d42a5dc581b163c20896b040"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.4.9"
weakdeps = ["ChainRulesCore", "DataFrames", "StaticArrays", "StructArrays", "Tables", "TypedTables"]

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTablesExt = "Tables"
    BangBangTypedTablesExt = "TypedTables"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BaseDirs]]
git-tree-sha1 = "bca794632b8a9bbe159d56bf9e31c422671b35e0"
uuid = "18cc8868-cbac-4acf-b575-c8ff214dc66f"
version = "1.3.2"

[[deps.BayesDensityCore]]
deps = ["Distributions", "Random", "StatsBase"]
git-tree-sha1 = "40e8bec09eef07e3eba58a266f9025b7123c9eb4"
uuid = "93712312-7690-42d2-80ab-295e38cb0be8"
version = "0.2.0"

    [deps.BayesDensityCore.extensions]
    BayesDensityCoreMakieExt = "Makie"
    BayesDensityCorePlotsExt = "Plots"

    [deps.BayesDensityCore.weakdeps]
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"

[[deps.BayesDensityHistSmoother]]
deps = ["BSplineKit", "BayesDensityCore", "DataFrames", "Distributions", "LinearAlgebra", "Logging", "MixedModels", "Random", "Reexport", "SparseArrays", "SpecialFunctions", "StatsBase"]
git-tree-sha1 = "e813c6a31902732ccf0bf21d6dc3169e72ada350"
uuid = "4ec5b509-5e49-418c-a128-3e7a20b6d93e"
version = "0.2.0"

[[deps.BayesHistogram]]
git-tree-sha1 = "5d5dda960067751bc1534aba765f771325044501"
uuid = "000d9b38-65fe-4c81-bdb9-69f01f102479"
version = "1.0.7"

[[deps.BenchmarkTools]]
deps = ["Compat", "JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "7fecfb1123b8d0232218e2da0c213004ff15358d"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.6.3"

[[deps.Bessels]]
git-tree-sha1 = "4435559dc39793d53a9e3d278e185e920b4619ef"
uuid = "0e736298-9ec6-45e8-9647-e4fc86a2fe38"
version = "0.2.8"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.BitIntegers]]
deps = ["Random"]
git-tree-sha1 = "091d591a060e43df1dd35faab3ca284925c48e46"
uuid = "c3b6d118-76ef-56ca-8cc7-ebb389d030a1"
version = "0.3.7"

[[deps.BlockArrays]]
deps = ["ArrayLayouts", "FillArrays", "LinearAlgebra"]
git-tree-sha1 = "0f606a9894e2bcda541ceb82a91a13c5d450ed97"
uuid = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
version = "1.9.3"
weakdeps = ["Adapt", "BandedMatrices"]

    [deps.BlockArrays.extensions]
    BlockArraysAdaptExt = "Adapt"
    BlockArraysBandedMatricesExt = "BandedMatrices"

[[deps.BlockBandedMatrices]]
deps = ["ArrayLayouts", "BandedMatrices", "BlockArrays", "FillArrays", "LinearAlgebra", "MatrixFactorizations"]
git-tree-sha1 = "4eef2d2793002ef8221fe561cc822eb252afa72f"
uuid = "ffab5731-97b5-5995-9138-79e8c1846df0"
version = "0.13.4"
weakdeps = ["SparseArrays"]

    [deps.BlockBandedMatrices.extensions]
    BlockBandedMatricesSparseArraysExt = "SparseArrays"

[[deps.Bonito]]
deps = ["Base64", "CodecZlib", "Colors", "Dates", "Deno_jll", "HTTP", "Hyperscript", "JSON", "LinearAlgebra", "Markdown", "MbedTLS", "MsgPack", "Observables", "OrderedCollections", "Random", "RelocatableFolders", "SHA", "Sockets", "Tables", "ThreadPools", "URIs", "UUIDs", "WidgetsBase"]
git-tree-sha1 = "bb43f72801f703ad3c66833bd02b8f54c7328238"
uuid = "824d6782-a2ef-11e9-3a09-e5662e0c26f8"
version = "4.2.0"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CRC32c]]
uuid = "8bf52ea8-c179-5cab-976a-9e18b702a9bc"
version = "1.11.0"

[[deps.CRlibm]]
deps = ["CRlibm_jll"]
git-tree-sha1 = "66188d9d103b92b6cd705214242e27f5737a1e5e"
uuid = "96374032-68de-5a5b-8d9e-752f78720389"
version = "1.0.2"

[[deps.CRlibm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e329286945d0cfc04456972ea732551869af1cfc"
uuid = "4e9b3aee-d8a1-5a3d-ad8b-7d824db253f0"
version = "1.0.1+0"

[[deps.Cairo]]
deps = ["Cairo_jll", "Colors", "Glib_jll", "Graphics", "Libdl", "Pango_jll"]
git-tree-sha1 = "71aa551c5c33f1a4415867fe06b7844faadb0ae9"
uuid = "159f3aea-2a34-519c-b102-8c37f9878175"
version = "1.1.1"

[[deps.CairoMakie]]
deps = ["CRC32c", "Cairo", "Cairo_jll", "Colors", "FileIO", "FreeType", "GeometryBasics", "LinearAlgebra", "Makie", "PrecompileTools"]
git-tree-sha1 = "fa072933899aae6dc61dde934febed8254e66c6a"
uuid = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
version = "0.15.9"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "d0efe2c6fdcdaa1c161d206aa8b933788397ec71"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.6+0"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9cb23bbb1127eefb022b022481466c0f1127d430"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.2"

[[deps.ChainRules]]
deps = ["Adapt", "ChainRulesCore", "Compat", "Distributed", "GPUArraysCore", "IrrationalConstants", "LinearAlgebra", "Random", "RealDot", "SparseArrays", "SparseInverseSubset", "Statistics", "StructArrays", "SuiteSparse"]
git-tree-sha1 = "3c190c570fb3108c09f838607386d10c71701789"
uuid = "082447d4-558c-5d27-93f4-14fc19e9eca2"
version = "1.73.0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "12177ad6b3cad7fd50c8b3825ce24a99ad61c18f"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.26.1"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.ChunkSplitters]]
git-tree-sha1 = "1c52c8e2673edc030191177ff1aee42d25149acb"
uuid = "ae650224-84b6-46f8-82ea-d812ca08434e"
version = "3.2.0"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "REPL", "UUIDs"]
git-tree-sha1 = "cfb7a2e89e245a9d5016b70323db412b3a7438d5"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "3.0.2"

[[deps.CodecLz4]]
deps = ["Lz4_jll", "TranscodingStreams"]
git-tree-sha1 = "d58afcd2833601636b48ee8cbeb2edcb086522c2"
uuid = "5ba52731-8f18-5e0d-9241-30f10d1ec561"
version = "0.4.6"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.CodecZstd]]
deps = ["TranscodingStreams", "Zstd_jll"]
git-tree-sha1 = "da54a6cd93c54950c15adf1d336cfd7d71f51a56"
uuid = "6b39b394-51ab-5f42-8807-6242bab2b4c2"
version = "0.8.7"

[[deps.ColorBrewer]]
deps = ["Colors", "JSON"]
git-tree-sha1 = "07da79661b919001e6863b81fc572497daa58349"
uuid = "a2cac450-b92f-5266-8821-25eda20663c8"
version = "0.4.2"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

[[deps.Combinatorics]]
git-tree-sha1 = "c761b00e7755700f9cdf5b02039939d1359330e1"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.1.0"

[[deps.CommonSolve]]
git-tree-sha1 = "78ea4ddbcf9c241827e7035c3a03e2e456711470"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.6"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

[[deps.CommonWorldInvalidations]]
git-tree-sha1 = "ae52d1c52048455e85a387fbee9be553ec2b68d0"
uuid = "f70d9fcc-98c5-4d4a-abd7-e4cdeebd8ca8"
version = "1.0.0"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.Compiler]]
git-tree-sha1 = "382d79bfe72a406294faca39ef0c3cef6e6ce1f1"
uuid = "807dbc54-b67e-4c79-8afb-eafe4df6f2e1"
version = "0.1.1"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.CompositeTypes]]
git-tree-sha1 = "bce26c3dab336582805503bed209faab1c279768"
uuid = "b152e2b5-7a66-4b01-a709-34e65c35f657"
version = "0.1.4"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ComputePipeline]]
deps = ["Observables", "Preferences"]
git-tree-sha1 = "3b4be73db165146d8a88e47924f464e55ab053cd"
uuid = "95dc2771-c249-4cd0-9c9f-1f3b4330693c"
version = "0.1.7"

[[deps.ConcreteStructs]]
git-tree-sha1 = "f749037478283d372048690eb3b5f92a79432b34"
uuid = "2569d6c7-a4a2-43d3-a901-331e8e4be471"
version = "0.2.3"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "21d088c496ea22914fe80906eb5bce65755e5ec8"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.5.1"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"
weakdeps = ["IntervalSets", "LinearAlgebra", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.CoreMath]]
deps = ["CoreMath_jll"]
git-tree-sha1 = "8c0480f92b1b1796239156a1b9b1bfb1b39499b4"
uuid = "b7a15901-be09-4a0e-87d2-2e66b0e09b5a"
version = "0.1.0"

[[deps.CoreMath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a692a4c1dc59a4b8bc0b6403876eb3250fde2bc3"
uuid = "a38c48d9-6df1-5ac9-9223-b6ada3b5572b"
version = "0.1.0+0"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DSP]]
deps = ["Bessels", "FFTW", "IterTools", "LinearAlgebra", "Polynomials", "Random", "Reexport", "SpecialFunctions", "Statistics"]
git-tree-sha1 = "5989debfc3b38f736e69724818210c67ffee4352"
uuid = "717857b8-e6f2-59f4-9121-6e50c889abd2"
version = "0.8.4"
weakdeps = ["OffsetArrays"]

    [deps.DSP.extensions]
    OffsetArraysExt = "OffsetArrays"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "d8928e9169ff76c6281f39a659f9bca3a573f24c"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.8.1"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "e86f4a2805f7f19bec5129bc9150c38208e5dc23"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.4"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DelaunayTriangulation]]
deps = ["AdaptivePredicates", "EnumX", "ExactPredicates", "Random"]
git-tree-sha1 = "c55f5a9fd67bdbc8e089b5a3111fe4292986a8e8"
uuid = "927a84f5-c5f4-47a5-9785-b46e178433df"
version = "1.6.6"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Deno_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cd6756e833c377e0ce9cd63fb97689a255f12323"
uuid = "04572ae6-984a-583e-9378-9577a1c2574d"
version = "1.33.4+0"

[[deps.Dictionaries]]
deps = ["Indexing", "Random", "Serialization"]
git-tree-sha1 = "a55766a9c8f66cf19ffcdbdb1444e249bb4ace33"
uuid = "85a47980-9c8c-11e8-2b9f-f7ca1fa99fb4"
version = "0.4.6"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.DifferentiationInterface]]
deps = ["ADTypes", "LinearAlgebra"]
git-tree-sha1 = "7ae99144ea44715402c6c882bfef2adbeadbc4ce"
uuid = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63"
version = "0.7.16"

    [deps.DifferentiationInterface.extensions]
    DifferentiationInterfaceChainRulesCoreExt = "ChainRulesCore"
    DifferentiationInterfaceDiffractorExt = "Diffractor"
    DifferentiationInterfaceEnzymeExt = ["EnzymeCore", "Enzyme"]
    DifferentiationInterfaceFastDifferentiationExt = "FastDifferentiation"
    DifferentiationInterfaceFiniteDiffExt = "FiniteDiff"
    DifferentiationInterfaceFiniteDifferencesExt = "FiniteDifferences"
    DifferentiationInterfaceForwardDiffExt = ["ForwardDiff", "DiffResults"]
    DifferentiationInterfaceGPUArraysCoreExt = "GPUArraysCore"
    DifferentiationInterfaceGTPSAExt = "GTPSA"
    DifferentiationInterfaceMooncakeExt = "Mooncake"
    DifferentiationInterfacePolyesterForwardDiffExt = ["PolyesterForwardDiff", "ForwardDiff", "DiffResults"]
    DifferentiationInterfaceReverseDiffExt = ["ReverseDiff", "DiffResults"]
    DifferentiationInterfaceSparseArraysExt = "SparseArrays"
    DifferentiationInterfaceSparseConnectivityTracerExt = "SparseConnectivityTracer"
    DifferentiationInterfaceSparseMatrixColoringsExt = "SparseMatrixColorings"
    DifferentiationInterfaceStaticArraysExt = "StaticArrays"
    DifferentiationInterfaceSymbolicsExt = "Symbolics"
    DifferentiationInterfaceTrackerExt = "Tracker"
    DifferentiationInterfaceZygoteExt = ["Zygote", "ForwardDiff"]

    [deps.DifferentiationInterface.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DiffResults = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
    Diffractor = "9f5e2b26-1114-432f-b630-d3fe2085c51c"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FastDifferentiation = "eb9bf01b-bf85-4b60-bf87-ee5de06c00be"
    FiniteDiff = "6a86dc24-6348-571c-b903-95158fe2bd41"
    FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    GTPSA = "b27dd330-f138-47c5-815b-40db9dd9b6e8"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    PolyesterForwardDiff = "98d1487c-24ca-40b6-b7ab-df2af84e126b"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SparseConnectivityTracer = "9f842d2f-2579-4b1d-911e-f412cf18a3f5"
    SparseMatrixColorings = "0a514795-09f3-496d-8182-132a7b665d35"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.DispatchDoctor]]
deps = ["MacroTools", "Preferences"]
git-tree-sha1 = "42cd00edaac86f941815fe557c1d01e11913e07c"
uuid = "8d63f2c5-f18a-4cf2-ba9d-b3f60fc568c8"
version = "0.4.28"

    [deps.DispatchDoctor.extensions]
    DispatchDoctorChainRulesCoreExt = "ChainRulesCore"
    DispatchDoctorEnzymeCoreExt = "EnzymeCore"

    [deps.DispatchDoctor.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "12184a8cf11c7cbd90a4db8b2cb2f7b6f057cc46"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.124"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.DomainSets]]
deps = ["CompositeTypes", "FunctionMaps", "IntervalSets", "LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "4599e0cd684f3ff6cbbab73c77553a3d01a8d74d"
uuid = "5b8099bc-c8ec-5219-889f-1d9e522a28bf"
version = "0.7.18"
weakdeps = ["Makie", "Random"]

    [deps.DomainSets.extensions]
    DomainSetsMakieExt = "Makie"
    DomainSetsRandomExt = "Random"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.DynamicHMC]]
deps = ["ArgCheck", "DocStringExtensions", "FillArrays", "LinearAlgebra", "LogDensityProblems", "LogExpFunctions", "ProgressMeter", "Random", "Statistics", "TensorCast"]
git-tree-sha1 = "3d22b5806afd1c5d675554f8e0d7a21b03e68237"
uuid = "bbc10e6e-7c05-544b-b16e-64fede858acb"
version = "3.6.0"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.EnumX]]
git-tree-sha1 = "c49898e8438c828577f04b92fc9368c388ac783c"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.7"

[[deps.ExactPredicates]]
deps = ["IntervalArithmetic", "Random", "StaticArrays"]
git-tree-sha1 = "83231673ea4d3d6008ac74dc5079e77ab2209d8f"
uuid = "429591f6-91af-11e9-00e2-59fbe8cec110"
version = "2.2.9"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "d36f682e590a83d63d1c7dbd287573764682d12a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.11"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "27af30de8b5445644e8ffe3bcb0d72049c089cf1"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.7.3+0"

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.Extents]]
git-tree-sha1 = "b309b36a9e02fe7be71270dd8c0fd873625332b4"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.6"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libva_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "66381d7059b5f3f6162f28831854008040a4e905"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "8.0.1+1"

[[deps.FFTA]]
deps = ["AbstractFFTs", "DocStringExtensions", "LinearAlgebra", "MuladdMacro", "Primes", "Random", "Reexport"]
git-tree-sha1 = "65e55303b72f4a567a51b174dd2c47496efeb95a"
uuid = "b86e33f2-c0db-4aa1-a6e0-ab43e668529e"
version = "0.3.1"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "Libdl", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "97f08406df914023af55ade2f843c39e99c5d969"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.10.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6d6219a004b8cf1e0b4dbe27a2860b8e04eba0be"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.11+0"

[[deps.FHist]]
deps = ["BayesHistogram", "LinearAlgebra", "Measurements", "Statistics", "StatsBase"]
git-tree-sha1 = "933544fcac69784e5e46f0fcb2d3f994cc4735fa"
uuid = "68837c9b-b678-4cd5-9925-8a54edc8f695"
version = "0.11.17"

    [deps.FHist.extensions]
    FHistHDF5Ext = "HDF5"
    FHistMakieExt = "Makie"
    FHistPlotsExt = ["RecipesBase", "Plots"]

    [deps.FHist.weakdeps]
    CairoMakie = "13f3f980-e62b-5c42-98c6-ff1f3baf88f0"
    HDF5 = "f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"

[[deps.FastGaussQuadrature]]
deps = ["LinearAlgebra", "SpecialFunctions", "StaticArrays"]
git-tree-sha1 = "0044e9f5e49a57e88205e8f30ab73928b05fe5b6"
uuid = "442a2c76-b920-505d-bb47-c5924d526838"
version = "1.1.0"

[[deps.FastTransforms]]
deps = ["AbstractFFTs", "ArrayLayouts", "BandedMatrices", "FFTW", "FastGaussQuadrature", "FastTransforms_jll", "FillArrays", "GenericFFT", "LazyArrays", "Libdl", "LinearAlgebra", "RecurrenceRelationships", "SpecialFunctions", "ToeplitzMatrices"]
git-tree-sha1 = "0cf70a407262f6f088eabcb464b0e887b485099e"
uuid = "057dd010-8810-581a-b7be-e3fc3b93f78c"
version = "0.17.1"

[[deps.FastTransforms_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "FFTW_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl", "MPFR_jll", "OpenBLAS_jll"]
git-tree-sha1 = "b28d81a08c4b79324d2cf942c398b4fe87280932"
uuid = "34b6f7d7-08f9-5794-9e10-3819e4c7e49a"
version = "0.6.4+0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "6522cfb3b8fe97bec632252263057996cbd3de20"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.18.0"
weakdeps = ["HTTP"]

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

[[deps.FilePaths]]
deps = ["FilePathsBase", "MacroTools", "Reexport"]
git-tree-sha1 = "a1b2fbfe98503f15b665ed45b3d149e5d8895e4c"
uuid = "8fc22ac5-c921-52a6-82fd-178b2807b824"
version = "0.9.0"

    [deps.FilePaths.extensions]
    FilePathsGlobExt = "Glob"
    FilePathsURIParserExt = "URIParser"
    FilePathsURIsExt = "URIs"

    [deps.FilePaths.weakdeps]
    Glob = "c27321d9-0574-5035-807b-f59d2c89b15c"
    URIParser = "30578b45-9adc-5946-b283-645ec420af67"
    URIs = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates"]
git-tree-sha1 = "3bab2c5aa25e7840a4b065805c0cdfc01f3068d2"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.24"
weakdeps = ["Mmap", "Test"]

    [deps.FilePathsBase.extensions]
    FilePathsBaseMmapExt = "Mmap"
    FilePathsBaseTestExt = "Test"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "2f979084d1e13948a3352cf64a25df6bd3b4dca3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.16.0"
weakdeps = ["PDMats", "SparseArrays", "StaticArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStaticArraysExt = "StaticArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "f85dac9a96a01087df6e3a749840015a0ca3817d"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.17.1+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "cddeab6487248a39dae1a960fff0ac17b2a28888"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "1.3.3"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FredholmDeterminants]]
deps = ["DifferentiationInterface", "Distributions", "FastGaussQuadrature", "ForwardDiff", "LinearAlgebra", "LogExpFunctions", "SpecialFunctions"]
git-tree-sha1 = "7875bbbeb3c628aafebd30f3561be2a459993897"
repo-rev = "main"
repo-url = "https://github.com/simeonschaub/FredholmDeterminants.jl"
uuid = "807c80a6-c809-4266-8359-c14a54b3d3b7"
version = "1.0.0-DEV"

[[deps.FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "907369da0f8e80728ab49c1c7e09327bf0d6d999"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.1.1"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "70329abc09b886fd2c5d94ad2d9527639c421e3e"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.14.3+1"

[[deps.FreeTypeAbstraction]]
deps = ["BaseDirs", "ColorVectorSpace", "Colors", "FreeType", "GeometryBasics", "Mmap"]
git-tree-sha1 = "4ebb930ef4a43817991ba35db6317a05e59abd11"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.10.8"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.FunctionMaps]]
deps = ["CompositeTypes", "LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "31bd99a57edf98990d1c21486032963955450e8d"
uuid = "a85aefff-f8ca-4649-a888-c8e5398bc76c"
version = "0.1.2"

[[deps.FunctionWrappers]]
git-tree-sha1 = "d62485945ce5ae9c0c48f124a84998d755bae00e"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.3"

[[deps.FunctionWrappersWrappers]]
deps = ["FunctionWrappers", "PrecompileTools", "TruncatedStacktraces"]
git-tree-sha1 = "3e13d0b39d117a03d3fb5c88a039e94787a37fcb"
uuid = "77dc65aa-8811-40c2-897b-53d922fa7daf"
version = "1.4.0"

    [deps.FunctionWrappersWrappers.extensions]
    FunctionWrappersWrappersEnzymeExt = ["Enzyme", "EnzymeCore"]
    FunctionWrappersWrappersMooncakeExt = "Mooncake"

    [deps.FunctionWrappersWrappers.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GLM]]
deps = ["Distributions", "LinearAlgebra", "Printf", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "StatsModels"]
git-tree-sha1 = "3bcb30438ee1655e3b9c42d97544de7addc9c589"
uuid = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
version = "1.9.3"

[[deps.GMP_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "781609d7-10c4-51f6-84f2-b8444358ff6d"
version = "6.3.0+2"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "83cf05ab16a73219e5f6bd1bdfa9848fa24ac627"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.2.0"

[[deps.GSL]]
deps = ["GSL_jll", "Libdl", "Markdown"]
git-tree-sha1 = "3ebd07d519f5ec318d5bc1b4971e2472e14bd1f0"
uuid = "92c85e6c-cbff-5e0c-80f7-495c94daaecd"
version = "1.0.1"

[[deps.GSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7da3a878517e1420850569d35d9ba583b745e39d"
uuid = "1b77fbbe-d8ee-58f0-85f9-836ddc23a7a4"
version = "2.8.1+0"

[[deps.GenericFFT]]
deps = ["AbstractFFTs", "FFTW", "LinearAlgebra", "Reexport"]
git-tree-sha1 = "1bc01f2ea9a0226a60723794ff86b8017739f5d9"
uuid = "a8297547-1b15-4a5a-a998-a2ac5f1cef28"
version = "0.1.6"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "Extents", "IterTools", "LinearAlgebra", "PrecompileTools", "Random", "StaticArrays"]
git-tree-sha1 = "1f5a80f4ed9f5a4aada88fc2db456e637676414b"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.5.10"

    [deps.GeometryBasics.extensions]
    GeometryBasicsGeoInterfaceExt = "GeoInterface"

    [deps.GeometryBasics.weakdeps]
    GeoInterface = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Giflib_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6570366d757b50fabae9f4315ad74d2e40c0560a"
uuid = "59f7168a-df46-5410-90c8-f2779963d0ec"
version = "5.2.3+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "24f6def62397474a297bfcec22384101609142ed"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.86.3+0"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "a641238db938fff9b2f60d08ed9030387daf428c"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.3"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a6dbda1fd736d60cc477d99f2e7a042acfa46e8"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.15+0"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "DataStructures", "Inflate", "LinearAlgebra", "Random", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "7eb45fe833a5b7c51cf6d89c5a841d5967e44be3"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.14.0"
weakdeps = ["Distributed", "SharedArrays"]

    [deps.Graphs.extensions]
    GraphsSharedArraysExt = "SharedArrays"

[[deps.GridLayoutBase]]
deps = ["GeometryBasics", "InteractiveUtils", "Observables"]
git-tree-sha1 = "93d5c27c8de51687a2c70ec0716e6e76f298416f"
uuid = "3955a311-db13-416c-9275-1d80ed98e5e9"
version = "0.11.2"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HCubature]]
deps = ["Combinatorics", "DataStructures", "LinearAlgebra", "QuadGK", "StaticArrays"]
git-tree-sha1 = "8ee627fb73ecba0b5254158b04d4745611b404a1"
uuid = "19dc6840-f33b-545b-b366-655c7e3ffd49"
version = "1.8.0"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "51059d23c8bb67911a2e6fd5130229113735fc7e"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.11.0"

[[deps.HalfIntegers]]
git-tree-sha1 = "9c3149243abb5bc0bad0431d6c4fcac0f4443c7c"
uuid = "f0d1745a-41c9-11e9-1dd9-e5d34d218721"
version = "1.6.0"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "f923f9a774fcf3f5cb761bfa43aeadd689714813"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.1+0"

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "68c173f4f449de5b438ee67ed0c9c748dc31a2ec"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.28"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "e12629406c6c4442539436581041d372d69c55ba"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.12"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "eb49b82c172811fd2c86759fa0553a2221feb909"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.7"

[[deps.ImageCore]]
deps = ["ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "8c193230235bbcee22c8066b0374f63b5683c2d3"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.5"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs", "WebP"]
git-tree-sha1 = "696144904b76e1ca433b886b4e7edd067d76cbf7"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.9"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "2a81c3897be6fbcde0802a0ebe6796d0562f63ec"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.10"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "dcc8d0cd653e55213df9b75ebc6fe4a8d3254c65"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.2.2+0"

[[deps.Indexing]]
git-tree-sha1 = "ce1566720fd6b19ff3411404d4b977acd4814f9f"
uuid = "313cdc1a-70c2-5d6a-ae34-0150d3930a38"
version = "1.1.1"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.InfiniteArrays]]
deps = ["ArrayLayouts", "FillArrays", "Infinities", "LazyArrays", "LinearAlgebra"]
git-tree-sha1 = "6370db1acf00f5a41f572f1edec036e3cf03e470"
uuid = "4858937d-0d70-526a-a4dd-2d5cb5dd786c"
version = "0.15.11"
weakdeps = ["BandedMatrices", "BlockArrays", "BlockBandedMatrices", "DSP", "Statistics"]

    [deps.InfiniteArrays.extensions]
    InfiniteArraysBandedMatricesExt = "BandedMatrices"
    InfiniteArraysBlockArraysExt = "BlockArrays"
    InfiniteArraysBlockBandedMatricesExt = "BlockBandedMatrices"
    InfiniteArraysDSPExt = "DSP"
    InfiniteArraysStatisticsExt = "Statistics"

[[deps.Infinities]]
git-tree-sha1 = "4495006c20b2fd27b8c453a1dd31d423654f3772"
uuid = "e1ba4f0e-776d-440f-acd9-e1d2e9742647"
version = "0.1.12"

[[deps.Inflate]]
git-tree-sha1 = "d1b1b796e47d94588b3757fe84fbf65a5ec4a80d"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.5"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InlineStrings]]
git-tree-sha1 = "8f3d257792a522b4601c24a577954b0a8cd7334d"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.5"
weakdeps = ["ArrowTypes", "Parsers"]

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

[[deps.IntegerMathUtils]]
git-tree-sha1 = "4c1acff2dc6b6967e7e750633c50bc3b8d83e617"
uuid = "18e54dd8-cb9d-406c-a71d-865a43cbb235"
version = "0.1.3"

[[deps.Integrals]]
deps = ["ArrayInterface", "CommonSolve", "HCubature", "LinearAlgebra", "MonteCarloIntegration", "QuadGK", "Random", "Reexport", "SciMLBase", "SciMLLogging"]
git-tree-sha1 = "a93089c3454ab25f08f2c8c7eb81e7bba4343c12"
uuid = "de52edbc-65ea-441a-8357-d3a637375a31"
version = "5.4.0"

    [deps.Integrals.extensions]
    IntegralsArblibExt = "Arblib"
    IntegralsCubaExt = "Cuba"
    IntegralsCubatureExt = "Cubature"
    IntegralsDifferentiationInterfaceExt = ["ADTypes", "DifferentiationInterface", "ChainRulesCore"]
    IntegralsFastGaussQuadratureExt = "FastGaussQuadrature"
    IntegralsFastTanhSinhQuadratureExt = "FastTanhSinhQuadrature"
    IntegralsForwardDiffExt = "ForwardDiff"
    IntegralsHAdaptiveIntegrationExt = "HAdaptiveIntegration"
    IntegralsMCIntegrationExt = "MCIntegration"
    IntegralsMooncakeExt = ["Mooncake", "Zygote", "ChainRulesCore"]
    IntegralsZygoteExt = ["Zygote", "ChainRulesCore"]

    [deps.Integrals.weakdeps]
    ADTypes = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
    Arblib = "fb37089c-8514-4489-9461-98f9c8763369"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Cuba = "8a292aeb-7a57-582c-b821-06e4c11590b1"
    Cubature = "667455a9-e2ce-5579-9412-b964f529a492"
    DifferentiationInterface = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63"
    FastGaussQuadrature = "442a2c76-b920-505d-bb47-c5924d526838"
    FastTanhSinhQuadrature = "b650e0df-f744-4436-b963-b44034668c57"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    HAdaptiveIntegration = "eaa5ad34-b243-48e9-b09c-54bc0655cecf"
    MCIntegration = "ea1e2de9-7db7-4b42-91ee-0cd1bf6df167"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "ec1debd61c300961f98064cfb21287613ad7f303"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2025.2.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "65d505fa4c0d7072990d659ef3fc086eb6da8208"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.16.2"
weakdeps = ["ForwardDiff", "Unitful"]

    [deps.Interpolations.extensions]
    InterpolationsForwardDiffExt = "ForwardDiff"
    InterpolationsUnitfulExt = "Unitful"

[[deps.IntervalArithmetic]]
deps = ["CRlibm", "CoreMath", "MacroTools", "OpenBLASConsistentFPCSR_jll", "Printf", "Random", "RoundingEmulator"]
git-tree-sha1 = "f1c42fcaca2d8034fe392f3e86c2e0809f75b2a1"
uuid = "d1acc4aa-44c8-5952-acd4-ba5d80a2a253"
version = "1.0.6"

    [deps.IntervalArithmetic.extensions]
    IntervalArithmeticArblibExt = "Arblib"
    IntervalArithmeticDiffRulesExt = "DiffRules"
    IntervalArithmeticForwardDiffExt = "ForwardDiff"
    IntervalArithmeticIntervalSetsExt = "IntervalSets"
    IntervalArithmeticIrrationalConstantsExt = "IrrationalConstants"
    IntervalArithmeticLinearAlgebraExt = "LinearAlgebra"
    IntervalArithmeticRecipesBaseExt = "RecipesBase"
    IntervalArithmeticSparseArraysExt = "SparseArrays"

    [deps.IntervalArithmetic.weakdeps]
    Arblib = "fb37089c-8514-4489-9461-98f9c8763369"
    DiffRules = "b552c78f-8df3-52c6-915a-8e097449b14b"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    IrrationalConstants = "92d709cd-6900-40b7-9082-c6be49f344b6"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.IntervalSets]]
git-tree-sha1 = "79d6bd28c8d9bccc2229784f1bd637689b256377"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.14"
weakdeps = ["Random", "RecipesBase", "Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.InvertedIndices]]
git-tree-sha1 = "6da3c4316095de0f5ee2ebd875df8721e7e0bdbe"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.1"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.Isoband]]
deps = ["isoband_jll"]
git-tree-sha1 = "f9b6d97355599074dc867318950adaa6f9946137"
uuid = "f1662d9f-8043-43de-a69a-05efc1cc6ff4"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "0533e564aae234aff59ab625543145446d8b6ec2"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.1"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "67c6f1f085cb2671c93fe34244c9cccde30f7a26"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.5.0"
weakdeps = ["ArrowTypes"]

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

[[deps.JSON3]]
deps = ["Dates", "Mmap", "Parsers", "PrecompileTools", "StructTypes", "UUIDs"]
git-tree-sha1 = "411eccfe8aba0814ffa0fdf4860913ed09c34975"
uuid = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
version = "1.14.3"
weakdeps = ["ArrowTypes"]

    [deps.JSON3.extensions]
    JSON3ArrowExt = ["ArrowTypes"]

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "9496de8fb52c224a2e3f9ff403947674517317d9"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.6"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c0c9b76f3520863909825cbecdef58cd63de705a"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.5+0"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "58927c485919bf17ea308d9d82156de1adf4b006"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.10.12"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTA", "Interpolations", "StatsBase"]
git-tree-sha1 = "4260cfc991b8885bf747801fb60dd4503250e478"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.11"

[[deps.KrylovKit]]
deps = ["LinearAlgebra", "PackageExtensionCompat", "Printf", "Random", "VectorInterface"]
path = "/home/simeon/.julia/dev/KrylovKit"
uuid = "0b1a1467-8014-51b9-945f-bf0ae24f4b77"
version = "0.10.2"
weakdeps = ["ChainRulesCore"]

    [deps.KrylovKit.extensions]
    KrylovKitChainRulesCoreExt = "ChainRulesCore"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "17b94ecafcfa45e8360a4fc9ca6b583b049e4e37"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.1.0+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eb62a3deb62fc6d8822c0c4bef73e4412419c5d8"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.8+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.LatticeRules]]
deps = ["Random"]
git-tree-sha1 = "7f5b02258a3ca0221a6a9710b0a0a2e8fb4957fe"
uuid = "73f95e8e-ec14-4e6a-8b18-0d2e271c4e55"
version = "0.0.1"

[[deps.LazyArrays]]
deps = ["ArrayLayouts", "FillArrays", "LinearAlgebra", "MacroTools", "SparseArrays"]
git-tree-sha1 = "33b5d8fafb7ab69eca907b359d00d0107feb2cbf"
uuid = "5078a376-72f3-5289-bfd5-ec5146d43c02"
version = "2.9.7"
weakdeps = ["BandedMatrices", "BlockArrays", "BlockBandedMatrices", "StaticArrays"]

    [deps.LazyArrays.extensions]
    LazyArraysBandedMatricesExt = "BandedMatrices"
    LazyArraysBlockArraysExt = "BlockArrays"
    LazyArraysBlockBandedMatricesExt = "BlockBandedMatrices"
    LazyArraysStaticArraysExt = "StaticArrays"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

[[deps.LazyStack]]
deps = ["ChainRulesCore", "Compat", "LinearAlgebra"]
git-tree-sha1 = "aff621f1f49e9262a34aaf0d57d02ea3b35aec60"
uuid = "1fad7336-0346-5a1a-a56f-a06ba010965b"
version = "0.1.3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.15.0+0"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cc3ad4faf30015a3e8094c9b5b7f19e85bdf2386"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.42.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "f04133fe05eff1667d2054c53d59f9122383fe05"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.2+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d620582b1f0cbe2c72dd1d5bd195a9ce73370ab1"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.42.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.LogDensityProblems]]
deps = ["ArgCheck", "DocStringExtensions", "Random"]
git-tree-sha1 = "d9625f27ded4ad726ceca7819394a4cc77ed25b3"
uuid = "6fdf6af0-433a-55f7-b3ed-c6c6e0b8df7c"
version = "2.2.0"

[[deps.LogDensityProblemsAD]]
deps = ["DocStringExtensions", "LogDensityProblems"]
git-tree-sha1 = "7b83f3ad0a8105f79a067cafbfd124827bb398d0"
uuid = "996a588d-648d-4e1f-a8f0-a84b347e47b1"
version = "1.13.1"

    [deps.LogDensityProblemsAD.extensions]
    LogDensityProblemsADADTypesExt = "ADTypes"
    LogDensityProblemsADDifferentiationInterfaceExt = ["ADTypes", "DifferentiationInterface"]
    LogDensityProblemsADEnzymeExt = "Enzyme"
    LogDensityProblemsADFiniteDifferencesExt = "FiniteDifferences"
    LogDensityProblemsADForwardDiffBenchmarkToolsExt = ["BenchmarkTools", "ForwardDiff"]
    LogDensityProblemsADForwardDiffExt = "ForwardDiff"
    LogDensityProblemsADReverseDiffExt = "ReverseDiff"
    LogDensityProblemsADTrackerExt = "Tracker"
    LogDensityProblemsADZygoteExt = "Zygote"

    [deps.LogDensityProblemsAD.weakdeps]
    ADTypes = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
    BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
    DifferentiationInterface = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    FiniteDifferences = "26cc04aa-876d-5657-8c51-4c34ba976000"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f00544d95982ea270145636c181ceda21c4e2575"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.2.0"

[[deps.LowRankMatrices]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "59c5bb0708be6796604caec16d4357013dc3d132"
uuid = "e65ccdef-c354-471a-8090-89bec1c20ec3"
version = "1.0.2"
weakdeps = ["FillArrays"]

    [deps.LowRankMatrices.extensions]
    LowRankMatricesFillArraysExt = "FillArrays"

[[deps.LoweredCodeUtils]]
deps = ["CodeTracking", "Compiler", "JuliaInterpreter"]
git-tree-sha1 = "5d4278f755440f70648d80cc6225f51e78e94094"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "3.5.1"

[[deps.Lz4_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "191686b1ac1ea9c89fc52e996ad15d1d241d1e33"
uuid = "5ced341a-0733-55b8-9ab6-a4889d929147"
version = "1.10.1+0"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "282cadc186e7b2ae0eeadbd7a4dffed4196ae2aa"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2025.2.0+0"

[[deps.MPFR_jll]]
deps = ["Artifacts", "GMP_jll", "Libdl"]
uuid = "3a97d323-0669-5f0c-9066-3539efd106a3"
version = "4.2.2+0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Makie]]
deps = ["Animations", "Base64", "CRC32c", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "ComputePipeline", "Contour", "Dates", "DelaunayTriangulation", "Distributions", "DocStringExtensions", "Downloads", "FFMPEG_jll", "FileIO", "FilePaths", "FixedPointNumbers", "Format", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageBase", "ImageIO", "InteractiveUtils", "Interpolations", "IntervalSets", "InverseFunctions", "Isoband", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MacroTools", "Markdown", "MathTeXEngine", "Observables", "OffsetArrays", "PNGFiles", "Packing", "Pkg", "PlotUtils", "PolygonOps", "PrecompileTools", "Printf", "REPL", "Random", "RelocatableFolders", "Scratch", "ShaderAbstractions", "Showoff", "SignedDistanceFields", "SparseArrays", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "TriplotBase", "UnicodeFun", "Unitful"]
git-tree-sha1 = "68af66ec16af8b152309310251ecb4fbfe39869f"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.24.9"

    [deps.Makie.extensions]
    MakieDynamicQuantitiesExt = "DynamicQuantities"

    [deps.Makie.weakdeps]
    DynamicQuantities = "06fc5a27-2a28-4c7c-a15d-362465fb6821"

[[deps.MappedArrays]]
git-tree-sha1 = "0ee4497a4e80dbd29c058fcee6493f5219556f40"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.3"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MathTeXEngine]]
deps = ["AbstractTrees", "Automa", "DataStructures", "FreeTypeAbstraction", "GeometryBasics", "LaTeXStrings", "REPL", "RelocatableFolders", "UnicodeFun"]
git-tree-sha1 = "7eb8cdaa6f0e8081616367c10b31b9d9b34bb02a"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.6.7"

[[deps.MatrixFactorizations]]
deps = ["ArrayLayouts", "LinearAlgebra", "Printf", "Random"]
git-tree-sha1 = "3bb3cf4685f1c90f22883f4c4bb6d203fa882b79"
uuid = "a3b82374-2e81-5b9e-98ce-41277c0e4c87"
version = "3.1.3"
weakdeps = ["BandedMatrices"]

    [deps.MatrixFactorizations.extensions]
    MatrixFactorizationsBandedMatricesExt = "BandedMatrices"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "8785729fa736197687541f7053f6d8ab7fc44f92"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.10"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ff69a2b1330bcb730b9ac1ab7dd680176f5896b8"
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.1010+0"

[[deps.Measurements]]
deps = ["Calculus", "LinearAlgebra", "Printf"]
git-tree-sha1 = "cb47f69a1cab9dcec7ff4a5d6e163410d6905866"
uuid = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
version = "2.14.1"

    [deps.Measurements.extensions]
    MeasurementsBaseTypeExt = "BaseType"
    MeasurementsJunoExt = "Juno"
    MeasurementsMakieExt = "Makie"
    MeasurementsRecipesBaseExt = "RecipesBase"
    MeasurementsSpecialFunctionsExt = "SpecialFunctions"
    MeasurementsUnitfulExt = "Unitful"

    [deps.Measurements.weakdeps]
    BaseType = "7fbed51b-1ef5-4d67-9085-a4a9b26f478c"
    Juno = "e5e0dc1b-0480-54bc-9374-aad01c23163d"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.MistyClosures]]
git-tree-sha1 = "d1a692e293c2a0dc8fda79c04cad60582f3d4de3"
uuid = "dbe65cb8-6be2-42dd-bbc5-4196aaced4f4"
version = "2.1.0"

[[deps.MixedModels]]
deps = ["Arrow", "BSplineKit", "Compat", "DataAPI", "Distributions", "GLM", "JSON3", "LinearAlgebra", "Markdown", "MixedModelsDatasets", "NLopt", "PooledArrays", "PrecompileTools", "Printf", "ProgressMeter", "Random", "RegressionFormulae", "SparseArrays", "StaticArrays", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "StatsModels", "StructTypes", "Tables", "TypedTables"]
git-tree-sha1 = "373313e0af5cfe04fb1d06008b974f3186c14a5b"
uuid = "ff71e718-51f3-5ec2-a782-8ffcbfa3c316"
version = "5.2.2"

    [deps.MixedModels.extensions]
    MixedModelsFiniteDiffExt = ["FiniteDiff"]
    MixedModelsForwardDiffExt = ["ForwardDiff"]
    MixedModelsPRIMAExt = ["PRIMA"]

    [deps.MixedModels.weakdeps]
    FiniteDiff = "6a86dc24-6348-571c-b903-95158fe2bd41"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    PRIMA = "0a7d04aa-8ac2-47b3-b7a7-9dbd6ad661ed"

[[deps.MixedModelsDatasets]]
deps = ["Arrow", "Artifacts", "LazyArtifacts"]
git-tree-sha1 = "ac0036e4f1829db000db46aad4cd5a207bba8465"
uuid = "7e9fb7ac-9f67-43bf-b2c8-96ba0796cbb6"
version = "0.1.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.Mocking]]
deps = ["Compat", "ExprTools"]
git-tree-sha1 = "2c140d60d7cb82badf06d8783800d0bcd1a7daa2"
uuid = "78c3b35d-d492-501b-9361-3d52fe80e533"
version = "0.8.1"

[[deps.MonteCarloIntegration]]
deps = ["Distributions", "QuasiMonteCarlo", "Random"]
git-tree-sha1 = "722ad522068d31954b4a976b66a26aeccbf509ed"
uuid = "4886b29c-78c9-11e9-0a6e-41e1f4161f7b"
version = "0.2.0"

[[deps.Mooncake]]
deps = ["ADTypes", "ChainRules", "ChainRulesCore", "DispatchDoctor", "ExprTools", "Graphs", "LinearAlgebra", "MistyClosures", "PrecompileTools", "Random", "Test"]
git-tree-sha1 = "bf6df81be03543cd072367880d729a331c5048a5"
uuid = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
version = "0.5.17"

    [deps.Mooncake.extensions]
    MooncakeAllocCheckExt = "AllocCheck"
    MooncakeCUDAExt = "CUDA"
    MooncakeDistributionsExt = "Distributions"
    MooncakeDynamicExpressionsExt = "DynamicExpressions"
    MooncakeFluxExt = "Flux"
    MooncakeFunctionWrappersExt = "FunctionWrappers"
    MooncakeJETExt = "JET"
    MooncakeLogExpFunctionsExt = "LogExpFunctions"
    MooncakeLuxLibExt = ["LuxLib", "MLDataDevices", "Static"]
    MooncakeLuxLibSLEEFPiratesExtension = ["LuxLib", "SLEEFPirates"]
    MooncakeNNlibExt = ["NNlib", "GPUArraysCore"]
    MooncakeSpecialFunctionsExt = "SpecialFunctions"

    [deps.Mooncake.weakdeps]
    AllocCheck = "9b6a8646-10ed-4001-bbdc-1d2f46dfbb1a"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
    DynamicExpressions = "a40a106e-89c9-4ca8-8020-a735e8728b6b"
    Flux = "587475ba-b771-5e3f-ad9e-33799f191a9c"
    FunctionWrappers = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    JET = "c3a54625-cd67-489e-a8e7-0a5a0ff4e31b"
    LogExpFunctions = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
    LuxLib = "82251201-b29d-42c6-8e01-566dec8acb11"
    MLDataDevices = "7e8f7934-dd98-4c1a-8fe8-92b47a384d40"
    NNlib = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
    SLEEFPirates = "476501e8-09a2-5ece-8869-fb82de89a1fa"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
    Static = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.11.4"

[[deps.MsgPack]]
deps = ["Serialization"]
git-tree-sha1 = "f5db02ae992c260e4826fe78c942954b48e1d9c2"
uuid = "99f44e22-a591-53d1-9472-aa23ef4bd671"
version = "1.2.1"

[[deps.MuladdMacro]]
git-tree-sha1 = "cac9cc5499c25554cba55cd3c30543cff5ca4fab"
uuid = "46d2c3a1-f734-5fdb-9937-b9b9aeba4221"
version = "0.2.4"

[[deps.NLopt]]
deps = ["CEnum", "NLopt_jll"]
git-tree-sha1 = "624785b15005a0e0f4e462b27ee745dbe5941863"
uuid = "76087f3c-5699-56af-9a33-bf431cd00edd"
version = "1.2.1"

    [deps.NLopt.extensions]
    NLoptMathOptInterfaceExt = ["MathOptInterface"]

    [deps.NLopt.weakdeps]
    MathOptInterface = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"

[[deps.NLopt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b0154a615d5b2b6cf7a2501123b793577d0b9950"
uuid = "079eb43e-fd8e-5478-9966-2cf3e3edb778"
version = "2.10.0+0"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "d92b107dbb887293622df7697a2223f9f8176fcd"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.1"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OddEvenIntegers]]
git-tree-sha1 = "256204fa8108cb52661d9d599821a23574097f50"
uuid = "8d37c425-f37a-4ca2-9b9d-a61bc06559d2"
version = "0.2.0"
weakdeps = ["HalfIntegers"]

    [deps.OddEvenIntegers.extensions]
    OddEvenIntegersHalfIntegersExt = "HalfIntegers"

[[deps.OffsetArrays]]
git-tree-sha1 = "117432e406b5c023f665fa73dc26e79ec3630151"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.17.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OhMyThreads]]
deps = ["BangBang", "ChunkSplitters", "ScopedValues", "StableTasks", "TaskLocalValues"]
git-tree-sha1 = "b3c63491156b66f60c2cd181ba05bc0b327e5c7d"
uuid = "67456a42-1dca-4109-a031-0a68de7e3ad5"
version = "0.8.5"
weakdeps = ["Markdown"]

    [deps.OhMyThreads.extensions]
    MarkdownExt = "Markdown"

[[deps.OpenBLASConsistentFPCSR_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f2b3b9e52a5eb6a3434c8cca67ad2dde011194f4"
uuid = "6cdc7f73-28fd-5e50-80fb-958a8875b1af"
version = "0.3.30+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "97db9e07fe2091882c765380ef58ec553074e9c7"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.3"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "9ac7c730c53b3b5d9a73fb900ac4b4fc263774db"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.4.9+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "NetworkOptions", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "1d1aaa7d449b58415f97d2839c318b70ffb525a0"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.6.1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.4+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e2bb57a313a74b8104064b7efd01406c0a50d2ff"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.6.1+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.44.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "e4cff168707d441cd6bf3ff7e4832bdf34278e4a"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.37"
weakdeps = ["StatsBase"]

    [deps.PDMats.extensions]
    StatsBaseExt = "StatsBase"

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "cf181f0b1e6a18dfeb0ee8acc4a9d1672499626c"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.4.4"

[[deps.PackageExtensionCompat]]
git-tree-sha1 = "fb28e33b8a95c4cee25ce296c817d89cc2e53518"
uuid = "65ce6f38-6b18-4e1d-a461-8949797d7930"
version = "1.0.2"
weakdeps = ["Requires", "TOML"]

[[deps.Packing]]
deps = ["GeometryBasics"]
git-tree-sha1 = "bc5bf2ea3d5351edf285a06b0016788a121ce92c"
uuid = "19eb6ba3-879d-56ad-ad62-d5c202156566"
version = "0.5.1"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58e5ed5e386e156bd93e86b305ebd21ac63d2d04"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.57.1+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "db76b1ecd5e9715f3d043cec13b2ec93ce015d53"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.44.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.1"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f9501cc0430a26bc3d156ae1b5b0c1b47af4d6da"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.3"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "26ca162858917496748aad52bb5d3be4d26a228a"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.4"

[[deps.PolygonOps]]
git-tree-sha1 = "77b3d3605fc1cd0b42d95eba87dfcd2bf67d5ff6"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.2"

[[deps.Polynomials]]
deps = ["LinearAlgebra", "OrderedCollections", "Setfield", "SparseArrays"]
git-tree-sha1 = "2d99b4c8a7845ab1342921733fa29366dae28b24"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "4.1.1"

    [deps.Polynomials.extensions]
    PolynomialsChainRulesCoreExt = "ChainRulesCore"
    PolynomialsFFTWExt = "FFTW"
    PolynomialsMakieExt = "Makie"
    PolynomialsMutableArithmeticsExt = "MutableArithmetics"
    PolynomialsRecipesBaseExt = "RecipesBase"

    [deps.Polynomials.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    MutableArithmetics = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PreallocationTools]]
deps = ["Adapt", "ArrayInterface", "PrecompileTools"]
git-tree-sha1 = "e16b73bf892c55d16d53c9c0dbd0fb31cb7e25da"
uuid = "d236fae5-4411-538c-8e31-a6e3d9e00b46"
version = "1.2.0"

    [deps.PreallocationTools.extensions]
    PreallocationToolsForwardDiffExt = "ForwardDiff"
    PreallocationToolsReverseDiffExt = "ReverseDiff"
    PreallocationToolsSparseConnectivityTracerExt = "SparseConnectivityTracer"

    [deps.PreallocationTools.weakdeps]
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseConnectivityTracer = "9f842d2f-2579-4b1d-911e-f412cf18a3f5"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "07a921781cab75691315adc645096ed5e370cb77"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.3"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "8b770b60760d4451834fe79dd483e318eee709c4"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.2"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "REPL", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "211530a7dc76ab59087f4d4d1fc3f086fbe87594"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "3.2.3"

    [deps.PrettyTables.extensions]
    PrettyTablesTypstryExt = "Typstry"

    [deps.PrettyTables.weakdeps]
    Typstry = "f0ed7684-a786-439e-b1e3-3b82803b501e"

[[deps.Primes]]
deps = ["IntegerMathUtils"]
git-tree-sha1 = "25cdd1d20cd005b52fc12cb6be3f75faaf59bb9b"
uuid = "27ebfcd6-29c5-5fa9-bf4b-fb8fc14df3ae"
version = "0.5.7"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Profile]]
deps = ["StyledStrings"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
version = "1.11.0"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "fbb92c6c56b34e1a2c4c36058f68f332bec840e7"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "4fbbafbc6251b883f4d2705356f3641f3652a7fe"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.4.0"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "472daaa816895cb7aee81658d4e7aec901fa1106"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.2"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "5e8e8b0ab68215d7a2b14b9921a946fee794749e"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.3"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.QuasiMonteCarlo]]
deps = ["Accessors", "ConcreteStructs", "LatticeRules", "LinearAlgebra", "PrecompileTools", "Primes", "Random", "Sobol"]
git-tree-sha1 = "017a6731a1754173013b98f458adb631e368a9d2"
uuid = "8a4e6c94-4038-4cdc-81c3-7e6ffdb2a71b"
version = "0.3.5"
weakdeps = ["Distributions"]

    [deps.QuasiMonteCarlo.extensions]
    QuasiMonteCarloDistributionsExt = "Distributions"

[[deps.REPL]]
deps = ["InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RandomMatrices]]
deps = ["Combinatorics", "Distributions", "FastGaussQuadrature", "GSL", "LinearAlgebra", "Random", "SpecialFunctions"]
git-tree-sha1 = "1d0563235c869e1c091655c3cc24667ddfafc739"
uuid = "2576dda1-a324-5b11-aa66-c48ed7e3c618"
version = "0.5.6"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecurrenceRelationships]]
git-tree-sha1 = "aa0b5958764e974a6e8d52f5b2daf51b26ede1a2"
uuid = "807425ed-42ea-44d6-a357-6771516d7b2c"
version = "0.2.0"
weakdeps = ["FillArrays", "LazyArrays", "LinearAlgebra"]

    [deps.RecurrenceRelationships.extensions]
    RecurrenceRelationshipsFillArraysExt = "FillArrays"
    RecurrenceRelationshipsLazyArraysExt = "LazyArrays"
    RecurrenceRelationshipsLinearAlgebraExt = "LinearAlgebra"

[[deps.RecursiveArrayTools]]
deps = ["Adapt", "ArrayInterface", "DocStringExtensions", "GPUArraysCore", "LinearAlgebra", "PrecompileTools", "RecipesBase", "StaticArraysCore", "SymbolicIndexingInterface"]
git-tree-sha1 = "f37545c2bbda4055cd4e0b41530229586b64bbe6"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "4.1.0"

    [deps.RecursiveArrayTools.extensions]
    RecursiveArrayToolsCUDAExt = "CUDA"
    RecursiveArrayToolsFastBroadcastExt = "FastBroadcast"
    RecursiveArrayToolsFastBroadcastPolyesterExt = ["FastBroadcast", "Polyester"]
    RecursiveArrayToolsForwardDiffExt = "ForwardDiff"
    RecursiveArrayToolsKernelAbstractionsExt = "KernelAbstractions"
    RecursiveArrayToolsMeasurementsExt = "Measurements"
    RecursiveArrayToolsMonteCarloMeasurementsExt = "MonteCarloMeasurements"
    RecursiveArrayToolsMooncakeExt = "Mooncake"
    RecursiveArrayToolsReverseDiffExt = ["ReverseDiff", "Zygote"]
    RecursiveArrayToolsSparseArraysExt = ["SparseArrays"]
    RecursiveArrayToolsStatisticsExt = "Statistics"
    RecursiveArrayToolsStructArraysExt = "StructArrays"
    RecursiveArrayToolsTablesExt = ["Tables"]
    RecursiveArrayToolsTrackerExt = "Tracker"
    RecursiveArrayToolsZygoteExt = "Zygote"

    [deps.RecursiveArrayTools.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    FastBroadcast = "7034ab61-46d4-4ed7-9d0f-46aef9175898"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    Polyester = "f517fe37-dbe3-4b94-8317-1923a5111588"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RegressionFormulae]]
deps = ["Combinatorics", "StatsModels"]
git-tree-sha1 = "d4dbe1f3f5dc6b3a7732aa9c54927b1a69f684e4"
uuid = "545c379f-4ec2-4339-9aea-38f2fb6a8ba2"
version = "0.1.4"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Revise]]
deps = ["CodeTracking", "FileWatching", "InteractiveUtils", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "Preferences", "REPL", "UUIDs"]
git-tree-sha1 = "5f4f629c085b87e71125eec6773f5f872c74a47a"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.14.2"
weakdeps = ["Distributed"]

    [deps.Revise.extensions]
    DistributedExt = "Distributed"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "5b3d50eb374cea306873b371d3f8d3915a018f0b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.9.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.RoundingEmulator]]
git-tree-sha1 = "40b9edad2e5287e05bd413a38f61a8ff55b9557b"
uuid = "5eaf0fd0-dfba-4ccb-bf02-d820a40db705"
version = "0.2.1"

[[deps.RuntimeGeneratedFunctions]]
deps = ["ExprTools", "SHA", "Serialization"]
git-tree-sha1 = "cfcdc949c4660544ab0fdeed169561cb22f835f4"
uuid = "7e49a35a-f44a-4d26-94aa-eba1b4ca6b47"
version = "0.5.18"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMD]]
deps = ["PrecompileTools"]
git-tree-sha1 = "e24dc23107d426a096d3eae6c165b921e74c18e4"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.7.2"

[[deps.SciMLBase]]
deps = ["ADTypes", "Accessors", "Adapt", "ArrayInterface", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "EnumX", "FunctionWrappersWrappers", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "Markdown", "PreallocationTools", "PrecompileTools", "Preferences", "Printf", "Random", "RecipesBase", "RecursiveArrayTools", "Reexport", "RuntimeGeneratedFunctions", "SciMLLogging", "SciMLOperators", "SciMLPublic", "SciMLStructures", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface"]
git-tree-sha1 = "4fdad3606c60fbbd52424737c31ec4141672c809"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "3.3.0"

    [deps.SciMLBase.extensions]
    SciMLBaseChainRulesCoreExt = "ChainRulesCore"
    SciMLBaseDifferentiationInterfaceExt = "DifferentiationInterface"
    SciMLBaseDistributionsExt = "Distributions"
    SciMLBaseEnzymeExt = "Enzyme"
    SciMLBaseForwardDiffExt = "ForwardDiff"
    SciMLBaseMakieExt = "Makie"
    SciMLBaseMeasurementsExt = "Measurements"
    SciMLBaseMonteCarloMeasurementsExt = "MonteCarloMeasurements"
    SciMLBaseMooncakeExt = "Mooncake"
    SciMLBasePartialFunctionsExt = "PartialFunctions"
    SciMLBasePyCallExt = "PyCall"
    SciMLBasePythonCallExt = "PythonCall"
    SciMLBaseRCallExt = "RCall"
    SciMLBaseReverseDiffExt = "ReverseDiff"
    SciMLBaseTrackerExt = "Tracker"
    SciMLBaseZygoteExt = ["Zygote", "ChainRulesCore"]

    [deps.SciMLBase.weakdeps]
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DifferentiationInterface = "a0c0ee7d-e4b9-4e03-894e-1c5f64a51d63"
    Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    Mooncake = "da2b9cff-9c12-43a0-ae48-6db2b0edb7d6"
    PartialFunctions = "570af359-4316-4cb7-8c74-252c00c2016b"
    PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
    PythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
    RCall = "6f49c342-dc21-5d91-9882-a32aef131414"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.SciMLLogging]]
deps = ["Logging", "LoggingExtras", "Preferences"]
git-tree-sha1 = "0161be062570af4042cf6f69e3d5d0b0555b6927"
uuid = "a6db7da4-7206-11f0-1eab-35f2a5dbe1d1"
version = "1.9.1"

    [deps.SciMLLogging.extensions]
    SciMLLoggingTracyExt = "Tracy"

    [deps.SciMLLogging.weakdeps]
    Tracy = "e689c965-62c8-4b79-b2c5-8359227902fd"

[[deps.SciMLOperators]]
deps = ["Accessors", "ArrayInterface", "DocStringExtensions", "LinearAlgebra"]
git-tree-sha1 = "234869cf9fee9258a95464b7a7065cc7be84db00"
uuid = "c0aeaf25-5076-4817-a8d5-81caf7dfa961"
version = "1.16.0"
weakdeps = ["SparseArrays", "StaticArraysCore"]

    [deps.SciMLOperators.extensions]
    SciMLOperatorsSparseArraysExt = "SparseArrays"
    SciMLOperatorsStaticArraysCoreExt = "StaticArraysCore"

[[deps.SciMLPublic]]
git-tree-sha1 = "0ba076dbdce87ba230fff48ca9bca62e1f345c9b"
uuid = "431bcebd-1456-4ced-9d72-93c2757fff0b"
version = "1.0.1"

[[deps.SciMLStructures]]
deps = ["ArrayInterface", "PrecompileTools"]
git-tree-sha1 = "607f6867d0b0553e98fc7f725c9f9f13b4d01a32"
uuid = "53ae85a6-f571-4167-b2af-e1d143709226"
version = "1.10.0"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "ac4b837d89a58c848e85e698e2a2514e9d59d8f6"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.6.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "ebe7e59b37c400f694f52b58c93d26201387da70"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.9"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "c5391c6ace3bc430ca630251d02ea9687169ca68"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.2"

[[deps.ShaderAbstractions]]
deps = ["ColorTypes", "FixedPointNumbers", "GeometryBasics", "LinearAlgebra", "Observables", "StaticArrays"]
git-tree-sha1 = "818554664a2e01fc3784becb2eb3a82326a604b6"
uuid = "65257c39-d410-5151-9873-9b3e5be5013e"
version = "0.5.0"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
version = "1.11.0"

[[deps.ShiftedArrays]]
git-tree-sha1 = "503688b59397b3307443af35cd953a13e8005c16"
uuid = "1277b4bf-5013-50f5-be3d-901d8477a67a"
version = "2.0.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SignedDistanceFields]]
deps = ["Statistics"]
git-tree-sha1 = "3949ad92e1c9d2ff0cd4a1317d5ecbba682f4b92"
uuid = "73760f76-fbc4-59ce-8f25-708e95d2df96"
version = "0.4.1"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "be8eeac05ec97d379347584fa9fe2f5f76795bcb"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.5"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "0494aed9501e7fb65daba895fb7fd57cc38bc743"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.5"

[[deps.SkewLinearAlgebra]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ba8f829daf7e16017f3122f9a4ba8c48b283a905"
uuid = "5c889d49-8c60-4500-9d10-5d3a22e2f4b9"
version = "1.1.0"

[[deps.Sobol]]
deps = ["DelimitedFiles", "Random"]
git-tree-sha1 = "5a74ac22a9daef23705f010f72c81d6925b19df8"
uuid = "ed01d8cd-4d21-5b2a-85b4-cc3bdc58bad4"
version = "1.5.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "64d974c2e6fdf07f8155b5b2ca2ffa9069b608d9"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.2"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.SparseInverseSubset]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "52962839426b75b3021296f7df242e40ecfc0852"
uuid = "dc90abb0-5640-4711-901d-7e5b23a2fada"
version = "0.1.2"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "2700b235561b0335d5bef7097a111dc513b8655e"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.7.2"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.SplitApplyCombine]]
deps = ["Dictionaries", "Indexing"]
git-tree-sha1 = "c06d695d51cfb2187e6848e98d6252df9101c588"
uuid = "03a91e81-4c3e-53e1-a0a4-9c0c8f19dd66"
version = "1.2.3"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "4f96c596b8c8258cc7d3b19797854d368f243ddc"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.4"

[[deps.StableTasks]]
git-tree-sha1 = "c4f6610f85cb965bee5bfafa64cbeeda55a4e0b2"
uuid = "91464d47-22a1-43fe-8b7f-2d57ee82463f"
version = "0.1.7"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "be1cf4eb0ac528d96f5115b4ed80c26a8d8ae621"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.2"

[[deps.Static]]
deps = ["CommonWorldInvalidations", "IfElse", "PrecompileTools", "SciMLPublic"]
git-tree-sha1 = "49440414711eddc7227724ae6e570c7d5559a086"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "1.3.1"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "246a8bb2e6667f832eea063c3a56aef96429a3db"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.18"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6ab403037779dae8c514bad259f32a447262455a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.4"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "178ed29fd5b2a2cfc3bd31c13375ae925623ff36"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.8.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "aceda6f4e598d331548e04cc6b2124a6148138e3"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.10"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "91f091a8716a6bb38417a6e6f274602a19aaa685"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.5.2"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StatsModels]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Printf", "REPL", "ShiftedArrays", "SparseArrays", "StatsAPI", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "08786db4a1346d17d0a8d952d2e66fd00fa18192"
uuid = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
version = "0.7.9"

[[deps.Strided]]
deps = ["LinearAlgebra", "StridedViews", "TupleTools"]
git-tree-sha1 = "0fa01489ecc57749f2f7dcbc2918e46a164af8b2"
uuid = "5e0ebb24-38b0-5f93-81fe-25c709ecae67"
version = "2.3.4"

    [deps.Strided.extensions]
    StridedAMDGPUExt = "AMDGPU"
    StridedCUDAExt = "CUDA"
    StridedGPUArraysExt = "GPUArrays"
    StridedJLArraysExt = "JLArrays"

    [deps.Strided.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    GPUArrays = "0c68f7d7-f131-5f86-a1c3-88cf8149b2d7"
    JLArrays = "27aeb0d3-9eb9-45fb-866b-73c2ecf80fcb"

[[deps.StridedViews]]
deps = ["LinearAlgebra", "PackageExtensionCompat"]
git-tree-sha1 = "b1b42ff0249fbb02df163633adc612b943c6ac74"
uuid = "4db3bf67-4bd7-4b4e-b153-31dc3fb37143"
version = "0.4.6"

    [deps.StridedViews.extensions]
    StridedViewsAMDGPUExt = "AMDGPU"
    StridedViewsCUDAExt = "CUDA"
    StridedViewsJLArraysExt = "JLArrays"
    StridedViewsPtrArraysExt = "PtrArrays"

    [deps.StridedViews.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    JLArrays = "27aeb0d3-9eb9-45fb-866b-73c2ecf80fcb"
    PtrArrays = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "d05693d339e37d6ab134c5ab53c29fce5ee5d7d5"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.4"

[[deps.StringViews]]
git-tree-sha1 = "f2dcb92855b31ad92fe8f079d4f75ac57c93e4b8"
uuid = "354b36f9-a18e-4713-926e-db85100087ba"
version = "1.3.7"

[[deps.StructArrays]]
deps = ["ConstructionBase", "DataAPI", "Tables"]
git-tree-sha1 = "ad8002667372439f2e3611cfd14097e03fa4bccd"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.7.3"

    [deps.StructArrays.extensions]
    StructArraysAdaptExt = "Adapt"
    StructArraysGPUArraysCoreExt = ["GPUArraysCore", "KernelAbstractions"]
    StructArraysLinearAlgebraExt = "LinearAlgebra"
    StructArraysSparseArraysExt = "SparseArrays"
    StructArraysStaticArraysExt = "StaticArrays"

    [deps.StructArrays.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.StructTypes]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "159331b30e94d7b11379037feeb9b690950cace8"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.11.0"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "aab80fbf866600f3299dd7f6656d80e7be177cfe"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.7.2"
weakdeps = ["Measurements", "StaticArraysCore", "Tables"]

    [deps.StructUtils.extensions]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsStaticArraysCoreExt = ["StaticArraysCore"]
    StructUtilsTablesExt = ["Tables"]

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.8.3+2"

[[deps.SymbolicIndexingInterface]]
deps = ["Accessors", "ArrayInterface", "RuntimeGeneratedFunctions", "StaticArraysCore"]
git-tree-sha1 = "94c58884e013efff548002e8dc2fdd1cb74dfce5"
uuid = "2efcf032-c050-4f8e-a9bb-153293bab1f5"
version = "0.3.46"
weakdeps = ["PrettyTables"]

    [deps.SymbolicIndexingInterface.extensions]
    SymbolicIndexingInterfacePrettyTablesExt = "PrettyTables"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TZJData]]
deps = ["Artifacts"]
git-tree-sha1 = "72df96b3a595b7aab1e101eb07d2a435963a97e2"
uuid = "dc5dba14-91b3-4cab-a142-028a31da12f7"
version = "1.5.0+2025b"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "f2c1efbc8f3a609aadf318094f8fc5204bdaf344"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TaskLocalValues]]
git-tree-sha1 = "67e469338d9ce74fc578f7db1736a74d93a49eb8"
uuid = "ed4db957-447d-4319-bfb6-7fa9ae7ecf34"
version = "0.1.3"

[[deps.TensorCast]]
deps = ["ChainRulesCore", "Compat", "LazyStack", "LinearAlgebra", "MacroTools", "Random", "StaticArrays", "TransmuteDims"]
git-tree-sha1 = "67606dd3e705cc852c032975a6454e7130fed92c"
uuid = "02d47bb6-7ce6-556a-be16-bb1710789e2b"
version = "0.4.9"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.ThreadPools]]
deps = ["Printf", "RecipesBase", "Statistics"]
git-tree-sha1 = "50cb5f85d5646bc1422aa0238aa5bfca99ca9ae7"
uuid = "b189fb0b-2eb5-4ed4-bc0c-d34c51242431"
version = "2.1.1"

[[deps.TiffImages]]
deps = ["CodecZstd", "ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "PrecompileTools", "ProgressMeter", "SIMD", "UUIDs"]
git-tree-sha1 = "9ca5f1f2d42f80df4b8c9f6ab5a64f438bbd9976"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.11.9"

[[deps.TimeZones]]
deps = ["Artifacts", "Dates", "Downloads", "InlineStrings", "Mocking", "Printf", "Scratch", "TZJData", "Unicode", "p7zip_jll"]
git-tree-sha1 = "d422301b2a1e294e3e4214061e44f338cafe18a2"
uuid = "f269a46b-ccf7-5d73-abea-4c690281aa53"
version = "1.22.2"
weakdeps = ["RecipesBase"]

    [deps.TimeZones.extensions]
    TimeZonesRecipesBaseExt = "RecipesBase"

[[deps.ToeplitzMatrices]]
deps = ["AbstractFFTs", "DSP", "FillArrays", "LinearAlgebra"]
git-tree-sha1 = "338d725bd62115be4ba7ffa891d85654e0bfb1a1"
uuid = "c751599d-da0a-543b-9d20-d0a503d91d24"
version = "0.8.5"
weakdeps = ["StatsBase"]

    [deps.ToeplitzMatrices.extensions]
    ToeplitzMatricesStatsBaseExt = "StatsBase"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.TransmuteDims]]
deps = ["Adapt", "ChainRulesCore", "GPUArraysCore", "LinearAlgebra", "Requires", "Strided"]
git-tree-sha1 = "488b1a1effa8c0b97cd3335a7c1cbd23a3f2f6d3"
uuid = "24ddb15e-299a-5cc3-8414-dbddc482d9ca"
version = "0.1.17"

[[deps.TriplotBase]]
git-tree-sha1 = "4d4ed7f294cda19382ff7de4c137d24d16adc89b"
uuid = "981d1d27-644d-49a2-9326-4793e63143c3"
version = "0.1.0"

[[deps.TruncatedStacktraces]]
deps = ["InteractiveUtils", "MacroTools", "Preferences"]
git-tree-sha1 = "ea3e54c2bdde39062abf5a9758a23735558705e1"
uuid = "781d530d-4396-4725-bb49-402e4bee1e77"
version = "1.4.0"

[[deps.TupleTools]]
git-tree-sha1 = "41e43b9dc950775eac654b9f845c839cd2f1821e"
uuid = "9d95972d-f1c8-5527-a6e0-b4b365fa01f6"
version = "1.6.0"

[[deps.TypedTables]]
deps = ["Adapt", "Dictionaries", "Indexing", "SplitApplyCombine", "Tables", "Unicode"]
git-tree-sha1 = "84fd7dadde577e01eb4323b7e7b9cb51c62c60d4"
uuid = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"
version = "1.4.6"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "57e1b2c9de4bd6f40ecb9de4ac1797b81970d008"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.28.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    ForwardDiffExt = "ForwardDiff"
    InverseFunctionsUnitfulExt = "InverseFunctions"
    LatexifyExt = ["Latexify", "LaTeXStrings"]
    NaNMathExt = "NaNMath"
    PrintfExt = "Printf"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"
    LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
    Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
    NaNMath = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
    Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.VectorInterface]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9166406dedd38c111a6574e9814be83d267f8aec"
uuid = "409d34a3-91d5-4945-b6ec-7529ddf182d8"
version = "0.5.0"

[[deps.WGLMakie]]
deps = ["Bonito", "Colors", "FileIO", "FreeTypeAbstraction", "GeometryBasics", "Hyperscript", "LinearAlgebra", "Makie", "Observables", "PNGFiles", "PrecompileTools", "RelocatableFolders", "ShaderAbstractions", "StaticArrays"]
git-tree-sha1 = "dcf36e49ebbfe068cec38d413eca7c4839a1918f"
uuid = "276b4fcb-3e11-5398-bf8b-a0c2d153d008"
version = "0.13.9"

[[deps.WebP]]
deps = ["CEnum", "ColorTypes", "FileIO", "FixedPointNumbers", "ImageCore", "libwebp_jll"]
git-tree-sha1 = "aa1ca3c47f119fbdae8770c29820e5e6119b83f2"
uuid = "e3aaa7dc-3e4b-44e0-be63-ffb868ccd7c1"
version = "0.1.3"

[[deps.WidgetsBase]]
deps = ["Observables"]
git-tree-sha1 = "30a1d631eb06e8c868c559599f915a62d55c2601"
uuid = "eead4739-05f7-45a1-878c-cee36b57321c"
version = "0.1.4"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "248a7031b3da79a127f14e5dc5f417e26f9f6db7"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.1.0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b29c22e245d092b8b4e8d3c09ad7baa586d9f573"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.3+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "808090ede1d41644447dd5cbafced4731c56bd2f"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.13+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "1a4a26870bf1e5d26cd585e38038d399d7e65706"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.8+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "75e00946e43621e09d431d9b95818ee751e6b2ef"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.2+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libpciaccess_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "4909eb8f1cbf6bd4b1c30dd18b2ead9019ef2fad"
uuid = "a65dc6b1-eb27-53a1-bb3e-dea574b5389e"
version = "0.18.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.isoband_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51b5eeb3f98367157a7a12a1fb0aa5328946c03c"
uuid = "9a68df92-36a6-505f-a73e-abb412b6bfb4"
version = "0.2.3+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "850b06095ee71f0135d644ffd8a52850699581ed"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.13.3+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "125eedcb0a4a0bba65b657251ce1d27c8714e9d6"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.4+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.libdrm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "63aac0bcb0b582e11bad965cef4a689905456c03"
uuid = "8e53e030-5e6c-5a89-a30b-be5b7263a166"
version = "2.4.125+1"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "646634dd19587a56ee2f1199563ec056c5f228df"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.4+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e51150d5ab85cee6fc36726850f0e627ad2e4aba"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.58+0"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "libpng_jll"]
git-tree-sha1 = "c1733e347283df07689d71d61e14be986e49e47a"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.5+0"

[[deps.libva_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll", "Xorg_libXfixes_jll", "libdrm_jll"]
git-tree-sha1 = "7dbf96baae3310fe2fa0df0ccbb3c6288d5816c9"
uuid = "9a156e7d-b971-5f62-b2c9-67348b8fb97c"
version = "2.23.0+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.libwebp_jll]]
deps = ["Artifacts", "Giflib_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libglvnd_jll", "Libtiff_jll", "libpng_jll"]
git-tree-sha1 = "4e4282c4d846e11dce56d74fa8040130b7a95cb3"
uuid = "c5f90fcd-3b7e-5836-afba-fc50a0988cb2"
version = "1.6.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "1350188a69a6e46f799d3945beef36435ed7262f"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2022.0.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.7.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14cc7083fc6dff3cc44f2bc435ee96d06ed79aa7"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.1+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e7b67590c14d487e734dcb925924c5dc43ec85f3"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "4.1.0+0"
"""

# ╔═╡ Cell order:
# ╠═ab32a4c8-ef62-47ca-99a5-4e9f5a72f898
# ╠═54c0aa98-fd35-11f0-8dc8-4fb8474a532c
# ╠═e35e1679-971e-470e-9bf5-1efd6283485b
# ╠═bfa16f35-e8ff-4605-a4bc-df385438a019
# ╠═a9b3177c-b619-49a2-b839-5e93742a795b
# ╠═99664353-1ca2-406c-9fc9-c9e0335bb76e
# ╠═ec4fec4a-8c22-450a-9721-c4b21f5957f6
# ╠═80242e98-449b-41ab-aaf1-6e054dffdbb1
# ╠═a0356246-980e-4725-bdb1-f067ca5d773c
# ╠═390dea24-024e-4c10-8ec1-991c3682ab6a
# ╠═291983e2-bb22-47f1-a745-a3849ff1f7bd
# ╠═31f01e7f-48a0-4105-8379-217ead9d7031
# ╠═a4b08932-9ea9-4d25-a052-4060e7bd5bee
# ╠═641e4213-4017-4694-b185-ed6b25a36ba4
# ╠═56d20146-8bc1-4bbe-a581-ab0301e89be7
# ╠═85435207-3685-45fb-937d-396eeb19e18e
# ╠═c49b6a97-56fd-4a50-ab47-63a384c61073
# ╠═be347205-af4f-49ef-9b8d-037613d94d1b
# ╠═0515ab0f-bfde-4a0d-988d-1dcb36e17478
# ╠═817eaab2-703d-4605-805b-4efc86f8bbb7
# ╠═a6addf6f-278c-4f7d-8063-a79b5193dba7
# ╠═ffea356e-d9f5-48e4-b79b-6fc7c61c03ca
# ╠═57e76596-23e0-42e9-8322-a415030c3186
# ╠═6d81d722-9f4a-4328-9ad7-df4dc738ce33
# ╠═af8adeda-3f36-4a70-9090-849a8bef35a3
# ╠═bd1f7abc-716c-4d9a-97f7-2b03f7e5876b
# ╠═f1645e25-2def-4fb3-861b-cb2a6801a250
# ╠═f5a11c98-a7b8-45d6-8f2b-4a880911a073
# ╠═2a32b857-3803-478e-9425-b3cb45aefd13
# ╠═de2c7e75-e8b1-4d58-871a-1a779f0c0358
# ╠═5f99c038-67b9-4b7d-893b-90279597f7c1
# ╠═6a9f7aed-f542-4494-9072-d7530b85d030
# ╠═d4ccb607-ff22-4718-8271-dde92d790da3
# ╠═df695fff-9770-4ed5-8702-719db2fefc6f
# ╠═ff34a5d9-6df4-4804-957b-f9442934ebf9
# ╠═8e9dec92-5175-4e94-b0dd-8b32887464e5
# ╠═e983f0ad-93b2-40dc-947b-86637c30da50
# ╠═92a81d80-5a3b-4869-b514-d4f86966431d
# ╠═1256d1b5-c862-404b-9949-564ff55c2339
# ╠═99d6765c-dfdb-4dad-840c-f7e99faaf199
# ╠═d409eccf-1971-40f7-8b16-4dfaedc0792c
# ╠═28157a28-fc41-4c2d-aa4f-eae711b77b66
# ╠═e01f28ee-240d-4eea-ba29-ba90603f69ef
# ╠═dc2bcab5-14cd-4919-a395-c7bdae7d1764
# ╠═f7b4b8d8-6607-4dda-8e2d-00714ed7c913
# ╠═1512ed7f-9660-4a4c-90f8-9b7d97ac2fa8
# ╠═dab7adf3-8500-44d5-9f39-2f6efb73e136
# ╠═bf652d8f-c51f-463e-b386-02606346e25e
# ╠═b3ef3789-26c9-45e9-be14-a3cf36c7a8e1
# ╠═59e8fa13-4e28-49b7-b78e-424d9b7a765f
# ╠═b545d6b1-b3cc-4268-b8dc-409b8e1bddf2
# ╠═ad4470f3-0f08-422c-be29-062aabacda77
# ╠═b85d1fee-d2a9-429e-87bc-066c01ece42e
# ╠═1b9964f6-8d8a-4731-b7f5-dc42b25a0c19
# ╠═be1925cf-d859-47b9-ac08-8f962e4ceb70
# ╠═c09eeec0-0958-4fd1-87ce-756a5f1def12
# ╠═69fae8b8-56ba-401b-9895-0b5d1346e86f
# ╠═ceecbb7f-a41d-4882-81eb-dd6306c78378
# ╠═a2ceb55d-1c72-490e-858f-4795d8e1c3cb
# ╠═5b22deaa-6087-4c76-b530-04e594a5e705
# ╠═ce812a23-9084-4671-b3bc-9f5693165186
# ╠═f188a104-c693-4005-bd5d-766bf056ed53
# ╠═72d85e49-3294-4568-b833-dda51fc38653
# ╠═7e14cebd-c78e-490b-a5c3-db1f6c2e2754
# ╠═b08fbcf5-5486-4d87-a4fa-44d8fe6f782c
# ╠═0b444cd0-ca0e-4109-8fc6-23bb7c499751
# ╠═f2c0ebc8-73e1-4468-8335-ca6d02e9543d
# ╠═0ed0bc4a-0ad1-4716-9eb2-4a06e7a58ede
# ╠═3e4b06a0-96a5-432e-b70a-3c222cbabfe4
# ╠═b0c54a2c-abe0-4ce8-9f50-61d25119d872
# ╠═31a32ac1-c4ef-4f5f-b350-0f7ed9ca839d
# ╠═fc5925a1-4b9a-4d74-a290-43a6ff147d8e
# ╠═6ab62af9-d3a3-48f0-b4e1-74f19a0de1e2
# ╠═2fb8d3a9-4afa-41d4-9c2d-a883565b8a71
# ╠═f604b534-86c4-4371-9f71-38f6846c84e7
# ╠═9e7b838e-7e94-4a83-8404-faa921fb1b5d
# ╠═3805f0d6-cd08-4fac-871d-6adfe36aead8
# ╠═f0813f13-dab7-47c9-8603-e885bcc4d451
# ╠═0e9b6628-bb12-4db3-b2fe-e93151b685f9
# ╠═778f3aed-0685-4422-a782-72927dd99d35
# ╠═288cae69-47ab-44ca-8134-c5809347fee3
# ╠═d242b2ca-264d-42e8-8368-f55dbac3a37b
# ╠═f0d08e48-1b50-4e11-8cb8-9908ef5ca115
# ╠═1e269a47-8f8d-48b9-97e9-5164c2b1e045
# ╠═fe5e43cf-fde2-4535-b914-a10a14e17f46
# ╠═439390ab-4e4b-4a01-9806-5c06b331dbc8
# ╠═1de56646-dc61-4e3a-864b-127053f29d24
# ╠═3ce601a3-2c80-4a15-80fc-8f2091cc9d0a
# ╠═d83c9bdb-c1b4-44aa-9273-0e9e9b244009
# ╠═bbb3b411-1f08-471e-b99c-3f36927e31b2
# ╠═eca631ec-38fa-4740-9dd1-96df841f752b
# ╠═89bda349-3a7a-46b3-ba57-62677a0c5e76
# ╠═b2362bbd-67c5-4913-8d8b-37ea6a12f49b
# ╠═606c23f6-05fd-4162-b1de-735d050eafef
# ╠═5e45b1c4-a317-4c33-95af-3cc4acc6367a
# ╠═f064608d-a0e4-4d21-a102-e492237d7d19
# ╠═2acb6e61-4558-4e14-95d2-10e98ebc12cb
# ╠═5922cf28-e9ef-4d0d-bbae-367318657522
# ╠═2d40242e-5dd5-42b3-9013-6011d9eb6f23
# ╠═8d97564e-98b5-4fbe-8467-4c6132e22914
# ╠═a80df87d-07b4-453f-8a58-93b86b87c52f
# ╠═03b05d6c-8f49-4a16-872b-3e2062366bfa
# ╠═fe654ee0-162a-4ab8-abec-cdfd38a7c1d2
# ╠═4a8919e4-449c-415e-97bc-d2c09d041da7
# ╠═a4de208b-2bfb-4235-a2a9-e409a5383cb1
# ╠═4d35d207-a4b8-43c6-8fba-7beb82456750
# ╠═a4b2315b-3b60-41b1-ae34-441c9365fc40
# ╠═40b78395-9ee6-47f8-b6df-80053c6b532e
# ╠═ec024d67-3212-47ed-89c2-7bff802a9786
# ╠═0ab944fa-4bdb-4057-ad90-ae170fc815a7
# ╠═fedb8d40-b3ca-496a-a283-f6fac1221efb
# ╠═6199d1b0-b438-4e31-ad22-6cb462766e86
# ╠═0f47c966-4195-4c13-9379-42cf428a12e2
# ╠═254bc018-1632-4960-a70d-80f908c1533d
# ╠═7da45d2c-1038-46ba-bbec-99a60fd5ad18
# ╠═0fd9d389-534b-47ad-b79e-44a78e7c4a5a
# ╠═bd09a846-afe4-44a3-93b9-0c9678f096aa
# ╠═4d9d74bf-d9e3-4490-84f1-79011ad27d1d
# ╠═e3338264-be74-440b-b099-0f0d12daf383
# ╠═798a73e9-1d84-4f7e-b2d1-9c89faea4c15
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
