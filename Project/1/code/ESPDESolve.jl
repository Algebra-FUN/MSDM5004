# Program to solve the PDE for evolution of planar stressed surface with FFT method.
# Copyright (c) 2023, Y. FAN

include("FFTWrapper.jl")

using Plots

f̂(ĥ,ω;r=1.5) = r^2*(12conv(abs.(ω) .* ĥ)-8conv(ĥ,ω .^ 2 .* ĥ)-4conv(ω .* ĥ))

function solve(ĥ⁰::Vector,N;L=2π,Δt=0.01,r=1.5,M=1000)
    ω = 2π*rfftfreq(N,N/L)
    λ = @. 4r^3*abs(ω)^3-r^4*ω^4
    f̂⁰ = f̂(ĥ⁰,ω;r=r)
    update(ĥⁿ,f̂ⁿ,f̂ⁿ⁻¹) = @. ((1/Δt+λ/2)*ĥⁿ-r^2*ω^2*(3*f̂ⁿ/2-f̂ⁿ⁻¹/2))/(1/Δt-λ/2)
    ĥ = zeros(ComplexF64,length(ĥ⁰),M)
    t = 0:Δt:M*Δt
    ĥ[:,1] = update(ĥ⁰,f̂⁰,f̂⁰)
    f̂ⁱ⁻¹ = f̂⁰
    for i in 1:M-1
        f̂ⁱ = f̂(ĥ[:,i],ω;r=r)
        ĥ[:,i+1] = update(ĥ[:,i],f̂ⁱ,f̂ⁱ⁻¹)
        f̂ⁱ⁻¹ = f̂ⁱ
    end
    return t,[ĥ⁰ ĥ]
end

N = 20
M = 30
L = 2π
Δts = [0.01;0.01;0.002]
rs = [1.5;3.8;5.]
h₀(x) = 0.01cos(x)
ĥ⁰ = rfftemplate(N)
ĥ⁰[1+1] = 0.01/2

hs = Dict()
ĥs = Dict()
ts = Dict()
for (r,Δt) in zip(rs,Δts)
    t,ĥ = solve(ĥ⁰,N;L=L,r=r,Δt=Δt,M=M)
    ĥs[r] = ĥ
    ts[r] = t
    hs[r] = zeros(Float64,N,M+1)
    for t in 1:M+1
       hs[r][:,t] = irfft(ĥ[:,t],N) 
    end
end

gr()
x = range(0,2π,length=N+1)
for r in rs
    p = plot(size = (600, 400), title="r=$r")
    t = ts[r]
    h = hs[r]
    for i in range(1,stop=M,step=floor(Int,M/7))
        plot!(p, x, [h[:,i];h[1,i]] ,label="t=$(t[i])")
    end
    xlabel!(p,"x")
    ylabel!(p,"h")
    idxs = Vector(0:0.5:2)
    xticks!(p,(π * idxs,string.(idxs).* " π"))
    display(p)
    savefig(p,"../img/surface_snapshot(r=$r).pdf")
end

gr()
for r in rs
    p = surface(hs[r],size=(600,500))
    xlabel!(p,"t")
    tidxs = Vector(range(1,M+1,step=10))
    xticks!(p,(tidxs,string.(ts[r][tidxs])))
    idxs = Vector(range(0,N,length=5))
    yticks!(p,(idxs,string.(2*idxs/(N)).* " π"))
    ylabel!(p,"x")
    zlabel!(p,"h")
    title!(p,"r=$r")
    display(p)
    savefig(p,"../img/surface_3d(r=$r).pdf")
end