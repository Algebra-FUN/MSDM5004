# Program to solve the PDE for evolution of planar stressed surface with FFT method.
# Copyright (c) 2023, Y. FAN

include("FFTWrapper.jl")

using Plots

f̂(ĥ,ω;r=1.5) = r^2*(4conv(abs.(ω) .* ĥ)+8abs.(ω) .* conv(ĥ,abs.(ω) .* ĥ)-8conv(ĥ,ω .^ 2 .* ĥ)-4conv(ω .* ĥ))

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

N = 50
M = 30
L = 2π
Δts = [0.005;0.005;0.001]
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