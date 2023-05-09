using FFTW
using Plots

plotly()

N = 101;
L = 1;
xj = (0:N-1)*L/N;
f = sin.(2π*xj)
df = 2π*cos.(2π*xj)

k = 2π*rfftfreq(N,N/L)
df_fft = irfft( im * k.* rfft(f), N)
# k = fftfreq(N)*N;
# df_fft = ifft( 2π*im/L * k.* fft(f) );

plot(xj,real(df),label="Exact derivative")
plot!(xj,real(df_fft),label="incorrect FFT derivative",markershape=:circle)

N = 2^10
x₀ = -π/2
xₙ = π/2
L = xₙ - x₀
xj = range(x₀,xₙ,N)
f = sin.(2xj)
df = 2cos.(2xj)

k = 2π*rfftfreq(N,N/L)
f̂ = rfft(f)
f̂[3:end] .= 0
df_fft = irfft(im * k .* f̂,N)

plot(xj,real.(df),label="Exact derivative")
plot!(xj,real.(df_fft),label="correct FFT derivative")


N = 2^10
L = 2π
xj = range(0,L,N)
f = sin.(2xj)
df = 2cos.(2xj)
d²f= -4sin.(2xj)

k = 2π*rfftfreq(N,N/L)
f̂ = rfft(f)
f̂[4:end] .= 0
df_fft = irfft(im * k .* f̂,N)
d²f_fft = irfft(- k.^2 .* f̂,N)

plot(xj,real(df),label="Exact derivative")
plot!(xj,real(df_fft),label="correct FFT derivative")
plot!(xj,real(d²f),label="Exact derivative2")
plot!(xj,real(d²f_fft),label="correct FFT derivative2")