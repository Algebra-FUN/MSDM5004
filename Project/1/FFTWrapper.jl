import FFTW

function conv(f,g)
    h = zero(f)
    N = length(f)
    for k in eachindex(h)
        h[k] = sum(m->f[m]*g[mod(k-m,N)+1],eachindex(f))
    end
    return h
end

conv(f) = conv(f,f)

rfftfreq = FFTW.rfftfreq
fftfreq = FFTW.fftfreq

rfftemplate(N) = zero(FFTW.rfftfreq(N))
fftemplate(N) = zero(FFTW.fftfreq(N))

fft(x) = FFTW.fft(x)/length(x)
rfft(x) = FFTW.rfft(x)/length(x)

ifft(x) = FFTW.ifft(x)*length(x)
irfft(x,N) = FFTW.irfft(x,N)*N