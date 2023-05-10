"""
This file contains a wrapper for the FFTW library.
"""

import FFTW

function conv(f, g)
    h = zero(f)
    N = length(g)
    for k in eachindex(h)
        # h[k] = sum(m->f[m]*g[mod(k-m,N)+1],eachindex(f))
        h[k] = sum(m -> 0 <= k - m < N ? f[m] * g[k-m+1] : 0, eachindex(f))
    end
    return h
end

conv(f) = conv(f, f)

fftfreq(N, fs) = FFTW.fftshift(FFTW.fftfreq(N, fs))

fftshift = FFTW.fftshift
ifftshift = FFTW.ifftshift

fftemplate(N) = zero(FFTW.fftfreq(N))

fft(x) = FFTW.fftshift(FFTW.fft(x) / length(x))

ifft(x) = FFTW.ifft(FFTW.ifftshift(x)) * length(x)