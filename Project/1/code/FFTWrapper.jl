"""
This file contains a wrapper for the FFTW library.
"""

import FFTW

fftfreq = FFTW.fftfreq

fftemplate(N) = zero(FFTW.fftfreq(N))

fftshift(x) = FFTW.fftshift(x)
ifftshift(x) = FFTW.ifftshift(x)

fftfreq = fftshift ∘ FFTW.fftfreq

fft(x) = fftshift(FFTW.fft(x)) / length(x)
ifft(x) = FFTW.ifft(ifftshift(x)) * length(x)

# fftconv(f, g) = FFTW.ifft(FFTW.fft(f) .* FFTW.fft(g))
# fftconv(f, g) = DSP.conv(f,g)[1:length(f)]

function conv(f, g, method=:zero_padding)
    h = zero(f)
    N = length(g)
    if method == :zero_padding
        for k in eachindex(h)
            h[k] = sum(m -> 0 <= k - m < N ? f[m] * g[k-m+1] : 0, eachindex(f))
        end
        return h
    elseif method == :circular
        for k in eachindex(h)
            h[k] = sum(m -> f[m] * g[mod(k-m,N)+1], eachindex(f))
        end
        return h
    end
    throw(ArgumentError("Invalid method"))
end
conv(f) = conv(f, f)