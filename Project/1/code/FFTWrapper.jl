"""
This file contains a wrapper for the FFTW and DSP library.
"""

import FFTW, DSP

fftfreq = FFTW.fftfreq

fftemplate = zero ∘ FFTW.fftfreq

fftshift = FFTW.fftshift
ifftshift = FFTW.ifftshift

fftfreq = fftshift ∘ FFTW.fftfreq

fft(x) = fftshift(FFTW.fft(x)) / length(x)
ifft(x) = FFTW.ifft(ifftshift(x)) * length(x)

# fftconv(f, g) = fftshift(FFTW.ifft(FFTW.fft(ifftshift(f)) .* FFTW.fft(ifftshift(g))))
# fftconv(f, g) = fftshift(DSP.conv(ifftshift(f),ifftshift(g))[1:length(f)])
# fftconv(f) = fftconv(f, f)

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