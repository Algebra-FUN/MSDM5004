"""
This file contains a wrapper for the FFTW library.
"""

import FFTW

rfftfreq = FFTW.rfftfreq
rfftemplate = zero ∘ FFTW.rfftfreq

rfft(x) = FFTW.rfft(x)/length(x)
irfft(x,N) = FFTW.irfft(x,N)*N

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