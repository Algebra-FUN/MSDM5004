"""
This file contains a wrapper for the FFTW library.
"""

import FFTW

fftfreq = FFTW.fftfreq

fftemplate(N) = zero(FFTW.fftfreq(N))

fft(x) = FFTW.fft(x) / length(x)
ifft(x) = FFTW.ifft(x) * length(x)

conv(f, g) = ifft(fft(f) .* fft(g))
conv(f) = conv(f, f)