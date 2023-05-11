"""
This file contains a wrapper for the FFTW and DSP library.
"""

import FFTW, DSP

fftfreq = FFTW.fftfreq

fftemplate(N) = zero(FFTW.fftfreq(N))

fft(x) = FFTW.fft(x) / length(x)
ifft(x) = FFTW.ifft(x) * length(x)

conv(f, g) = DSP.conv(f, g)[1:length(f)]
conv(f) = conv(f, f)