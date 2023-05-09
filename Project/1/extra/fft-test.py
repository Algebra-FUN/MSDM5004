# %%
import numpy as np
from matplotlib import pyplot as plt

N = 201
L = np.pi
xj = np.linspace(0,L,N)
f = np.sin(2*xj)
df = 2*np.cos(2*xj)

k = 2*np.pi*np.fft.fftfreq(N,L/N)
df_fft = np.fft.ifft(1j * k * np.fft.fft(f) )

plt.plot(xj,np.real(df_fft),marker="o",label="by_fft")
plt.plot(xj,df,label="theoretical")
plt.legend()
plt.grid(True)
plt.show()

# %%
d2f= -4*np.sin(2*xj)
d2f_fft = np.fft.ifft(- k ** 2 * np.fft.fft(f) )

plt.plot(xj,np.real(d2f_fft),marker="o",label="by_fft")
plt.plot(xj,d2f,label="theoretical")
plt.ylim(-5,5)
plt.legend()
plt.grid(True)
plt.show()


