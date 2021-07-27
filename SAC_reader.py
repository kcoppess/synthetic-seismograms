import numpy as np
import matplotlib.pyplot as plt
import sys
import sacpy
import os

args = sys.argv[1:]

model = args[0]
os.chdir(model)
filename = args[1]
f = sacpy.sac(filename)
data_len = f.npts
f_list = f.depvar
dt = f.delta

x = []
for i in range(data_len):
    x.append(i * dt)

a = 1
b = 10
c = 1
gaussian = []
for t in x:
    gaussian.append(a * np.exp(-(t - b)**2 / (2 * c**2)))

gaussian_fourier = np.fft.fft(gaussian) * dt
f_fourier = np.fft.fft(f_list) * dt
convolve = f_fourier * gaussian_fourier
final = np.fft.ifft(convolve) / dt

plt.plot(x, final)
plt.show()

