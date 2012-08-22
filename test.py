#!/usr/bin/env python

import idam
import matplotlib.pyplot as plt

# Read a 1D dataset (plasma current) and plot
d = idam.Data("amc_plasma current", 15100)
plt.title("Plasma Current")
plt.plot(d.time, d.data)
plt.xlabel(d.dim[0].label)
plt.ylabel(d.label)
plt.show()

# Read a 3D array (psi) and plot a time-slice
d = idam.Data("efm_psi(r,z)", 23320)
f = plt.figure()
plt.title("EFIT magnetic reconstruction")
plt.contour(d.data[20,:,:], 50)
plt.show()






