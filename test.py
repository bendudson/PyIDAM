#!/usr/bin/env python

import idam

d = idam.Data("amc_plasma current", 15100)

import matplotlib.pyplot as plt

plt.plot(d.dim[0].data, d.data)
plt.xlabel(d.dim[0].label)
plt.ylabel(d.label)
plt.show()







