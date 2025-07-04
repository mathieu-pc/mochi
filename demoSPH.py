"""
Mochi demo using synthetic data
"""

import numpy as np
from astropy import units
from matplotlib import pyplot as plt
from martini.sph_kernels import _QuarticSplineKernel
from Mochi import Interpolants
import Mochi
from demoSource import demoSource

kernel = _QuarticSplineKernel().kernel #Mochi accepts base martini kernels

N = 1000
particles = demoSource(N)

wallaby = {
	"beam sigma": 30 * units.arcsec / (2 * np.sqrt(2 * np.log(2))),
	"pixel size": 6 * units.arcsec,
	"channel width": 4 * units.km / units.s
}
galaxyDistance = 10 * units.Mpc
pixelNumber = 120

cube = Mochi.makeCube(
	galaxyDistance,
	particles,
	kernel,
	pixelNumber,
	wallaby["pixel size"],
	wallaby["channel width"],
	wallaby["beam sigma"],
	Interpolants.SPH,
	convolveMode = True,
	resizeMode = True
)

plt.imshow(np.sum(cube.value, axis = 0)) #moment0 map
plt.show()

#from astropy.io import fits
#cards = [fits.Card("OBJ", "mochi_test" + str(N))]
#header = fits.Header(cards)
#fits.writeto("mochitest.fits", cube.value, header = header, overwrite = True)