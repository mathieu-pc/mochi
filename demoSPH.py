"""
Mochi demo using synthetic data
"""

import numpy as np
from astropy import units
from matplotlib import pyplot as plt
from martini.sph_kernels import _QuarticSplineKernel
from radio_beam import Beam

from Mochi import Interpolants
import Mochi
from demoSource import demoSource

kernel = _QuarticSplineKernel().kernel #Mochi accepts base martini kernels

N = 1000
particles = demoSource(N)

wallaby = {
	"beam": Beam(30 * units.arcsec),
	"pixel size": 6 * units.arcsec,
	"channel width": 4 * units.km / units.s,
	"noise rms": 1.6e-3 * units.Jy / units.beam
}
galaxyDistance = 10 * units.Mpc
pixelNumber = 100

cube = Mochi.makeCube(
	galaxyDistance,
	particles,
	kernel,
	pixelNumber,
	wallaby["pixel size"],
	wallaby["channel width"],
	wallaby["beam"],
	Interpolants.SPH,
	adaptiveMode = True,
	convolveMode = True,
	resizeMode = True
)

noiseCube = (
	Mochi.PostProcessing.getJyFromMass(cube, wallaby["beam"], wallaby["pixel size"], wallaby["channel width"], galaxyDistance)
	+ Mochi.PostProcessing.getNoiseCube(cube.shape, wallaby["noise rms"], wallaby["beam"], pixelSize = wallaby["pixel size"])
)

plt.imshow(np.sum(noiseCube.value, axis = 0)) #moment0 map
plt.show()

#from astropy.io import fits
#cards = [fits.Card("OBJ", "mochi_test" + str(N))]
#header = fits.Header(cards)
#fits.writeto("mochitest.fits", cube.value, header = header, overwrite = True)