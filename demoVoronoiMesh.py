"""
Mochi demo using synthetic data for voronoi mesh codes
"""

import numpy as np
from astropy import units
from matplotlib import pyplot as plt
from Mochi import Interpolants, refineGridToOccupancy
import Mochi
from demoSource import demoSource

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
	None,					#kernel does not need to be specified
	pixelNumber,
	wallaby["pixel size"],
	wallaby["channel width"],
	wallaby["beam sigma"],
	Interpolants.voronoiMesh,		#use voronoi mesh interpolant
	convolveMode = True,
	resizeMode = True,
	refineAlgorithm = refineGridToOccupancy	#recommended adaptive resolution for voronoi mesh
)

plt.imshow(np.sum(cube.value, axis = 0)) #moment0 map
plt.show()

#from astropy.io import fits
#cards = [fits.Card("OBJ", "mochi_test" + str(N))]
#header = fits.Header(cards)
#fits.writeto("mochitest.fits", cube.value, header = header, overwrite = True)