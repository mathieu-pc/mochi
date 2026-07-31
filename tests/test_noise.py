"""
test that noise follows input standard deviation and expected drift
"""
import numpy as np
from astropy import units
from radio_beam import Beam
import pytest

from Mochi.PostProcessing import getNoiseCube

wallaby = {
	"beam": Beam(30 * units.arcsec),
	"pixel size": 6 * units.arcsec,
	"channel width": 4 * units.km / units.s,
	"noise rms": 1.6e-3 * units.Jy / units.beam
}

def test_noise():
	shape = (100,50,50)
	N = np.prod(shape)
	N_effective = N * (wallaby["pixel size"] ** 2 / wallaby["beam"].sr).decompose()
	noise_cube = getNoiseCube(shape, wallaby["noise rms"], wallaby["beam"], pixelSize = wallaby["pixel size"])
	rms = np.sqrt(np.average(noise_cube ** 2))
	relative_noise = np.abs(rms - wallaby["noise rms"])/wallaby["noise rms"]
	relative_drift = np.abs(np.average(noise_cube) / wallaby["noise rms"])

	tolerance_drift = 3 / np.sqrt(N_effective) #I think drift is expected ~ 1/sqrt(N_effective) so tolerance is tripled
	#note that my estimate ignores edge effects
	tolerance_noise = 0.02 #arbitrary value. This should be refined with statistics
	
	print(relative_noise, tolerance_noise)
	print(relative_drift, tolerance_drift)
	assert relative_noise < tolerance_noise
	assert relative_drift < tolerance_drift