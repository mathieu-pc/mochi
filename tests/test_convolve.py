"""
test that convolve function gives the appropriate padding
"""
import numpy as np
from astropy import units
from radio_beam import Beam
import pytest

from Mochi.PostProcessing import convolve

def test_convolve():
	initial_cube_shape = (10,10,10)
	pad_value = 10
	expected_cube_shape = tuple(x + 2 * pad_value for x in initial_cube_shape)
	cube = np.empty(initial_cube_shape, dtype = float) * units.Msun
	beam = Beam(30 * units.arcsec)
	pixelSize = 5 * units.arcsec
	convolved_cube = convolve(cube, beam, pixelSize, pad = 10 * units.pixel)
	assert convolved_cube.shape == expected_cube_shape

