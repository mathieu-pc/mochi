"""
Testing velocity centering.
With the new adaptive optically thin algorithm, it is not trivial that velocity centering is preserved.
"""
import numpy as np
from astropy import units
from martini.sph_kernels import _QuarticSplineKernel
from radio_beam import Beam
import pytest

from Mochi import Interpolants
from Mochi import RadiativeTransfer
import Mochi


def test_velocity_center():
	cellMasses = np.array([1,-1]) * units.Msun
	cellVelocities = np.array([0, 25]) * units.km / units.s
	cellDispersions = np.array([2, 2]) * (units.km / units.s) ** 2
	channelWidth = 1 * units.km / units.s
	cellVolume = 1 * units.kpc ** 3
	volumeShape = (2,1,1)
	cells = np.array([[0,0,0,1], [1,0,0,1]])
	result = RadiativeTransfer.adaptiveOpticallyThin(
		cellMasses,
		cellVelocities,
		cellDispersions,
		channelWidth,
		cellVolume,
		volumeShape,
		cells = cells
	).squeeze()
	assert np.argmax(result) == result.shape[0] // 2  # the maximum placed at 0 should be centered in the spectrum.