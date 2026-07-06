"""
test particle mass conservation when converting particles to HI cube
"""
import numpy as np
from astropy import units
from martini.sph_kernels import _QuarticSplineKernel
from radio_beam import Beam
import pytest

from Mochi import Interpolants
from Mochi import RadiativeTransfer
import Mochi

from testSource import generateTestParticles

kernel = _QuarticSplineKernel().kernel #Mochi accepts base martini kernels
np.random.seed(0)
particleNumber = 10000
particles = generateTestParticles(particleNumber, 20)


@pytest.mark.parametrize(
	"interpolant, particles", [
	(Interpolants.SPH, particles),
	(Interpolants.MFM, particles)
])
def test_interpolant_mass(interpolant, particles, tol = 1e-4):
	galaxyDistance = 10 * units.Mpc
	pixelSize = 6 * units.arcsec
	channelSize = 4 * units.km / units.s
	boxLength = 2 * np.max(particles["xyz_g"] + particles["hsm_g"][:, np.newaxis])
	pixelNumber = int(np.ceil(((boxLength / galaxyDistance) * units.rad / pixelSize.to(units.rad)).decompose()))

	cube = Mochi.makeCube(
		galaxyDistance,
		particles,
		kernel,
		pixelNumber,
		pixelSize,
		channelSize,
		interpolant,
		adaptiveMode = True,
		resizeMode = True
	)
	totalParticleMass = np.sum(particles["mHI_g"])
	totalCubeMass = np.sum(cube)
	massRatio = (totalParticleMass / totalCubeMass).decompose().value
	print(massRatio-1.)
	assert np.abs(massRatio - 1.) < tol

if __name__ == "__main__":
	testAll()

