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

from tests.testSource import generateTestParticles

kernel = _QuarticSplineKernel().kernel #Mochi accepts base martini kernels
np.random.seed(0)
particleNumber = 10000
particles = generateTestParticles(particleNumber, 20)
small_particle = {
	"m": np.ones(1) * units.Msun,
	"T_g": np.ones(1) * (units.km / units.s) ** 2,
	"mHI_g": np.ones(1) * units.Msun,
	"xyz_g": np.zeros((1,3)) * units.kpc,
	"vxyz_g": np.zeros((1,3)) * units.km / units.s,
	"hsm_g": np.ones(1) * 1e-8 * units.kpc
}
cold_particle = {
	"m": np.ones(1) * units.Msun,
	"T_g": np.zeros(1) * (units.km / units.s) ** 2,
	"mHI_g": np.ones(1) * units.Msun,
	"xyz_g": np.zeros((1,3)) * units.kpc,
	"vxyz_g": np.zeros((1,3)) * units.km / units.s,
	"hsm_g": np.ones(1) * units.kpc
}

@pytest.mark.parametrize(
	"interpolant, particles", [
	(Interpolants.SPH, small_particle),
	(Interpolants.SPH, cold_particle),
	(Interpolants.MFM, small_particle),
])
def test_run(interpolant, particles):
	"""
	Test graceful failure cases.
	These should one day be resolved and moved to a mass conservation test.
	"""
	galaxyDistance = 10 * units.Mpc
	pixelSize = 6 * units.arcsec
	channelSize = 4 * units.km / units.s
	boxLength = 2 * np.max(np.abs(particles["xyz_g"]) + particles["hsm_g"][:, np.newaxis])
	pixelNumber = int(np.ceil(((boxLength / galaxyDistance) * units.rad / pixelSize.to(units.rad)).decompose()))
	pixelNumber = max(pixelNumber, 10)
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
	assert not np.any(np.isnan(cube))
	assert not np.any(np.isinf(cube))


@pytest.mark.parametrize(
	"interpolant, particles", [
	(Interpolants.SPH, particles),
	(Interpolants.MFM, particles),
])
def test_interpolant_mass(interpolant, particles, tol = 1e-4):
	galaxyDistance = 10 * units.Mpc
	pixelSize = 6 * units.arcsec
	channelSize = 4 * units.km / units.s
	boxLength = 2 * np.max(np.abs(particles["xyz_g"]) + particles["hsm_g"][:, np.newaxis])
	pixelNumber = int(np.ceil(((boxLength / galaxyDistance) * units.rad / pixelSize.to(units.rad)).decompose()))
	pixelNumber = max(pixelNumber, 10)
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
	assert np.abs(massRatio - 1.) < tol

