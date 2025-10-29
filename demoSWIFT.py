"""
This provides utility to using SWIFT to make mock HI cubes with MOCHI.
This demo is incomplete - there is no accompanying demo SWIFT galaxy data.
"""
from astropy import units, constants
import Mochi

def swiftHIMass(gas):
	"""
	Calculate particle MHI from swiftgalaxy.gas
	"""
	return gas.masses.to_astropy() * gas.element_mass_fractions.hydrogen.to_astropy() * gas.species_fractions.HI.to_astropy()

def swiftGalaxyToParticle(galaxy):
	h = galaxy.metadata.cosmology.h
	mHI_g = swiftHIMass(galaxy.gas)
	kernel_function = galaxy.metadata.hydro_scheme["Kernel function"].decode()
	compact_support_per_h = {
		"Quartic spline (M5)": 2.018932,
		"Quintic spline (M6)": 2.195775,
		"Cubic spline (M4)": 1.825742,
		"Wendland C2": 1.936492,
		"Wendland C4": 2.207940,
		"Wendland C6": 2.449490
	}[kernel_function]
	hsm_g = galaxy.gas.smoothing_lengths.to_astropy() * compact_support_per_h
	particles = dict(
		xyz_g = galaxy.gas.coordinates.to_astropy(),
		vxyz_g = galaxy.gas.velocities.to_astropy(),
		T_g = (galaxy.gas.temperatures.to_astropy() * constants.k_B / constants.m_p).to( (units.km / units.s) ** 2),
		hsm_g = hsm_g,
		mHI_g = mHI_g,
		m = galaxy.gas.masses.to_astropy()
	)
	return particles

def mochiFromSwiftGalaxy(distance, swiftGalaxy, kernel, pixelNumber, pixelSize, channelWidth, interpolant, radiativeTransfer, **kwargs):
	return Mochi.makeCube(
		distance,
		swiftGalaxyToParticle(swiftGalaxy),
		kernel,
		pixelNumber,
		pixelSize,
		channelWidth,
		interpolant,
		radiativeTransfer,
		**kwargs
	)