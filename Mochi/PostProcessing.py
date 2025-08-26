"""
This is to handle post processing of mochi cubes
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from astropy import units

from .mochiUtils import _astropyUnitWrap

SIGMA_TO_FWHM_FACTOR = 2 * np.sqrt(2 * np.log(2))

def convertBeamToBeamSigma(beam, spectralSigma, pixelSize):
	beamFactor = 1 / pixelSize / SIGMA_TO_FWHM_FACTOR
	return (spectralSigma, beam.major * beamFactor, beam.minor * beamFactor)

def getNoiseCube(shape, noiseRMS, beam, pixelSize, spectralSigma = 0):
	beamSigma = convertBeamToBeamSigma(beam, spectralSigma, pixelSize)
	noiseRMSPreBeam = noiseRMS * 2 * np.sqrt(beamSigma[1] * beamSigma[2]) * np.sqrt(np.pi)
	cube = np.random.normal(loc = 0, scale = noiseRMSPreBeam.value, size = shape)
	cube = gaussian_filter(cube, beamSigma, mode = "wrap")
	return cube * noiseRMSPreBeam.unit

def convolve(cube, beam, pixelSize, spectralSigma = 0, pad = 2):
	beamSigma = convertBeamToBeamSigma(beam, spectralSigma, pixelSize)
	unitlessCube, unit = _astropyUnitWrap(cube)
	if pad != 0:
		padWidth = [(int(sigma * pad + 0.5),) * 2 for sigma in beamSigma]
		unitlessCube = np.pad(unitlessCube, padWidth, 'constant', constant_values = 0)
	return gaussian_filter(unitlessCube, sigma = beamSigma, mode = 'constant', cval = 0) * unit

def getMassFromFlux(flux, beam, pixelSize, channelSize, distance):
	beamArea = (beam.sr/(pixelSize**2)).decompose()
	return 2.356e5 * (distance / units.Mpc).decompose()**2 * (flux/units.Jy).decompose() / beamArea * (channelSize/(units.km / units.s)).decompose() * units.Msun

def getJyFromMass(cube, beam, pixelSize, channelWidth, distance):
	converter = getMassFromFlux(1 * units.Jy, beam, pixelSize, channelWidth, distance)

	return (cube / converter).decompose() * units.Jy / units.beam

