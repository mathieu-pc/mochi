"""
This is to handle post processing of mochi cubes
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from astropy import units

from .utils import _astropyUnitWrap, _convertBeamToBeamSigma


def getMassFromFlux(flux, beam, pixelSize, channelWidth, distance):
	beamArea = (beam.sr/(pixelSize**2)).decompose()
	mass = 2.356e5 * (distance / units.Mpc).decompose()**2 * (flux/units.Jy).decompose() / beamArea * (channelWidth/(units.km / units.s)).decompose() * units.Msun
	return mass

def getJyFromMass(cube, beam, pixelSize, channelWidth, distance):
	converter = getMassFromFlux(1 * units.Jy, beam, pixelSize, channelWidth, distance)
	flux = (cube / converter).decompose() * units.Jy / units.beam
	return flux

