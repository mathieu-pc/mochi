"""
This is to handle post processing of mochi cubes
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from astropy import units
from warnings import warn

from .utils import _astropyUnitWrap, _convertBeamToBeamSigma


def getMassFromFlux(flux, beam, pixelSize, channelSize, distance):
	beamArea = (beam.sr/(pixelSize**2)).decompose()
	return 2.356e5 * (distance / units.Mpc).decompose()**2 * (flux/units.Jy).decompose() / beamArea * (channelSize/(units.km / units.s)).decompose() * units.Msun

def getJyFromMass(cube, beam, pixelSize, channelWidth, distance):
	converter = getMassFromFlux(1 * units.Jy, beam, pixelSize, channelWidth, distance)
	return (cube / converter).decompose() * units.Jy / units.beam

