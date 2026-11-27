import numpy as np
from scipy.ndimage import gaussian_filter
from astropy import units

from .utils import _astropyUnitWrap, _convertBeamToBeamSigma

def convolve(cube, beam, pixelSize, spectralSigma = 0, pad = 2):
	if beam.major != beam.minor:
		warn("Only circular beams are supported. Will use major axis.")
	beamSigma = _convertBeamToBeamSigma(beam, spectralSigma, pixelSize)
	unitlessCube, unit = _astropyUnitWrap(cube)
	if pad != 0:
		padWidth = _getPadArray(pad, beamSigma)
		unitlessCube = np.pad(unitlessCube, padWidth, 'constant', constant_values = 0)
	convolvedCube = gaussian_filter(unitlessCube, sigma = beamSigma, mode = 'constant', cval = 0) * unit
	return convolvedCube

def _getPadArray(pad, beamSigma):
	if (pad * units.dimensionless_unscaled).unit.is_equivalent(units.pixel):
		return np.full( (len(beamSigma), 2), int((pad / units.pixel).decompose().value))
	return [(int(sigma * pad + 0.5),) * 2 for sigma in beamSigma]