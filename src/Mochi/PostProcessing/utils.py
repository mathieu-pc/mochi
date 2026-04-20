import numpy as np

def _astropyUnitWrap(cube):
	try:
		return cube.value.astype(float), cube.unit
	except:
		return cube.astype(float), 1


SIGMA_TO_FWHM_FACTOR = 2 * np.sqrt(2 * np.log(2))


def _convertBeamToBeamSigma(beam, spectralSigma, pixelSize):
	beamFactor = 1 / pixelSize / SIGMA_TO_FWHM_FACTOR
	return (spectralSigma, beam.major * beamFactor, beam.minor * beamFactor)
