import numpy as np
from scipy.ndimage import gaussian_filter

from .utils import _convertBeamToBeamSigma

def getNoiseCube(shape, noiseRMS, beam, pixelSize, spectralSigma = 0):
	beamSigma = _convertBeamToBeamSigma(beam, spectralSigma, pixelSize)
	noiseRMSPreBeam = noiseRMS * 2 * np.sqrt(beamSigma[1] * beamSigma[2]) * np.sqrt(np.pi)
	noiseCube = np.random.normal(loc = 0, scale = noiseRMSPreBeam.value, size = shape)
	noiseCube = gaussian_filter(noiseCube, beamSigma, mode = "wrap")
	noiseCube = noiseCube * noiseRMSPreBeam.unit
	return noiseCube