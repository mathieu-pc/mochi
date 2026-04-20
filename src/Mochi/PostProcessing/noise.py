import numpy as np
from scipy.ndimage import gaussian_filter

from .utils import _convertBeamToBeamSigma

def getNoiseCube(shape, noiseRMS, beam, pixelSize, spectralSigma = 0):
	beamSigma = _convertBeamToBeamSigma(beam, spectralSigma, pixelSize)
	noiseRMSPreBeam = noiseRMS * 2 * np.sqrt(beamSigma[1] * beamSigma[2]) * np.sqrt(np.pi)
	cube = np.random.normal(loc = 0, scale = noiseRMSPreBeam.value, size = shape)
	cube = gaussian_filter(cube, beamSigma, mode = "wrap")
	return cube * noiseRMSPreBeam.unit