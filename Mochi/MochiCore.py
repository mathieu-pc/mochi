"""
Transform particle data into mock cubes
"""


import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from astropy import units
from Mochi.ScanlineHI import makeCube as makeFixedCube
from Mochi.AdaptiveScanline import makeAdaptiveCube
import warnings

def makeCube(distance, particles, kernel, pixelNumber, pixelSize, channelWidth, beamSigma, interpolant, *, adaptiveMode = True, resizeMode = True, convolveMode = True, pad = 0, **kwargs):
	"""
	make a mock HI cube
	Parameters
	----------
	distance:
		distance to set mock galaxy at
	particles: dictionary
		particles dictionary of arrays
			xyz_g:	positions
			hsm_g:	smoothing lengths as kernel radius
			mHI_g:	particle HI mass
			m:	particle mass
			T_g:	particle thermal velocity dispersions
			vxyz_g:	particle velocity
	kernel:
		simulation smoothing kernel
	pixelNumber: int
		number of pixels
	pixelSize:
		angular pixel size
	channelWidth:
		channel size in velocity units
	beamSigma:
		angular beam STDEV. Gaussian beam is assumed.
	interpolant:
		interpolation method (example: MFM or SPH)
	adaptiveMode: bool
		Adaptive resolution when interpolating fields is used if True.
		Default: True.
	resizeMode: bool
		Resize the cube to match input pixel size if True.
		Default: True.
	convolveMode: bool
		Convolve using beam sigma if True.
		Default: True.
	pad: float
		Pad by number of beam sigma before convolution.
		default: 2.
	Returns
	-------
	cube:
		Mock HI cube with observation properties if resizeMode and convolveMode are True.
	"""
	if not adaptiveMode:
		n, deltaX = getScanlineParamsFromObservationParams(np.min(particles["hsm_g"])/2, pixelNumber, pixelSize, distance)
		cube = makeFixedCube( (n,) * 3, deltaX, particles, kernel, channelWidth, interpolant)
	else:
		cubeRange = (distance * pixelNumber * pixelSize.to(units.rad) / units.rad / 2).to(particles["hsm_g"].unit)
		cubeRange = (-cubeRange, cubeRange)
		cube = makeAdaptiveCube(particles, cubeRange, interpolant, kernel, channelWidth, **kwargs)
	if resizeMode:
		cube = resize(cube, [pixelNumber, pixelNumber])
		if convolveMode:
			cube = convolve(cube, beamSigma/pixelSize)
	else:
		if convolveMode:
			warnings.warn("Error: can't convolve without resize")
	return cube

def _astropyUnitWrap(cube):
	try:
		return cube.value.astype(float), cube.unit
	except:
		return cube.astype(float), 1

def applyObservation(cube, pixelSize, telescopePixelSize, beamSize = 0):
	resizeFactor = np.round( np.array(cube.shape[1:]) * pixelSize / telescopePixelSize )
	cube = resize(cube, [cube.shape[0], cube.shape[1] * resizeFactor[0], cube.shape[2] * resizeFactor[1]])
	return cube

def resize(cube, targetShape):
	"""
	resize a data cube to the target shape
	"""
	if( np.all( np.array(cube.shape[1:]) == np.array(targetShape))):
		print(cube.shape, targetShape)
		return cube
	targetShape = tuple(targetShape)
	unitlessCube, unit = _astropyUnitWrap(cube)
	result = np.zeros( (cube.shape[0],)+targetShape)
	for i in range(cube.shape[0]):
		result[i] = cv2.resize(unitlessCube[i].astype(float), targetShape[::-1], interpolation = cv2.INTER_AREA)
	return result * np.prod(cube.shape[1:]) / np.prod(targetShape) * unit

def convolve(cube, beamSigma, spectralSigma = 0, pad = 2):
	try:
		iterator = iter(beamSigma)
	except TypeError:
		sigma = (spectralSigma, beamSigma, beamSigma)
	else:
		sigma = (spectralSigma,) + tuple(beamSigma)
	unitlessCube, unit = _astropyUnitWrap(cube)
	if pad != 0:
		padWidth = [(int(x * pad + 0.5),) * 2 for x in sigma]
		unitlessCube = np.pad(unitlessCube, padWidth, 'constant', constant_values = 0)
	return gaussian_filter(unitlessCube, sigma = sigma, mode = 'constant', cval = 0) * unit

def getScanlineParamsFromObservationParams(scanlineResolution, pixelNumber, pixelSize, distance):
	"""
	Given a desired Scanline Resolution, returns the best number of scanline elements and best scanline resolution for observation parameters.
	This serves to ensure that the cube's length remains an integer number of both the scanline elements and pixel sizes.
	"""
	physicalPixelSize = (pixelSize.to(units.rad) * distance / units.rad).to(units.kpc)
	if(physicalPixelSize < scanlineResolution):
		return pixelNumber, physicalPixelSize
	cubeLength = pixelNumber * physicalPixelSize
	n = int(np.ceil(cubeLength / scanlineResolution))
	deltaX = cubeLength / n
	return n, deltaX
