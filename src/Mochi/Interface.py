import warnings
import cv2
import numpy as np
from astropy import units

from . import PostProcessing
from .ScanlineHI import makeCube as makeFixedCube
from .AdaptiveScanline import makeAdaptiveCube
from .PostProcessing import _astropyUnitWrap
from .RadiativeTransfer import adaptiveOpticallyThin


def makeCubeFromSource(martiniSource, kernel, pixelNumber, pixelSize, channelWidth, interpolant, **kwargs):
	"""
	Make a MOCHI cube from a MARTINI source object.
	"""
	particles = {
		"mHI_g": martiniSource.mHI_g,
		"m": martiniSource.mHI_g,
		"hsm_g": martiniSource.hsm_g,
		"xyz_g": martiniSource.coordinates_g.get_xyz().T,
		"vxyz_g": martiniSource.coordinates_g.differentials["s"].get_d_xyz().T,
		"T_g": (martiniSource.T_g * constants.k_B / constants.m_p).decompose()
	}
	cube = Mochi.makeCube(
		martiniSource.distance,
		particles,
		kernel,
		pixelNumber,
		pixelSize,
		channelWidth,
		interpolant,
		**kwargs
	)
	return cube


def makeCube(distance, particles, kernel, pixelNumber, pixelSize, channelWidth, interpolant, radiativeTransferModel = adaptiveOpticallyThin,
		*,
 		beam = None,
		adaptiveMode = True,
		resizeMode = True,
		pad = 0,
		**kwargs
	):
	"""
	make a mock HI cube
	"""
	if not adaptiveMode:
		n, deltaX = _getScanlineParamsFromObservationParams(np.min(particles["hsm_g"])/2, pixelNumber, pixelSize, distance)
		cube = makeFixedCube( (n,) * 3, deltaX, particles, kernel, channelWidth, interpolant, radiativeTransferModel)
	else:
		cubeRange = (distance * pixelNumber * pixelSize.to(units.rad) / units.rad / 2).to(particles["xyz_g"].unit)
		cubeRange = (-cubeRange, cubeRange)
		cube = makeAdaptiveCube(particles, cubeRange, interpolant, kernel, channelWidth, radiativeTransferModel, **kwargs)
	if resizeMode:
		cube = resize(cube, [pixelNumber, pixelNumber])
		if beam is not None:
			cube = PostProcessing.convolve(cube, beam, pixelSize)
	else:
		if beam is not None:
			warnings.warn("Can't convolve when resizeMode is not True")
	return cube


def resize(cube, targetShape):
	"""
	Resize a data cube to the target shape.
	Since MOCHI uses adaptive resolution and specific pixel resolutions are needed for mocks,
	this step if often needed.
	"""
	if( np.all( np.array(cube.shape[1:]) == np.array(targetShape))):
		resizedCube = cube
		return resizedCube
	targetShape = tuple(targetShape)
	unitlessCube, unit = _astropyUnitWrap(cube)
	result = np.zeros( (cube.shape[0],)+targetShape)
	for i in range(cube.shape[0]):
		result[i] = cv2.resize(unitlessCube[i].astype(float), targetShape[::-1], interpolation = cv2.INTER_AREA)
	# the normalization doesn't seem necessary, cv2.INTER_AREA preserves flux.
	resizedCube = result * np.prod(cube.shape[1:]) / np.prod(targetShape) * unit
	return resizedCube


def _getScanlineParamsFromObservationParams(scanlineResolution, pixelNumber, pixelSize, distance):
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