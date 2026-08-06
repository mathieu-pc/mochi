"""
Interpolating the fields on a high res grid is expensive.
It is advantageous to dynamically set the resolution.
Currently, this doubles the distance computations, something I need to fix.
But it generally saves significant RAM for MFM interpolation.
And also saves computation time for both MFM and SPH interpolation.
"""
import numpy as np
from astropy import units
from functools import partial
from . import RadiativeTransfer


def _refineGridBisect(cell, particleIndices, incell, newCells, newCellsOver, newCellsParticleIndices):
	"""
	Bisect operation for refine grid algorithms
	"""
	newSize = cell[-1] / 2.0
	newCells.extend([
		(cell[0] + dx * newSize, cell[1] + dy * newSize, cell[2] + dz * newSize, newSize) 
		for dx in range(2) for dy in range(2) for dz in range(2)
	])
	newCellsOver.extend([False] * 8)
	newCellsParticleIndices.extend([particleIndices[incell]] * 8)


def _passCompleteCell(cellsLists, contentList):
	for i in range(len(cellsLists)):
		cellsLists[i].append(contentList[i])


def refineGrid(particleSelection, bisectCondition, cells, positions, particlesRadii, threshold, stopIter):
	"""
	Starting from a coarse grid, refine until no cell satisfy bisectCondition.
	"""
	cellsNumber = len(cells)
	cellsOver = np.zeros(cellsNumber, dtype = bool)
	cellsParticleIndices = [np.arange(len(particlesRadii))] * cellsNumber
	newCells = []
	newCellsOver = []
	newCellsParticleIndices = []

	iter = 0
	while iter < stopIter:
		for n in range(cellsNumber):
			if cellsOver[n]:
				_passCompleteCell([newCells, newCellsOver, newCellsParticleIndices], [cells[n], True, True])
				continue
			incell = particleSelection(cellsParticleIndices[n], positions, particlesRadii, cells[n], threshold)
			if bisectCondition(incell):
				_refineGridBisect(cells[n], cellsParticleIndices[n], incell, newCells, newCellsOver, newCellsParticleIndices)
			else:
				_passCompleteCell([newCells, newCellsOver, newCellsParticleIndices], [cells[n], True, True])
		cells = newCells

		if len(cells) == cellsNumber or iter == stopIter:
			break
		cellsNumber = len(cells)
		cellsOver = newCellsOver
		cellsParticleIndices = newCellsParticleIndices
		newCells = []
		newCellsOver = []
		newCellsParticleIndices = []
		iter += 1
	refinedCells = np.array(cells)
	return refinedCells


def occupancyIncell(mask, particlesPos, particlesRadii, cell, threshold):
	occupyingParticlesMask = np.sum( np.abs(particlesPos[mask] - cell[:3] - cell[3]/2), axis = 1) < cell[3] * 2
	return occupyingParticlesMask


def isNotSingleOccupancy(incellParticleMask, threshold = 1):
	count = np.sum(incellParticleMask)
	isCountOverThreshold = (count > threshold)
	return isCountOverThreshold


def composeRefinementStrategy(particleSelection, bisectCondition):
	def refinementAlgorithm(cells, particlesPos, particlesRadii, threshold, stopIter = 8):
		refinedCells = refineGrid(particleSelection, bisectCondition, cells, particlesPos, particlesRadii, threshold, stopIter)
		return refinedCells
	return refinementAlgorithm


RF = np.sqrt(3)/2				#factor to convert cell size into effective radius contribution. Taken as max possible


def intersectIncell(mask, particlesPos, particlesRadii, cell, threshold):
	smallParticle = particlesRadii[mask] * threshold < cell[3] 	#No need to consider particles larger than cell
	intersectingSmallParticleMask = (np.linalg.norm(particlesPos[mask] - cell[:3] - cell[3]/2, axis = 1) < particlesRadii[mask] + cell[3] * RF) & smallParticle
	return intersectingSmallParticleMask


refineGridToParticleScale = composeRefinementStrategy(intersectIncell, np.any)
refineGridToOccupancy = composeRefinementStrategy(occupancyIncell, isNotSingleOccupancy)


def _getCellCentres(cells):
	"""Return a Nx3 numpy array of the cell centres."""
	return cells[:,:-1] + cells[:,-1][:,np.newaxis]/2

def _getCellVolumes(cells):
	"""Return a N numpy array of the cell volumes."""
	return cells[:,-1]**3

def _createRegularArray(cells, xyzRange, dtype = np.uintc):
	"""Converts an adaptive set of cells into a regular array"""
	xyz0 = np.min(cells, axis = 0)
	dx = xyz0[-1]
	xyz0[-1] = 0
	grid_shape = [ int((myRange[1]-myRange[0])//dx) for myRange in xyzRange]
	N = len(cells)
	cellRange = np.arange(N, dtype = dtype)
	grid = np.empty(grid_shape, dtype = dtype)#np.empty(grid_shape, dtype=int) #grid = np.full(grid_shape, np.prod(grid_shape)+10, dtype = int) slower but good for testing
	cellsBegin = np.round((cells[:,:-1] - xyz0[:-1])/dx).astype(int)
	cellsFinish = np.round((cells[:,:-1] - xyz0[:-1] + cells[:,-1][:,np.newaxis])/dx).astype(int)
	for i in cellRange:
		x_start, y_start, z_start = cellsBegin[i]
		x_end, y_end, z_end = cellsFinish[i]
		grid[x_start:x_end, y_start:y_end, z_start:z_end] = i
	dvolume = dx ** 3
	return grid, dvolume

def makeAdaptiveCube(particles, xRange, interpolant, kernel, channelWidth, radiativeTransferModel,
	*,
	initialGridSize = 2,
	threshold = 0.5,
	refinementAlgorithm = refineGridToParticleScale,
	**kwargs
	):
	"""
	Make a cube using adaptive resolution.
	"""
	xyzRange = [(xRange[0].value, xRange[1].value)]*3
	initialCells = [
		(x, y, z, (xRange[1].value-xRange[0].value)/initialGridSize)
		for x in np.linspace(*xyzRange[0], initialGridSize, endpoint = False)
		for y in np.linspace(*xyzRange[1], initialGridSize, endpoint = False)
		for z in np.linspace(*xyzRange[2], initialGridSize, endpoint = False)
	]
	positions = (particles["xyz_g"] / xRange[0].unit).decompose().value
	if particles["hsm_g"] is None:
		radii = np.ones(len(positions))
	else:
		radii = (particles["hsm_g"] / xRange[0].unit).decompose().value
	finalCells = refinementAlgorithm(initialCells, positions, radii, threshold)
	cellCentres = _getCellCentres(finalCells) * particles["xyz_g"].unit
	cellsVolume = _getCellVolumes(finalCells) * particles["xyz_g"].unit ** 3
	fieldV, fieldMHI, fieldT = interpolant(
		particles["xyz_g"],
		particles["vxyz_g"],
		particles["hsm_g"],
		particles["mHI_g"],
		particles["T_g"],
		particles["m"],
		kernel,
		cellCentres,
		cellsVolume,
		**kwargs
	)
	cubeFieldIndices, finalCellVolume = _createRegularArray(finalCells, xyzRange)
	finalCellVolume *= cellsVolume.unit
	cubeShape = cubeFieldIndices.shape
	cubeFieldIndices = cubeFieldIndices.flatten()#a
	return radiativeTransferModel(
		fieldMHI,
		fieldV,
		fieldT,
		channelWidth,
		finalCellVolume,
		cubeShape,
		cells = finalCells,
		cellUnit = particles["xyz_g"].unit,
		**kwargs
	)
