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

def _refineGridBisect(size, x, y, z, mask, incell, newCells, newCellsOver, newCellsMasks):
	"""
	Bisect operation for refine grid algorithms
	"""
	newSize = size / 2.0
	newCells.extend([
		(x + dx * newSize, y + dy * newSize, z + dz * newSize, newSize) 
		for dx in range(2) for dy in range(2) for dz in range(2)
	])
	newCellsOver.extend([False] * 8)
	newCellsMasks.extend([mask[incell]] * 8)

def _passCompleteCell(cellsLists, contentList):
	for i in range(len(cellsLists)):
		cellsLists[i].append(contentList[i])

def refineGrid(incellFunction, bisectCondition, cells, positions, radii, threshold, cellsOver = None, cellsMasks = None, minSize = 0.25):
	"""
	Starting from a coarse grid, refine until no cell satisfy bisectCondition.
	
	Parameters
	----------
	incellFunction: function
		delimits the particles to consider for the cell
	bisectCondition: function
		if returns true, the cell is bisected
	cells: list
		list of [x,y,z,h] where x,y,z is the 3D position of the low corner and h is the size of the cell
	positions: array N x 3
		array of particle positions 
	radii: array N (unused)
		array of particle radii
	threshold: integer
		if any cell has occupancy number > threshold, the cell is bisected.
	cellsOver: None or list
		list of cells that no longer need to be checked
	cellsMasks: None or list
		list of particle indices of particles intersecting with cells

	Returns
	-------
	newCells:
		array of cells [x,y,z,h] where x,y,z is the 3D position of the low corner and h is the size of the cell
	"""
	cellsNumber = len(cells)
	if cellsOver is None:
		cellsOver = [False] * cellsNumber
	if cellsMasks is None:
		cellsMasks = [np.arange(len(radii))] * cellsNumber
	newCells = []
	newCellsOver = []
	newCellsMasks = []
	for n in range(cellsNumber):
		x, y, z, size = cells[n]
		if cellsOver[n]:
			_passCompleteCell([newCells, newCellsOver, newCellsMasks], [cells[n], True, True])
			continue
		incell = incellFunction(cellsMasks[n], positions, radii, [x,y,z], size)
		if bisectCondition(size, incell, minSize, threshold, radii):
			_refineGridBisect(size, x, y, z, cellsMasks[n], incell, newCells, newCellsOver, newCellsMasks)
		else:
			_passCompleteCell([newCells, newCellsOver, newCellsMasks], [cells[n], True, True])
	if len(newCells) == len(cells):
		return np.array(newCells)
	return refineGridToOccupancy(newCells, positions, radii, threshold, newCellsOver, newCellsMasks)

def occupancyIncell(mask, particlesPos, particlesRadii, cellPos, cellSize):
	return np.sum( np.abs(particlesPos[mask] - cellPos - cellSize/2), axis = 1) < cellSize * 2

def isNotSingleOccupancy(cellSize, incell, minSize, threshold, particlesRadii):
	count = np.sum(incell)
	return (count > threshold) & (cellSize > minSize)

refineGridToOccupancy = partial(refineGrid, occupancyIncell, isNotSingleOccupancy)

RF = np.sqrt(3)/2 #factor to convert cell size into effective radius contribution. Taken as max possible

def intersectIncell(mask, particlesPos, particlesRadii, cellPos, cellSize):
	return np.linalg.norm(particlesPos[mask] - cellPos - cellSize/2, axis = 1) < particlesRadii[mask] + cellSize * RF

def isNotParticleScale(cellSize, incell, minSize, threshold, particlesRadii):
	minRadius = np.min(particlesRadii[incell]) if np.any(incell) else np.inf
	return (minRadius * threshold < cellSize) & (cellSize > minSize)

refineGridToParticleScale = partial(refineGrid, intersectIncell, isNotParticleScale)

def getCellCentres(cells):
	"""Return a Nx3 numpy array of the cell centres."""
	return cells[:,:-1] + cells[:,-1][:,np.newaxis]/2

def getCellVolumes(cells):
	"""Return a N numpy array of the cell volumes."""
	return cells[:,-1]**3

def createRegularArray(cells, xyzRange, dtype = np.uintc):
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
	return grid, dx**3

def makeAdaptiveCube(particles, xRange, interpolant, kernel, channelSize, radiativeTransferModel,
		*,
		initialGridSize = 2,
		threshold = 0.5,
		minimumElement = 1 * units.kpc,
		refineAlgorithm = refineGridToParticleScale,
		**kwargs
	):
	"""
	Make a cube using adaptive resolution.
	Parameters
	----------
	particles : 
		Simulation particles dict
	xRange : 
		Iterable size 2 with start and end range in simulation space
		The (xRange[0],xRange[1])**3 volume will be interpolated
		Assumed to be in particles xyz_g units
	interpolant :
		interpolation method
	kernel :
		smoothing kernel
	channelSize:
		cube spectral channel size
	initialGridSize :
		starting number of cells along length for the grid before refinement
	threshold :
		Cell size threshold of minimum particle smoothing length after which cells are split.
		Lower -> higher resolution.

	Returns
	-------
	cube
	"""
	xyzRange = [(xRange[0].value, xRange[1].value)]*3
	initialCells = [
		(x, y, z, (xRange[1].value-xRange[0].value)/initialGridSize)
		for x in np.linspace(*xyzRange[0], initialGridSize, endpoint = False)
		for y in np.linspace(*xyzRange[1], initialGridSize, endpoint = False)
		for z in np.linspace(*xyzRange[2], initialGridSize, endpoint = False)
	]
	positions = (particles["xyz_g"] / xRange[0].unit).decompose()
	minRadius = (minimumElement / xRange[0].unit).decompose()
	if particles["hsm_g"] is None:
		radii = np.ones(len(positions)) * minRadius
	else:
		radii = (particles["hsm_g"] / xRange[0].unit).decompose()
		radii[radii < minRadius] = minRadius
	finalCells = refineAlgorithm(initialCells, positions, radii, threshold)
	cellCentres = getCellCentres(finalCells) * particles["xyz_g"].unit
	cellVolumes = getCellVolumes(finalCells) * particles["xyz_g"].unit ** 3
	fieldV, fieldMHI, fieldT = interpolant(
		particles["xyz_g"],
		particles["vxyz_g"],
		particles["hsm_g"],
		particles["mHI_g"],
		particles["T_g"],
		particles["m"],
		kernel,
		cellCentres,
		cellVolumes,
		**kwargs
	)
	cubeFieldIndices, finalCellVolume = createRegularArray(finalCells, xyzRange)
	finalCellVolume *= cellVolumes.unit
	cubeShape = cubeFieldIndices.shape
	cubeFieldIndices = cubeFieldIndices.flatten()
	return radiativeTransferModel(
		fieldMHI,
		fieldV,
		fieldT,
		channelSize,
		finalCellVolume,
		cubeShape,
		cubeFieldIndices = cubeFieldIndices,
		**kwargs
	)

