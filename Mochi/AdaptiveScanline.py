"""
Interpolating the fields on a high res grid is expensive.
It is advantageous to dynamically set the resolution.
Currently, this doubles the distance computations, something I need to fix.
But it generally saves significant RAM for MFM interpolation.
And also saves computation time for both MFM and SPH interpolation.
"""
import numpy as np
from Mochi.ScanlineHI import makeCubeFromFields
from astropy import units

def _refineGridBisect(size, x, y, z, mask, incell, newCells, newCellsOver, newCellsMasks):
	"""
	bisect operation for refine grid algorithms
	"""
	newSize = size / 2.0
	newCells.extend([
		(x + dx * newSize, y + dy * newSize, z + dz * newSize, newSize) 
		for dx in range(2) for dy in range(2) for dz in range(2)
	])
	newCellsOver.extend([False] * 8)
	newCellsMasks.extend([mask[incell]] * 8)

def refineGridToOccupancy(cells, positions, radii, threshold, cellsOver = None, cellsMasks = None):
	"""
	Starting from a coarse grid, refine until all cells are within a factor of particle radii locally
	
	Parameters
	----------
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
			newCells.append((x, y, z, size))
			newCellsOver.append(True)
			newCellsMasks.append([True])
			continue
		# Find particles inside this cell
		mask = cellsMasks[n]
		incell = np.sum( np.abs(positions[mask] - [x,y,z] - size/2), axis = 1) < size
		count = np.count(incell)
		if count > threshold:
			_refineGridBisect(size, x, y, z, mask, incell, newCells, newCellsOver, newCellsMasks)
		else:
			newCells.append((x, y, z, size))
			newCellsOver.append(True)
			newCellsMasks.append([True])
	if len(newCells) == len(cells):
		return np.array(newCells)
	return refineGridToOccupancy(newCells, positions, radii, threshold, newCellsOver, newCellsMasks)


def refineGridToParticleScale(cells, positions, radii, threshold, cellsOver = None, cellsMasks = None):
	"""
	Starting from a coarse grid, refine until all cells are within a factor of particle radii locally
	
	Parameters
	----------
	cells: list
		list of [x,y,z,h] where x,y,z is the 3D position of the low corner and h is the size of the cell
	positions: array N x 3
		array of particle positions 
	radii: array N
		array of particle radii
	threshold: float
		scale factor between radii and cell size
		if any radius * threshold < cell size, the cell is bisected
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
	sizeRadiusFactor = np.sqrt(3)/2 #factor to convert cell size into effective radius contribution. Taken as max possible
	for n in range(cellsNumber):
		x, y, z, size = cells[n]
		if cellsOver[n]:
			newCells.append((x, y, z, size))
			newCellsOver.append(True)
			newCellsMasks.append([True])
			continue
		# Find particles inside this cell
		mask = cellsMasks[n]
		incell = np.linalg.norm(positions[mask] - [x,y,z] - size/2, axis = 1) < radii[mask] + size * sizeRadiusFactor
		minRadius = np.min(radii[mask][incell]) if np.any(incell) else np.inf		
		if minRadius * threshold < size:
			_refineGridBisect(size, x, y, z, mask, incell, newCells, newCellsOver, newCellsMasks)
		else:
			newCells.append((x, y, z, size))
			newCellsOver.append(True)
			newCellsMasks.append([True])
	if len(newCells) == len(cells):
		return np.array(newCells)
	return refineGridToParticleScale(newCells, positions, radii, threshold, newCellsOver, newCellsMasks)

def getCellCentres(cells):
	"""Return a Nx3 numpy array of the cell centres."""
	return cells[:,:-1] + cells[:,-1][:,np.newaxis]/2

def getCellVolumes(cells):
	"""Return a N numpy array of the cell volumes."""
	return cells[:,-1]**3

def createRegularArray(cells, xyzRange):
	"""Convers an adaptive set of cells into a regular array"""
	xyz0 = np.min(cells, axis = 0)
	dx = xyz0[-1]
	xyz0[-1] = 0
	grid_shape = [ int((myRange[1]-myRange[0])//dx) for myRange in xyzRange]
	grid = np.empty(grid_shape, dtype=int) #grid = np.full(grid_shape, np.prod(grid_shape)+10, dtype = int) slower but good for testing
	N = len(cells)
	cellsBegin = ((cells[:,:-1] - xyz0[:-1])/dx).astype(int)
	cellsFinish = np.ceil((cells[:,:-1] - xyz0[:-1] + cells[:,-1][:,np.newaxis])/dx).astype(int)
	for i in range(N):
		x_start, y_start, z_start = cellsBegin[i]
		x_end, y_end, z_end = cellsFinish[i]
		grid[x_start:x_end, y_start:y_end, z_start:z_end] = i
	return grid, dx**3

def makeAdaptiveCube(
		particles,
		xRange,
		interpolant,
		kernel,
		channelSize, *,
		initialGridSize = 2,
		threshold = 0.5,
		trigger = 1,
		minimumElement = 1 * units.kpc,
		refineAlgorithm = refineGridToParticleScale
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
	trigger :
		trigger after which particle kernel is re-computed

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
	radii = (particles["hsm_g"] / xRange[0].unit).decompose()
	minRadius = (minimumElement / xRange[0].unit).decompose()
	radii[radii < minRadius] = minRadius
	finalCells = refineGridToParticleScale(initialCells, positions, radii, threshold)
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
		trigger = trigger
	)
	cubeFieldIndices, cellVolume = createRegularArray(finalCells, xyzRange)
	cubeShape = cubeFieldIndices.shape
	cubeFieldIndices = cubeFieldIndices.flatten()
	return makeCubeFromFields(fieldMHI[cubeFieldIndices], fieldV[cubeFieldIndices], fieldT[cubeFieldIndices], channelSize, cellVolume, cubeShape)