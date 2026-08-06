import inspect
import ast
import warnings
import textwrap
from sphinx.util import logging
import re

logger = logging.getLogger(__name__)

indent = "    "
astropyQuantity = ":class:`astropy.units.Quantity`\n"
astropyUnit = ":class:`astropy.units.Unit`"
astropyCard = ":class:`astropy.io.fits.Card`"
numpyArray = ":class:`numpy.ndarray`\n"
martiniSource = ":class:`martini.sources.sph_source.SPHSource`"
radioBeam = ":class:`radio_beam.Beam`"


def get_live_object(name: str):
	"""Dynamically imports and retrieves a Python object from its fully-qualified string name."""
	parts = name.split(".")
	for i in range(len(parts) - 1, 0, -1):
		mod_name = ".".join(parts[:i])
		attr_path = parts[i:]
		try:
			obj = importlib.import_module(mod_name)
			for attr in attr_path:
				obj = getattr(obj, attr)
			return obj
		except (ImportError, AttributeError):
			continue
	return None


class TrackingDictionary(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._unused = set(self)

    def __getitem__(self, key):
        self._unused.discard(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self._unused.discard(key)
        return super().get(key, default)

    @property
    def unused(self):
        return self._unused


Argument_Dictionary = TrackingDictionary({
	"shape": "tuple[int]\n",
	"targetShape": "tuple[int]\n",
	"deltaX" : "?\n" +
		indent + "???",
	"stopIter": "int\n" + indent +"Maximum number of iterations.",
	"positions": "array\n" + indent + "Nx3 array of positions.",
	"dtype": "type\n" + 
		indent + "data type",
	#"mHI": "array\n" + 
	#	indent + "Emission mass",
	"cells": numpyArray + 
		indent +"Nx4 array of cells, where N is the number of cells.\n" + 
		indent + "Formated (x,y,z,l), with l being the cuboid cell length.",
	"cards": "list\n" +
		indent + "List of instances of " + astropyCard + ".",
	"cube": astropyQuantity +
		indent + "Datacube. (channel, pixel, pixel) format is expected.",
	"resizedCube": astropyQuantity +
		indent + "Datacube. (channel, pixel, pixel) format is expected.\n" +
		indent + "Cube which has been decimated or upscaled to targeteShape.",
	"noiseCube" : astropyQuantity +
		indent + "Datacube. (channel, pixel, pixel) format is expected.\n" +
		indent + "Cube of synthetic noise.",
	"convolvedCube": astropyQuantity +
		indent + "Convolved datacube.",
	"pixelSize": astropyQuantity +
		indent + "Angular pixel size.",
	"channelWidth": astropyQuantity +
		indent + "Velocity channel size.",
	"beam": radioBeam + "\n" +
		indent + "Beam object. Instance of " + radioBeam + ".",
	"particleSelection": "callable\n" +
		indent + "Selects which particle should be considered by the cell.",
	"bisectCondition": "callable\n" +
		indent + "Condition which determines whether or not to bisect a cell.\n" +
		indent + "When True, the cell is bisected.",
	"refinementAlgorithm": "callable\n" +
		indent + "Refines a coarse grid for variable resolution allocation.",
	#"xyzRange": "tuple\n" +
	#	indent + "The range along the x,y,z axes.",
	"refinedCells": numpyArray +
		indent +"Nx4 array of cells, where N is the number of cells.\n" + 
		indent + "Formated (x,y,z,l), with l being the cuboid cell length." +
		indent + "Adaptive resolution cells with variable length.",
	#"grid": "array\n" +
	#	indent + "Grid array",
	"intersectingSmallParticleMask": "array\n" +
		indent + "Mask of intersecting small particles for a given cell.",
	"occupyingParticlesMask": numpyArray +
		indent + "Mask (Boolean) of particles occupying input cell.",
	"mask": numpyArray +
		indent + "Mask (Boolean).",
	"particlesPos": numpyArray +
		indent + "Nx3 unitless array of particle positions.",
	"X": "array\n" +
		indent + "Nx3 particle positions.",
	"V": "array\n" +
		indent + "N array of particle radial velocities.\n" +
		indent + "If multiple dimensions are given, V[:,0] is taken as the radial velocity.",
	"VX": "array\n" +
		indent + "N array of particle radial velocities\n",
	"H": "array\n" +
		indent + "N array of particle radii.",
	"MHI": "array\n" +
		indent + "N array of particle emission mass.",
	"T": "array\n" +
		indent + "N array of particle velocity dispersions.\n" +
		indent + "Expected to be in V units squared.",
	"M": "array\n" +
		indent + "N array of particle interpolation masses.",
	"fieldPos": "array\n" +
		indent + "Nx3 Array of field positions.",
	"dVolume": astropyQuantity +
		indent + "Array of cell volumes.\n" +
		indent + "Has units.",
	"cellsVolume": numpyArray +
		indent + "Unitless array of cell volumes.",
	"kernelCacheResolution": "int\n" +
		indent + "Number of resolution elements in kernel cache.\n" +
		indent + "To avoid costly kernel evaluations, the kernel is evaluated onto discrete points.\n" +
		indent + "Kernel calls are then evaluated by interpolating these points.\n" +
		indent + "This parameter sets the number of points used for kernel evaluation.",
	"kernelCache": "array\n" +
		indent + "Array of kernel values at regular intervals.",
	"particlesRadii": numpyArray +
		indent + "N unitless array of particle radii.",
	"cell": numpyArray +
		indent + "4 array of cell x,y,z lower corner position and length.",
	"threshold": "?\n" +
		indent + "Sensitivity.",
	"incellParticleMask": numpyArray +
		indent + "Mask of particles cell needs to consider for operations.",
	"isCountOverThreshold": "bool\n" +
		indent + "True if the number of non-zero values of the input mask is greater than threshold.",
	"particles": "dict\n" +
		indent + "Dict of particles using :class:`astropy.units.Quantity` arrays.\n" +
		indent + "[\"xyz_g\"] particle positions\n" +
		indent + "[\"hsm_g\"] particle radii",
	"xRange": "?\n" +
		indent + "Range",
	"interpolant": "callable\n" +
		indent + "Interpolant to interpolate the fields from the particles.",
	"subInterpolant": "callable\n" +
		indent + "Subinterpolant to be used by higher order interpolant.",
	"kernel": "callable\n" +
		indent + "Smoothing kernel used by interpolants.",
	"radiativeTransferModel": "callable\n" +
		indent + "Collapses fields into mock cube.",
	"initialGridSize": "int\n" +
		indent + "Number of elements in initial grid before refinement.",
	"kwargs": "kwargs\n" +
		indent + "Function kwargs.",
	"nPos": "int\n" +
		"Number of positions.",
	"nParticles": "int\n" +
		"Number of particles.",
	"fields": "tuple\n" +
		indent + "Tuple of interpolated fields.\n" +
		indent + "Contains\n\n" +
		indent + "[0] field radial velocities\n\n" +
		indent + "[1] field emission masses (fieldMHI)\n\n" +
		indent + "[2] field velocity dispersions\n\n",
	"fieldV": "array\n" +
		indent + "Array of field radial velocities.\n",
	"fieldMHI": "array\n" +
		indent + "Array of field emission masses.\n",
	"fieldT": "array\n" +
		indent + "Array of field velocity dispersions.\n",
	"fieldM": "array\n" +
		indent + "Array of interpolation mass.\n",
	"fieldSpectrum": "array\n" +
		indent + "Array of spectrum evaluated at field positions.\n",
	"particlesMomentum": "array\n" +
		indent + "Unitless momentum of particles along line of sight.",
	"dist": "array\n" +
		indent + "Distances.",
	"slices": "list\n" +
		indent + "List of array of indices.",
	"velocityUnit": astropyUnit + "\n" +
		indent + "Astropy velocity unit.",
	"massUnit": astropyUnit + "\n" +
		indent + "Astropy mass unit.",
	"volumeUnit": astropyUnit + "\n" +
		indent + "Astropy volume unit.",
	"maskOutOfBound": "array\n" +
		indent + "Array of bools.\n" +
		indent + "True for particles out of bounds for interpolation.",
	"volumeShape": "tuple\n" +
		indent + "Simulation space interpolated volume.",
	"cellUnit": "unit\n" +
		indent + "???",
	"indexType": "type\n" +
		indent + "Type used for indexing.\n" +
		indent + "This typically defaults to something small like np.uintc.\n" +
		indent + "Other types may be specified when needed for larger cubes.",
	"defaultRenderer": "callable\n" +
		indent + "Renderer called when cells is missing for adaptive renderers.",
	"distance": astropyQuantity +
		indent + "Distance to set particles at from observer.",
	"pixelNumber": "int\n" +
		indent + "Number of pixels for mock observation.",
	"adaptiveMode": "bool\n" +
		indent + "If True, dynamic resolution allocation will be used.",
	"resizeMode": "bool\n" +
		indent + "If True, mock cube will be resized to pixel size.",
	"pad" : "float\n" +
		indent + "Pad number of beam sigma lengths to pad on cube.",
	"martiniSource": martiniSource + "\n" +
		indent + "Instance of class derived from " + martiniSource +".\n" +
		indent + "Source object to use for mock observation.",
	"channelNumber": "int\n" +
		indent + "Number of channels for datacube.",
	"minChannelNumber": "int\n" +
		indent + "Minimum number of channels.",
	"maxChannelNumber": "int\n" +
		indent + "Maximum number of channels.",
	"spectralSigma": "float\n" +
		indent + "Spectral channel Gaussian PSF smear sigma.",
	"noiseRMS": astropyQuantity +
		indent + "Noise RMS.",
	"flux": astropyQuantity +
		indent + "Flux.",
	"mass": astropyQuantity +
		indent + "Mass.",
})

def check_unused(app, exception):
	if exception is not None:
		return
	print("Unused:")
	print(Argument_Dictionary.unused)

def _get_return_names(fun):
	return_names = []
	source = textwrap.dedent(inspect.getsource(fun))
	tree = ast.parse(source)
	subReturns = []
	mainDef = True
	for node in ast.walk(tree):
		if isinstance(node, ast.FunctionDef):
			if mainDef:
				mainDef = False
				continue
			for child in ast.walk(node):
				if isinstance(child, ast.Return):
					subReturns += [child]
	for node in ast.walk(tree):
		if isinstance(node, ast.Return):
			if node in subReturns:
				continue
			if type(node.value) == ast.Tuple:
				values = node.value.elts
			else:
				values = [node.value]
			for value in values:
				if type(value) == ast.Name:
					return_names.append(value.id)
					continue
				if type(value) == ast.Call:
					return_names.append(value.func.id)
					continue
				warnings.warn("Unnamed or unknown return value in " + fun.__name__, UserWarning)
				continue
	return_names = list(set(return_names))
	return return_names


def _get_var_description(var_name, variable_descriptions, fun):
	var_description = variable_descriptions.get(var_name)
	if var_description is None:
		var_description = "Missing\n" + indent + "Missing"
		variable_descriptions[var_name] = var_description
		error_message = "Missing variable description " + var_name + " in " + fun.__name__
		warnings.warn(error_message, UserWarning)
	return var_description


def assign_doc(fun, variable_descriptions = Argument_Dictionary, assign = True):
	"""Automatically assign the parameters and return documentation to a function's docstring."""
	if not callable(fun):
		#logger.info(type(fun))
		#logger.info("Exit")
		return ""
	#logger.info("Entry")
	signature = inspect.signature(fun)
	arg_names = list(signature.parameters.keys())
	if fun.__doc__ is None:
		doc = ""
	else:
		doc = fun.__doc__
	doc += "\n\nParameters\n----------"
	for arg_name in arg_names:
		arg_description = _get_var_description(arg_name, variable_descriptions, fun)
		doc += "\n" + arg_name + " : " + arg_description + "\n"
	return_names = _get_return_names(fun)
	doc += "\nReturns\n-------"
	for return_name in return_names:
		return_description = _get_var_description(return_name, variable_descriptions, fun)
		doc += "\n" + return_name + " : " + return_description + "\n"
	if assign:
		if fun.__doc__ is None:
			fun.__doc__ = ""
		doc = fun.__doc__ + doc
	return doc


def process_docstring(app, what, name, obj, options, lines):
	if what != "function":
		return
	if obj is None:
		obj = get_live_object(name)
	if obj is None:
		return
	if not obj.__module__.startswith("Mochi"):
		return
	docString = assign_doc(obj, assign = False)
	lines[:] = docString.splitlines()

def setup(app):
	app.connect("autodoc-process-docstring", process_docstring)
	app.connect("build-finished", check_unused)

	return {
		"version": "1.0",
		"parallel_read_safe": True,
	}
