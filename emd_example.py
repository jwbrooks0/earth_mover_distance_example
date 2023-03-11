
import numpy as _np
import xarray as _xr
import matplotlib.pyplot as _plt
import ot as _ot
import imageio as _imageio
from glob import glob as _glob
import os as _os


def earth_mover_distance_1D(da1, da2, plot=False):
	"""
	1D earth mover distance. Uses the Python Optimal Transport library.

	Parameters
	----------
	da1 : xarray.core.dataarray.DataArray
		First distribution. 1D DataArray with appropriate 1D coordinate
	da2 : xarray.core.dataarray.DataArray
		Second distribution. 1D DataArray with appropriate 1D coordinate
	plot : bool, optional
		Optional plot of results. The default is False.

	Raises
	------
	Exception
		DESCRIPTION.

	Returns
	-------
	emd_result : float
		EMD result
		
	"""
	
	## check inputs
	if _np.any(da1.values < 0) or _np.any(da2.values < 0):
		raise Exception("Values must be positive")
		
	## normalize each signal such that integral of each = 1.0
	da1 = da1 / da1.sum()
	da2 = da2 / da2.sum()
	
	## isolate coordinates
	coords1 = da1[list(da1.coords.keys())[0]].to_numpy()
	coords1 = coords1.reshape((len(coords1), 1))
	coords2 = da1[list(da1.coords.keys())[0]].to_numpy()
	coords2 = coords2.reshape((len(coords2), 1))
	
	## calculate euclidean distance between each set of coordinates
	M = _ot.dist(x1=coords1, x2=coords2, metric='euclidean') # euclidean distance from each pair of points to every other pair of points
	norm = M.max()
	M /= norm
	
	## perform EMD
	G0 = _ot.emd(a=da1.values, b=da2.values, M=M)
	emd_result = _np.sum(_np.sum(_np.multiply(G0, M))) * norm
	
	if plot is True:
		fig, ax = _plt.subplots()
		da1.plot(ax=ax, label="da1")
		da2.plot(ax=ax, label="da2")
		ax.legend()
		ax.set_title("EMD = %.3e" % emd_result)
	
	return emd_result


# %% main
if __name__ == "__main__":

	def noise(M, amp):
		""" Random noise """
		return (_np.random.rand(M) - 0.5) * amp
	
	def gaussian(x, sigma, x0=0):
	    """ Return the normalized Gaussian with standard deviation sigma. """
	    # c = _np.sqrt(2 * _np.pi)
	    return _np.exp(-0.5 * ((x - x0) / sigma)**2) #  / (_np.sqrt(2 * _np.pi) * _np.abs(sigma))
	
	_plt.ioff()
	
	## parameters
	sigma = 0.05
	noise_amp = 0.0001
	
	## create coordinates
	x = _np.linspace(-1.25, 1.25, 1000)
	x = _xr.DataArray(x, coords={'x': x})
	
	def make_plot(da1, da2, emd_result, filename):
			
		fig, ax = _plt.subplots(2, sharex=True)
		ax[0].axvline(0, ls="--", color="grey", lw=0.5)
		ax[1].axvline(0, ls="--", color="grey", lw=0.5)
		da1.plot(ax=ax[0], label="Distribution 1")
		da2.plot(ax=ax[0], label="Distribution 2")
		ax[0].legend()
		ax[0].set_ylabel("Gaussian distributions")
		if len(emd_result) > 1:
			emd_result.plot(ax=ax[1])
		ax[0].set_ylim([0, 1.5])
		ax[1].set_ylim([0, 1.1])
		ax[0].set_xlim([-1.25, 1.25])
		ax[0].set_title("")
		ax[1].set_xlabel("x")
		ax[1].set_ylabel("EMD result")
		fig.set_tight_layout(True)
		fig.savefig(filename, dpi=100)
		_plt.close(fig)
		
	## move the second gaussian from -1 to +1 and calculate EMD for each location
	x0_array = _np.linspace(-1, 1, 75)
	x0_array = _xr.DataArray(x0_array, coords={"x0": x0_array})
	emd_array = x0_array * 0.0
	for i, x0 in enumerate(x0_array):
		da1 = gaussian(x, sigma=sigma, x0=0) + _np.abs(noise(len(x), noise_amp))
		da2 = gaussian(x, sigma=sigma, x0=x0) + _np.abs(noise(len(x), noise_amp))
		emd_array[i] = earth_mover_distance_1D(da1, da2)
		make_plot(da1, da2, emd_result=emd_array[:(i+1)], filename="%.4d.png" % i)
			
	## create animation file
	files = _glob("*.png")
	ims = [_imageio.imread(f) for f in files]
	_imageio.mimwrite("animation.gif", ims)
	
	## delete all png files
	for file in files:
		_os.remove(file)
		
		
		