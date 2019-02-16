#from blimpy import Waterfall
import pycuda.compiler as nvcc
import pycuda.gpuarray as gpuarray
import pycuda.driver as cu
import pycuda.autoinit
from pycuda import cumath
import numpy as np
from pycuda.tools import make_default_context

"""
  Author: Yunfan G. Zhang
  04/2017
"""



# def test(ntimes=4, nfreqs=8, block_size=(8, 4, 1)):
# 	data = np.zeros((ntimes,nfreqs), dtype=np.float32)
# 	data[:,nfreqs/2] = 1
# 	kernel_source = open("./turbo.cu").read()
# 	kernel_code = kernel_source % {
# 		'BDX': block_size[0],
# 		'BDY': block_size[1]
# 	}
# 	main_module = nvcc.SourceModule(kernel_code)
# 	reduce_kernel = main_module.get_function("reduce")

# 	spectr_d = gpuarray.empty([ntimes, nfreqs], dtype=np.float32) # y dimension by x dimension
# 	output_d = gpuarray.zeros([nfreqs], dtype=np.float32)

# 	spectr_d = gpuarray.to_gpu(data)

# 	grid_size = (nfreqs/block_size[0],ntimes/block_size[1], 1) # X by Y by Z
# 	print grid_size, grid_size

# 	reduce_kernel(spectr_d, output_d, np.int32(nfreqs), block=block_size, grid=grid_size)

# 	out = output_d.get()

# 	import IPython; IPython.embed()

# 	return


def generate_delay_table(drift_rates, delta_t, ntimes):
	"""
	return table is number of freq offsets as a function of drift rate and time sample
	shape = (ntimes, ndrift)
	"""
	return (drift_rates[np.newaxis,:] * delta_t * np.arange(ntimes)[:,np.newaxis]).astype(np.int32)

class TurboSolver:

	def __init__(self, ntimes=16, nfreqs=128, ndelays=32, delta_t=20., connectivity=2, block_size=(16, 32, 1)):

		"""
		nfreqs: Data will be fed in a chunk at at time, this is number of channels in a chunk
		ndelays: number of delays to compute
		"""

		kernel_source = open("./turbo.cu").read()

		kernel_code = kernel_source % {
		 	'BDX': block_size[0],
			'BDY': block_size[1],
			'NTIMES': ntimes,
			'NDELAYS': ndelays,
			'CONN': connectivity
		}
		main_module = nvcc.SourceModule(kernel_code)

		self.spectr_d = gpuarray.empty([ntimes, nfreqs], dtype=np.float32) # y dimension by x dimension
		self.output_d = gpuarray.zeros([ndelays, nfreqs], dtype=np.float32)
		self.mask_d = gpuarray.zeros_like(self.output_d, dtype=bool)

		delay_table_h = generate_delay_table(np.linspace(-0.1, 0.1, ndelays), 
													delta_t=delta_t, ntimes=ntimes)
		use_const = False
		if use_const: #use constant memory
			self.sweep_kernel = main_module.get_function("sweep_const_mem")
			self.delay_table_d = main_module.get_global('delay_table_const')[0]
			cu.memcpy_htod(self.delay_table_d, delay_table_h)
		else:
			self.sweep_kernel = main_module.get_function("sweep")
			self.delay_table_d = gpuarray.to_gpu(delay_table_h)
		self.threshold_kernel = main_module.get_function("threshold_and_local_max")

		self.nfreqs = np.int32(nfreqs)
		self.ntimes = np.int32(ntimes)
		self.ndelays = np.int32(ndelays)
		self.delta_t = delta_t
		self.block_size = block_size
		self.grid_size = (nfreqs/block_size[0], ndelays/block_size[1], 1)
		print self.grid_size


	def bench(self, data, plot=True, get=False):

		""" Function to test drift rate search
			note this bench mark contains compilation times and multiple kernel calls. 
			larger input would be faster. 
		"""
		start = cu.Event()
		copy_htod = cu.Event()
		compute = cu.Event()
		stop = cu.Event()

		start.record()
		self.spectr_d = gpuarray.to_gpu(data)
		copy_htod.record(); copy_htod.synchronize()
		self.sweep_kernel(self.spectr_d, self.output_d, self.delay_table_d, 
					self.nfreqs, self.ntimes, self.ndelays, 
					block=self.block_size, grid=self.grid_size)
		compute.record(); compute.synchronize()

		if get:
			out = self.output_d.get()
		else:
			operand = self.output_d[self.ndelays//2]
			mean = gpuarray.sum(operand/np.float32(self.nfreqs)).get()
			var = gpuarray.sum((operand-mean)*(operand-mean)/np.float32(self.nfreqs))
			std = np.sqrt(var.get())
			self.threshold_kernel(self.output_d, self.mask_d, np.float32(3*std + mean), 
								self.nfreqs, self.ndelays,
								block=self.block_size, grid=self.grid_size)
			out = (self.output_d*self.mask_d).get()

		stop.record(); stop.synchronize()

		print "{} seconds to load data".format(start.time_till(copy_htod)*1.e-3)
		print "{} seconds to compute {} channels, with {} delays".format(
			copy_htod.time_till(compute)*1.e-3,self.nfreqs, self.ndelays)
		print copy_htod.time_till(compute)/(self.nfreqs*self.ndelays)*1.e6, "nanoseconds per channel, per delay"
		if plot:
			import pylab as plt
			f, axes = plt.subplots(2,1)
			axes[0].imshow(data, aspect='auto', extent=[0,self.nfreqs,0, self.ntimes*self.delta_t], origin='lower')
			axes[0].set_xlabel('Freq [Hz]')
			axes[0].set_ylabel('Time [s]')
			axes[1].imshow(out, aspect='auto', extent=[0,self.nfreqs,-0.1, 0.1], origin='lower')
			axes[1].set_xlabel('Freq [Hz]')
			axes[1].set_ylabel('Drift [Hz/s]')
			plt.tight_layout()
			plt.show()
		return out 

	def run(self, data, get=False):

		""" Function to perform drift rate conversion"""

		self.spectr_d = gpuarray.to_gpu(data)
		self.sweep_kernel(self.spectr_d, self.output_d, self.delay_table_d, 
					self.nfreqs, self.ntimes, self.ndelays, 
					block=self.block_size, grid=self.grid_size)
		if get:
			out = self.output_d.get()
			return out
		else:
			operand = self.output_d[self.ndelays//2]
			mean = gpuarray.sum(operand/np.float32(self.nfreqs))
			var = gpuarray.sum((operand-mean)*(operand-mean)/np.float32(self.nfreqs))
			std = np.sqrt(var.get())
			self.output_d = self.output_d - mean
			thresholded = self.output_d > 3*std
			return thresholded.get()

if __name__ == "__main__":

	ntimes = 16
	nfreqs = 128 

	data = np.zeros((ntimes,nfreqs), dtype=np.float32)
	data[:,nfreqs/2] = 1
	temp = np.arange(ntimes)
	data[temp, temp+nfreqs/2] = 2.

	solver = TurboSolver(ntimes=16, nfreqs=128, ndelays=32, block_size=(16, 32, 1))
	out = solver.bench(data, plot=True)
	#import IPython; IPython.embed()



