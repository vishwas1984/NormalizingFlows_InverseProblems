import torch
from torch import nn

import gpytorch
from gpytorch import models, means, kernels, likelihoods


class ExactGPRegression(models.ExactGP):
	def __init__(self,
			x,
			y, 
			mean_fn=means.ZeroMean,
			kernel_fn=kernels.RBFKernel,
			scale_kernel=True,
			likelihood=likelihoods.GaussianLikelihood,
			ard_num_dims=None,
			warp_fn=None
				):
		"""
		INPUTS:
			1. x <torch.tensor> of shape [N, D]
			2. y <torch.tensor> of shape [N,] (note that there is no 2nd dimension)
			3. mean_fn <gpytorch.means.Mean> derived class
			4. kernel_fn <gpytorch.kernels.Kernel> derived class
			5. scale_kernel <Bool> - Whether to wrap the specified kernel function
									 with a scaling factor (or variance). 
			6. ard_num_dims <int> - The number of dimensions of the input to the 
									kernel.
									Note that this may differ from the dimensionality 
									of the input `x` if a warping function is applied 
									which changes the dimensionlity of x. 
									If no integer is passed, this argument is assumed 
									to be the dimensionality of x. 

			7. warp_fn <callable> - A function that is applied to the inputs 
									before passing into the covariance (for example
									an orthogonal linear projection or a 
									nonlinear neural network feature extractor 
									for dimensionality reduction). 
		"""
		super().__init__(x, y, likelihood())

		self.warp_fn = warp_fn
		self.mean_fn = mean_fn()
		
		if ard_num_dims:
			self.kernel_fn = kernel_fn(ard_num_dims=ard_num_dims)
		else:
			self.kernel_fn = kernel_fn()  # 1 lengthscale for all dimensions.

		if scale_kernel:
			self.kernel_fn = kernels.ScaleKernel(self.kernel_fn)

		self.warp_fn = warp_fn


	def forward(self, x):
		"""
		Returns the prior distribution. 

		f \\sim N( f | m(x) , k(x, x'))
		"""
		if self.warp_fn:
			x = self.warp_fn(x)

		m = self.mean_fn(x)
		K = self.kernel_fn(x)
		f = gpytorch.distributions.MultivariateNormal(m, K)

		return f

		


