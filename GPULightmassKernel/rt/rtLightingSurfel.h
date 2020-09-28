#pragma once


__global__ void CalculateSurfelLighting()
{
	int triangle_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (triangle_index < CalculateSurfelsNum)
	{

	}
}

__host__ void rtRasterizeModel(GPULightmass::SurfelData *surfels, const int SurfelNum)
{
	const int Stride = 64;
	dim3 blockDim(Stride, 1);
	dim3 gridDim(divideAndRoundup(SurfelNum, Stride), 1);
	CalculateSurfelLighting << <gridDim, blockDim >> > ();
}