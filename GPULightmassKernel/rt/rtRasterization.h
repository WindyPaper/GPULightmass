#pragma once

__global__ void rtVertexTransform()
{

}


__host__ void rtRasterizeModel(const int NumVertices, const int NumTriangles)
{
	/*const int Stride = 64;
	dim3 blockDim(Stride, 1);
	dim3 gridDim(divideAndRoundup(LaunchSizeX * LaunchSizeY, Stride), 1);

	CalculateDirectLighting << <gridDim, blockDim >> > ();*/

	//vertex transform
	int NumVerticesSqrt = std::ceil(std::sqrtf(NumVertices));

}