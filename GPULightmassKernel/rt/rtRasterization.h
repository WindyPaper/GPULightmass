#pragma once

__global__ void rtVertexTransform()
{
	int vertex_id = blockIdx.x * blockDim.x + threadIdx.x;

	if (vertex_id < RasNumVertices)
	{

	}
}

__global__ void PlaneRasterization()
{
	int triangle_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (triangle_index < RasNumTriangles)
	{
		int index_0 = RasTriangleIndexs[triangle_index * 3];
		int index_1 = RasTriangleIndexs[triangle_index * 3 + 1];
		int index_2 = RasTriangleIndexs[triangle_index * 3 + 2];

		float3 p0 = RasVertexLocalPos[index_0];
		float3 p1 = RasVertexLocalPos[index_1];
		float3 p2 = RasVertexLocalPos[index_2];

		//map to yz plane
		
	}
}


__host__ void rtRasterizeModel(const int NumVertices, const int NumTriangles)
{
	/*const int Stride = 64;
	dim3 blockDim(Stride, 1);
	dim3 gridDim(divideAndRoundup(LaunchSizeX * LaunchSizeY, Stride), 1);

	CalculateDirectLighting << <gridDim, blockDim >> > ();*/

	//vertex transform
	int NumVerticesSqrt = std::ceil(std::sqrtf(NumVertices));
	dim3 blockDim(NumVerticesSqrt);
	dim3 gridDim(NumVerticesSqrt);

	rtVertexTransform << <gridDim, blockDim >> > ();
}