#pragma once


//__global__ void CalculateSurfelLighting()
//{
//	int triangle_index = blockIdx.x * blockDim.x + threadIdx.x;
//
//	if (triangle_index < CalculateSurfelsNum)
//	{
//
//	}
//}
//
//__host__ void rtRasterizeModel(const int SurfelNum)
//{
//	const int Stride = 64;
//	dim3 blockDim(Stride, 1);
//	dim3 gridDim(divideAndRoundup(SurfelNum, Stride), 1);
//	CalculateSurfelLighting << <gridDim, blockDim >> > ();
//}

__global__ void CalSurfelDirectLighting()
{
	int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (surfel_index < RasMaxLinkNodeCount)
	{
		float3 Normal = make_float3(CalculateIndirectedSurfels[surfel_index].normal);
		float3 WorldPosition = make_float3(CalculateIndirectedSurfels[surfel_index].pos);
		for (int index = 0; index < NumDirectionalLights; ++index)
		{
			float3 RayInWorldSpace = normalize(-DirectionalLights[index].Direction);			

			/*if (ReflectanceMap[TargetTexelLocation].w == 1.0f && dot(RayInWorldSpace, Normal) < 0.0f)
				Normal = -Normal;*/

			float3 RayOrigin = WorldPosition + Normal * 0.5f;

			HitInfo OutHitInfo;

			rtTrace(
				OutHitInfo,
				make_float4(RayOrigin, 0.01),
				make_float4(RayInWorldSpace, 1e20), true);

			if (OutHitInfo.TriangleIndex == -1)
			{
				float3 radiance = DirectionalLights[index].Color * make_float3(max(dot(RayInWorldSpace, Normal), 0.0f));

				//float3 RayInTangentSpace = WorldToTangent(RayInWorldSpace, tangent1, tangent2, WorldNormal);
				//OutLightmapData[TargetTexelLocation].PointLightWorldSpace(radiance, RayInTangentSpace, RayInWorldSpace);
				CalculateIndirectedSurfels[surfel_index].diff_alpha += make_float4(radiance, 0.0f);
			}			
		}
	}
}

__host__ void rtSurfelDirectLighting(const int SurfelNum)
{
	const int Stride = 64;
	dim3 blockDim(Stride, 1);
	dim3 gridDim(divideAndRoundup(SurfelNum, Stride), 1);
	CalSurfelDirectLighting << <gridDim, blockDim >> > ();
}

__global__ void CalSurfelIndirectedLighting()
{
	int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (surfel_index < RasMaxLinkNodeCount)
	{
		float3 p = make_float3(CalculateIndirectedSurfels[surfel_index].pos);		

		transform_vertex((*RasViewMat), p); //to camera space

		p = p / RasGridElementSize;

		int w = RasBBox[1].x - RasBBox[0].x;
		int surfel_index = w * int(p.z - RasBBox[0].z) + (p.x - RasBBox[0].x);

		if (RasCurrLinkCount > RasMaxLinkNodeCount)
		{
			return; //over flow
		}

		int local_curr_link_count = atomicAdd(&RasCurrLinkCount, 1);		
		
		int old_index = atomicExch(&RasLastIdxNodeBuffer[surfel_index], local_curr_link_count);
		RasIntLightingLinkBuffer[local_curr_link_count] = old_index;
		//printf("%d\n", RasIntLightingLinkBuffer[local_curr_link_count]);
	}
}

__host__ void rtSurfelIndirectedLighting(const int SurfelNum)
{
	const int Stride = 64;
	dim3 blockDim(Stride, 1);
	dim3 gridDim(divideAndRoundup(SurfelNum, Stride), 1);
	CalSurfelIndirectedLighting << <gridDim, blockDim >> > ();
}