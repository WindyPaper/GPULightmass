#pragma once

#include "rt/rtConstUtil.h"

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
				float3 radiance = DirectionalLights[index].Color * make_float3(max(dot(RayInWorldSpace, Normal), 0.0f)) / PI;

				//float3 RayInTangentSpace = WorldToTangent(RayInWorldSpace, tangent1, tangent2, WorldNormal);
				//OutLightmapData[TargetTexelLocation].PointLightWorldSpace(radiance, RayInTangentSpace, RayInWorldSpace);
				CalculateIndirectedSurfels[surfel_index].diff_alpha += make_float4(radiance, 0.0f);
				//CalculateIndirectedSurfels[surfel_index].diff_alpha += make_float4(1.0f);
			}			
		}

		//point light
		for (int index = 0; index < NumPointLights; ++index)
		{
			//if (PointLights[index].BakeType == GPULightmass::ALL_BAKED)
			{
				float3 LightPosition = PointLights[index].WorldPosition;
				float Distance = length(WorldPosition - LightPosition);
				if (Distance < PointLights[index].Radius)
				{
					float3 RayOrigin = WorldPosition + Normal * 0.5f;
					float3 RayInWorldSpace = normalize(LightPosition - WorldPosition);

					HitInfo OutHitInfo;

					rtTrace(
						OutHitInfo,
						make_float4(RayOrigin, 0.01),
						make_float4(RayInWorldSpace, Distance - 0.00001f), true);

					if (OutHitInfo.TriangleIndex == -1)
					{
						float3 radiance = PointLights[index].Color / (Distance * Distance + 1.0f);
						radiance = radiance * make_float3(max(dot(RayInWorldSpace, Normal), 0.0f)) / PI;
						//float3 RayInTangentSpace = WorldToTangent(RayInWorldSpace, tangent1, tangent2, WorldNormal);
						//OutLightmapData[TargetTexelLocation].PointLightWorldSpace(radiance, RayInTangentSpace, RayInWorldSpace);
						CalculateIndirectedSurfels[surfel_index].diff_alpha += make_float4(radiance, 0.0f);
					}
				}
			}
		}

		// SpotLights
		for (int index = 0; index < NumSpotLights; ++index)
		{
			//if (PointLights[index].BakeType == GPULightmass::ALL_BAKED)
			{
				float3 LightPosition = SpotLights[index].WorldPosition;
				float Distance = length(WorldPosition - LightPosition);
				if (Distance < SpotLights[index].Radius)
				{
					if (dot(normalize(WorldPosition - LightPosition), SpotLights[index].Direction) > SpotLights[index].CosOuterConeAngle)
					{
						float3 RayOrigin = WorldPosition + Normal * 0.5f;
						float3 RayInWorldSpace = normalize(LightPosition - WorldPosition);

						HitInfo OutHitInfo;

						rtTrace(
							OutHitInfo,
							make_float4(RayOrigin, 0.01),
							make_float4(RayInWorldSpace, Distance - 0.00001f), true);

						if (OutHitInfo.TriangleIndex == -1)
						{
							float SpotAttenuation = clampf(
								(dot(normalize(WorldPosition - LightPosition), SpotLights[index].Direction) - SpotLights[index].CosOuterConeAngle) / (SpotLights[index].CosInnerConeAngle - SpotLights[index].CosOuterConeAngle)
								, 0.0f, 1.0f);
							SpotAttenuation *= SpotAttenuation;

							//float3 RayInTangentSpace = WorldToTangent(RayInWorldSpace, tangent1, tangent2, WorldNormal);
							float3 radiance = SpotLights[index].Color / (Distance * Distance + 1.0f) * SpotAttenuation;
							radiance = radiance * make_float3(max(dot(RayInWorldSpace, Normal), 0.0f)) / PI;
							//OutLightmapData[TargetTexelLocation].IncidentLighting += SpotLights[index].Color / (Distance * Distance + 1.0f) * SpotAttenuation;
							CalculateIndirectedSurfels[surfel_index].diff_alpha += make_float4(radiance, 0.0f);
							//OutLightmapData[TargetTexelLocation].PointLightWorldSpace(radiance, RayInTangentSpace, RayInWorldSpace);
						}
					}
				}
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

__global__ void SurfelMapToPlane()
{
	int cal_surfel_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (cal_surfel_index < RasMaxLinkNodeCount)
	{
		float3 p = make_float3(CalculateIndirectedSurfels[cal_surfel_index].pos);

		transform_vertex((*RasViewMat), p); //to camera space
		p = p / RasGridElementSize;

		const int w = (RasBBox[1].x) - (RasBBox[0].x);
		const int h = (RasBBox[1].z) - (RasBBox[0].z);
	
		float offset_z = floor(p.z - RasBBox[0].z);
		float offset_x = floor(p.x - RasBBox[0].x);
		int curr_plane_surfel_index = w * offset_z + offset_x;
		curr_plane_surfel_index = min(curr_plane_surfel_index, w * h - 1);
		

		if (RasCurrLinkCount > RasMaxLinkNodeCount - 1)
		{
			return; //over flow
		}

		int local_curr_link_count = atomicAdd(&RasCurrLinkCount, 1);		
		
		int old_index = atomicExch(&RasLastIdxNodeBuffer[curr_plane_surfel_index], local_curr_link_count);		

		RasIntLightingLinkBuffer[local_curr_link_count].PrevIndex = old_index;
		RasIntLightingLinkBuffer[local_curr_link_count].SurfelIndex = cal_surfel_index;		
	}
}

__host__ void rtSurfelMapToPlane(const int SurfelNum)
{
	const int Stride = 64;
	dim3 blockDim(Stride, 1);
	dim3 gridDim(divideAndRoundup(SurfelNum, Stride), 1);
	SurfelMapToPlane << <gridDim, blockDim >> > ();
}

__global__ void GIVolumeDataMapToPlane()
{
	int BakeGIVolumeIdx = blockIdx.x * blockDim.x + threadIdx.x;

	if (BakeGIVolumeIdx < BakeGIVolumeMaxLinkCount)
	{
		float3 p = make_float3(BakeGIVolumeSHData[BakeGIVolumeIdx].pos);

		transform_vertex((*RasViewMat), p); //to camera space
		p = p / RasGridElementSize;

		const int w = (RasBBox[1].x) - (RasBBox[0].x);
		const int h = (RasBBox[1].z) - (RasBBox[0].z);

		float offset_z = floor(p.z - RasBBox[0].z);
		float offset_x = floor(p.x - RasBBox[0].x);
		int curr_plane_surfel_index = w * offset_z + offset_x;
		curr_plane_surfel_index = min(curr_plane_surfel_index, w * h - 1);


		if (BakeGIVolumeCurrLinkIndex > BakeGIVolumeMaxLinkCount - 1)
		{
			return; //over flow
		}

		int local_curr_link_count = atomicAdd(&BakeGIVolumeCurrLinkIndex, 1);

		//BakeGIVolumeLastBuffer
		int old_index = atomicExch(&BakeGIVolumeLastBuffer[curr_plane_surfel_index], local_curr_link_count);

		//BakeGIVolumeLinkBuffer
		BakeGIVolumeLinkBuffer[local_curr_link_count].PrevIndex = old_index;
		BakeGIVolumeLinkBuffer[local_curr_link_count].GIVolumeDataIndex = BakeGIVolumeIdx;
	}
}

__host__ void rtGIVolumeMapToPlane(const int GIVolumeNum)
{
	const int Stride = 64;
	dim3 blockDim(Stride, 1);
	dim3 gridDim(divideAndRoundup(GIVolumeNum, Stride), 1);
	GIVolumeDataMapToPlane << <gridDim, blockDim >> > ();
}

__global__ void GenerateSurfelNumPlane()
{
	int BufferIdx = blockIdx.x * blockDim.x + threadIdx.x;
	const int w = RasBBox[1].x - RasBBox[0].x;
	const int h = RasBBox[1].z - RasBBox[0].z;
	const int PlaneBufferSize = w * h;

	if (BufferIdx < PlaneBufferSize)
	{
		int head = RasLastIdxNodeBuffer[BufferIdx];

		int next = head;
		int SurfelNum = 0;
		while (next != -1)
		{
			next = RasIntLightingLinkBuffer[next].PrevIndex;
			++SurfelNum;
		}

		//printf("BufferIdx = %d, SurfelNum = %d\n", BufferIdx, SurfelNum);
		RasSurfelSortOffsetNumBuffer[BufferIdx] = SurfelNum;
	}
}

__global__ void SortingAndLightingSurfel()
{
	int BufferIdx = blockIdx.x * blockDim.x + threadIdx.x;
	const int w = RasBBox[1].x - RasBBox[0].x;
	const int h = RasBBox[1].z - RasBBox[0].z;
	const int PlaneBufferSize = w * h;

	if (BufferIdx < PlaneBufferSize)
	{		
		int offset = 0;
		for (int i = 0; i < BufferIdx; ++i)
		{
			offset += RasSurfelSortOffsetNumBuffer[i];
		}

		//Get elements
		int head = RasLastIdxNodeBuffer[BufferIdx];
		int next = head;
		int idx_count = 0;
		while (next != -1)
		{
			SurfelSortLinkBuffer[offset + idx_count] = RasIntLightingLinkBuffer[next].SurfelIndex;

			next = RasIntLightingLinkBuffer[next].PrevIndex;
			++idx_count;
		}

		/*if (idx_count != RasSurfelSortOffsetNumBuffer[BufferIdx])
		{
			printf("error index count = %d\n", idx_count);
		}*/

		//printf("index count = %d\n", idx_count);

		//Sorting
		for (int i = 1; i < idx_count; ++i)
		{
			for (int j = i; j > 0; --j)
			{
				int CurrIndex = SurfelSortLinkBuffer[offset + j];
				int PrevIndex = SurfelSortLinkBuffer[offset + j - 1];

				float3 CurrCamPos = make_float3(CalculateIndirectedSurfels[CurrIndex].pos);
				float3 PrevCamPos = make_float3(CalculateIndirectedSurfels[PrevIndex].pos);
				transform_vertex((*RasViewMat), CurrCamPos);
				transform_vertex((*RasViewMat), PrevCamPos);
				if (CurrCamPos.y < PrevCamPos.y)
				{
					int temp = SurfelSortLinkBuffer[offset + j];
					SurfelSortLinkBuffer[offset + j] = SurfelSortLinkBuffer[offset + j - 1];
					SurfelSortLinkBuffer[offset + j - 1] = temp;
				}
			}
		}

		//debug print
		/*printf("start bufferidx = %d\n", BufferIdx);
		for (int i = 0; i < idx_count; ++i)
		{
			int CurrIndex = SurfelSortLinkBuffer[offset + i];
			float3 CurrCamPos = make_float3(CalculateIndirectedSurfels[CurrIndex].pos);
			transform_vertex((*RasViewMat), CurrCamPos);
			printf("sort index = %d, pos = (%f, %f, %f)\n", i, CurrCamPos.x, CurrCamPos.y, CurrCamPos.z);
		}
		printf("end bufferidx = %d\n", BufferIdx);*/

		//Lighting
		for (int i = 0; i < idx_count - 1; ++i)
		{
			int FrontSurfelIdx = SurfelSortLinkBuffer[offset + i];
			int NextSurfelIdx = SurfelSortLinkBuffer[offset + i + 1];

			const GPULightmass::SurfelData &fdata = CalculateIndirectedSurfels[FrontSurfelIdx];
			const GPULightmass::SurfelData &ndata = CalculateIndirectedSurfels[NextSurfelIdx];

			//Is face to face?
			float3 face_offset = make_float3(ndata.pos) - make_float3(fdata.pos);
			float3 f_to_n = normalize(face_offset);
			float ndl_f = max(dot(f_to_n, make_float3(fdata.normal)), 0.0f);
			float ndl_n = max(dot(-f_to_n, make_float3(ndata.normal)), 0.0f);
			float diff_brdf = 1.0f / PI;
			if (ndl_f > 0.0f &&
				ndl_n > 0.0f)
			{
				//float d = length(face_offset);
				
				float3 n_diff = make_float3(ndata.diff_alpha);
				float3 f_diff = make_float3(fdata.diff_alpha);
				float3 radiance_f = ndl_f * n_diff * diff_brdf * 2.0f * PI; // / (d * d / 10000.0f + 1.0f); //fixme!
				float3 radiance_n = ndl_n * f_diff * diff_brdf * 2.0f * PI; // / (d * d / 10000.0f + 1.0f);

				//save radiances
				//fixme! 0
				SurfelLightingBuffer[0].radiance[0][FrontSurfelIdx] += make_float4(radiance_f, 0.0f);
				SurfelLightingBuffer[0].radiance[0][NextSurfelIdx] += make_float4(radiance_n, 0.0f);
			}
		}
	}
}

__host__ void rtSurfelSortAndLighting(const int PlaneSize)
{
	const int Stride = 64;
	dim3 blockDim(Stride, 1);
	dim3 gridDim(divideAndRoundup(PlaneSize, Stride), 1);
	
	GenerateSurfelNumPlane << <gridDim, blockDim >> > ();

	SortingAndLightingSurfel << <gridDim, blockDim >> > ();
}

__global__ void SortingAndLightingBaking()
{
	int BufferIdx = blockIdx.x * blockDim.x + threadIdx.x;
	const int w = RasBBox[1].x - RasBBox[0].x;
	const int h = RasBBox[1].z - RasBBox[0].z;
	const int PlaneBufferSize = w * h;

	if (BufferIdx < PlaneBufferSize)
	{
		int offset = 0;
		for (int i = 0; i < BufferIdx; ++i)
		{
			offset += RasSurfelSortOffsetNumBuffer[i];
		}

		//Get elements
		int head = RasLastIdxNodeBuffer[BufferIdx];
		int next = head;
		int idx_count = 0;
		while (next != -1)
		{
			SurfelSortLinkBuffer[offset + idx_count] = RasIntLightingLinkBuffer[next].SurfelIndex;

			next = RasIntLightingLinkBuffer[next].PrevIndex;
			++idx_count;
		}		

		//Sorting
		for (int i = 1; i < idx_count; ++i)
		{
			for (int j = i; j > 0; --j)
			{
				int CurrIndex = SurfelSortLinkBuffer[offset + j];
				int PrevIndex = SurfelSortLinkBuffer[offset + j - 1];

				float3 CurrCamPos = make_float3(CalculateIndirectedSurfels[CurrIndex].pos);
				float3 PrevCamPos = make_float3(CalculateIndirectedSurfels[PrevIndex].pos);
				transform_vertex((*RasViewMat), CurrCamPos);
				transform_vertex((*RasViewMat), PrevCamPos);
				if (CurrCamPos.y < PrevCamPos.y)
				{
					int temp = SurfelSortLinkBuffer[offset + j];
					SurfelSortLinkBuffer[offset + j] = SurfelSortLinkBuffer[offset + j - 1];
					SurfelSortLinkBuffer[offset + j - 1] = temp;
				}
			}
		}		

		//Lighting
		for (int i = 0; i < idx_count - 1; ++i)
		{
			int FrontSurfelIdx = SurfelSortLinkBuffer[offset + i];
			int NextSurfelIdx = SurfelSortLinkBuffer[offset + i + 1];

			const GPULightmass::SurfelData &fdata = CalculateIndirectedSurfels[FrontSurfelIdx];
			const GPULightmass::SurfelData &ndata = CalculateIndirectedSurfels[NextSurfelIdx];

			//Is face to face?
			float3 face_offset = make_float3(ndata.pos) - make_float3(fdata.pos);
			float3 f_to_n = normalize(face_offset);
			float ndl_f = max(dot(f_to_n, make_float3(fdata.normal)), 0.0f);
			float ndl_n = max(dot(-f_to_n, make_float3(ndata.normal)), 0.0f);
			float diff_brdf = 1.0f / PI;
			if (ndl_f > 0.0f &&
				ndl_n > 0.0f)
			{
				//float d = length(face_offset);

				float3 n_diff = make_float3(ndata.diff_alpha);
				float3 f_diff = make_float3(fdata.diff_alpha);
				float3 radiance_f = ndl_f * n_diff * diff_brdf * 2.0f * PI; // / (d * d / 10000.0f + 1.0f); //fixme!
				float3 radiance_n = ndl_n * f_diff * diff_brdf * 2.0f * PI; // / (d * d / 10000.0f + 1.0f);

				//save radiances
				//fixme! 0 100.0f
				SurfelLightingBuffer[0].radiance[0][FrontSurfelIdx] += make_float4(radiance_f, 0.0f);
				SurfelLightingBuffer[0].radiance[0][NextSurfelIdx] += make_float4(radiance_n, 0.0f);
			}
		}

		//Bake SH Volume
		int GIVolumeDataHead = BakeGIVolumeLastBuffer[BufferIdx];
		int GIVolumeDataNext = GIVolumeDataHead;
		//int idx_count = 0;
		while (GIVolumeDataNext != -1)
		{
			int GIVolumeDataIdx = BakeGIVolumeLinkBuffer[GIVolumeDataNext].GIVolumeDataIndex;
			GPULightmass::GIVolumeSHData &shdata = BakeGIVolumeSHData[GIVolumeDataIdx];

			float3 GIVolumeDataWPos = make_float3(shdata.pos);
			float3 GIVolumeDataCamPos = GIVolumeDataWPos;
			transform_vertex((*RasViewMat), GIVolumeDataCamPos);

			//shdata.SHData.addIncomingRadiance(make_float3(0.5, 0.5, 0.5), 1.0f, normalize(GIVolumeDataWPos));

			//Find nearest surfel
			for (int i = 0; i < idx_count; ++i)
			{
				int CurrIndex = SurfelSortLinkBuffer[offset + i];

				float3 SurfelWPos = make_float3(CalculateIndirectedSurfels[CurrIndex].pos);
				float3 SurfelWNormal = make_float3(CalculateIndirectedSurfels[CurrIndex].normal);
				float3 SurfelDiff = make_float3(CalculateIndirectedSurfels[CurrIndex].diff_alpha);
				float3 SurfelCamPos = SurfelWPos;
				transform_vertex((*RasViewMat), SurfelCamPos);

				float diff_brdf = 1.0f / PI;
				if (GIVolumeDataCamPos.y < SurfelCamPos.y)
				{
					//isLastOne = false;
					//Calculate GIVolume lighting
					float3 VolumeToSurfelDir = normalize(SurfelWPos - GIVolumeDataWPos);
					float ndl = dot(SurfelWNormal, -VolumeToSurfelDir);
					if (ndl > 0.0f)
					{												
						float3 radiance = SurfelDiff * diff_brdf * 2.0f * PI;
						shdata.SHData.addIncomingRadiance(radiance * 100.0f, 1.0f, VolumeToSurfelDir);
					}					

					if (i > 0)
					{
						int PrevIndex = SurfelSortLinkBuffer[offset + i - 1];
						SurfelWPos = make_float3(CalculateIndirectedSurfels[PrevIndex].pos);
						SurfelWNormal = make_float3(CalculateIndirectedSurfels[PrevIndex].normal);
						SurfelDiff = make_float3(CalculateIndirectedSurfels[PrevIndex].diff_alpha);
						SurfelCamPos = SurfelWPos;
						transform_vertex((*RasViewMat), SurfelCamPos);

						VolumeToSurfelDir = normalize(SurfelWPos - GIVolumeDataWPos);
						ndl = dot(SurfelWNormal, -VolumeToSurfelDir);
						if (ndl > 0.0f)
						{
							float3 radiance = SurfelDiff * diff_brdf * 2.0f * PI;
							shdata.SHData.addIncomingRadiance(radiance * 100.0f, 1.0f, VolumeToSurfelDir);
						}						
					}

					break;
				}

				if (i == idx_count - 1) //GI Volume data is last one
				{
					int LastIndex = SurfelSortLinkBuffer[offset + i];
					SurfelWPos = make_float3(CalculateIndirectedSurfels[LastIndex].pos);
					SurfelWNormal = make_float3(CalculateIndirectedSurfels[LastIndex].normal);
					SurfelDiff = make_float3(CalculateIndirectedSurfels[LastIndex].diff_alpha);
					SurfelCamPos = SurfelWPos;
					transform_vertex((*RasViewMat), SurfelCamPos);

					float3 VolumeToSurfelDir = normalize(SurfelWPos - GIVolumeDataWPos);
					float ndl = dot(SurfelWNormal, -VolumeToSurfelDir);
					if (ndl > 0.0f)
					{
						float3 radiance = SurfelDiff * diff_brdf * 2.0f * PI;
						shdata.SHData.addIncomingRadiance(radiance, 1.0f, VolumeToSurfelDir);
					}
				}
			}			

			//next gi volume data
			GIVolumeDataNext = BakeGIVolumeLinkBuffer[GIVolumeDataNext].PrevIndex;
		}
	}
}

__host__ void rtSurfelSortAndLightingBaking(const int PlaneSize)
{
	const int Stride = 64;
	dim3 blockDim(Stride, 1);
	dim3 gridDim(divideAndRoundup(PlaneSize, Stride), 1);

	GenerateSurfelNumPlane << <gridDim, blockDim >> > ();

	SortingAndLightingBaking << <gridDim, blockDim >> > ();
}

__global__ void SurfelRadianceToSrcTest()
{
	int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (surfel_index < RasMaxLinkNodeCount)
	{
		CalculateIndirectedSurfels[surfel_index].diff_alpha = SurfelLightingBuffer->radiance[0][surfel_index]/(8.0f * 8.0f);		
	}
}

__host__ void rtSurfelRadianceToSrcTest(const int SurfelNum)
{
	const int Stride = 64;
	dim3 blockDim(Stride, 1);
	dim3 gridDim(divideAndRoundup(SurfelNum, Stride), 1);
	SurfelRadianceToSrcTest << <gridDim, blockDim >> > ();
}
