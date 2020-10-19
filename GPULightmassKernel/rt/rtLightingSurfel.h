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
				//CalculateIndirectedSurfels[surfel_index].diff_alpha += make_float4(1.0f);
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
		//float3 sp = p;		

		transform_vertex((*RasViewMat), p); //to camera space

		p = p / RasGridElementSize;

		const int w = (RasBBox[1].x) - (RasBBox[0].x);
		const int h = (RasBBox[1].z) - (RasBBox[0].z);
		//int surfel_index = max(min(w * int(p.z - RasBBox[0].z) + int(p.x - RasBBox[0].x), w * h - 1), 0); //fixme! should not to be neg num
		float offset_z = p.z - RasBBox[0].z;
		float offset_x = p.x - RasBBox[0].x;
		int curr_plane_surfel_index = w * offset_z + offset_x;
		curr_plane_surfel_index = min(curr_plane_surfel_index, w * h - 1); //fixme! should not to be neg num		
		/*printf("Set idx = %d, p = (%d, %d, %d), left = %f, bottom = %f, w = %d, h = %d\n", 
			curr_plane_surfel_index, int(p.x), int(p.y), int(p.z), RasBBox[0].x, RasBBox[0].z, w, h);*/
		/*printf("p[%d] = (%f, %f, %f), idx = %d, w = %d, h = %d, z = %f, x = %f, minx = %f, minz = %f, maxx = %f, maxz = %f\n", 
			cal_surfel_index, sp.x, sp.y, sp.z, curr_plane_surfel_index,
			w, h, offset_z, offset_x,
			RasBBox[0].x, RasBBox[0].z, RasBBox[1].x, RasBBox[1].z);*/

		if (RasCurrLinkCount > RasMaxLinkNodeCount - 1)
		{
			return; //over flow
		}

		int local_curr_link_count = atomicAdd(&RasCurrLinkCount, 1);		
		
		int old_index = atomicExch(&RasLastIdxNodeBuffer[curr_plane_surfel_index], local_curr_link_count);
		/*printf("SurfelMapToPlane surfel_index = %d, old idx = %d, local_curr_link_count = %d\n", 
			cal_surfel_index, old_index, local_curr_link_count);*/

		//printf("old_index = %d\n", old_index);
		//if (old_index != -1)
		//{
		//	float3 old_pos = make_float3(CalculateIndirectedSurfels[old_index].pos);
		//	float3 curr_pos = make_float3(CalculateIndirectedSurfels[cal_surfel_index].pos);

		//	transform_vertex((*RasViewMat), old_pos); //to camera space
		//	old_pos = old_pos / RasGridElementSize;
		//	old_pos = make_float3(int(old_pos.x), int(old_pos.y), int(old_pos.z));

		//	transform_vertex((*RasViewMat), curr_pos); //to camera space
		//	curr_pos = curr_pos / RasGridElementSize;
		//	curr_pos = make_float3(int(curr_pos.x), int(curr_pos.y), int(curr_pos.z));

		//	int CurrIdx = w * (curr_pos.z - int(RasBBox[0].z)) + (curr_pos.x - int(RasBBox[0].x));
		//	CurrIdx = min(CurrIdx, w * h - 1); //fixme! should not to be neg num		

		//	int OldIdx = w * (old_pos.z - int(RasBBox[0].z)) + (old_pos.x - int(RasBBox[0].x));
		//	OldIdx = min(OldIdx, w * h - 1); //fixme! should not to be neg num		

		//	printf("old_pos(%f, %f, %f), curr_pos(%f, %f, %f), old idx = %d, curr_idx = %d \n", old_pos.x, old_pos.y, old_pos.z,
		//		curr_pos.x, curr_pos.y, curr_pos.z, OldIdx, CurrIdx);
		//}

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

		//printf("SurfelNum = %d\n", SurfelNum);
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
			if (ndl_f > 0.0f &&
				ndl_n > 0.0f)
			{
				float d = length(face_offset);
				
				float3 n_diff = make_float3(ndata.diff_alpha);
				float3 f_diff = make_float3(fdata.diff_alpha);
				float3 radiance_f = ndl_f * n_diff;// / (d * d + 1.0f); //fixme!
				float3 radiance_n = ndl_n * f_diff;// / (d * d + 1.0f);				

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

__global__ void SurfelRadianceToSrcTest()
{
	int surfel_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (surfel_index < RasMaxLinkNodeCount)
	{
		CalculateIndirectedSurfels[surfel_index].diff_alpha = SurfelLightingBuffer->radiance[0][surfel_index];		
	}
}

__host__ void rtSurfelRadianceToSrcTest(const int SurfelNum)
{
	const int Stride = 64;
	dim3 blockDim(Stride, 1);
	dim3 gridDim(divideAndRoundup(SurfelNum, Stride), 1);
	SurfelRadianceToSrcTest << <gridDim, blockDim >> > ();
}