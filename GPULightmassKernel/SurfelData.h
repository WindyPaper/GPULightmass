#pragma once

//class float3;
//class float4;
namespace GPULightmass
{
	struct SurfelData
	{
		float4 pos;
		float4 normal;
		float4 diff_alpha;
	};

	struct SurfelLinkData
	{
		float2 uvw;
		int triangle_index;
	};

	struct LinkListData
	{
		SurfelLinkData data;
		int prev_index;

		LinkListData()
		{
			memset(&data, 0, sizeof(SurfelLinkData));
			prev_index = -1;
		}
	};
}