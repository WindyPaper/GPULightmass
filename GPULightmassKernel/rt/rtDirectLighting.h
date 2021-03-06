#pragma once

__global__ void CalculateDirectLighting()
{

	const int TargetTexelLocation = blockIdx.x * blockDim.x + threadIdx.x;

	if (TargetTexelLocation >= BindedSizeX * BindedSizeY)
		return;

	float3 WorldPosition = make_float3(tex1Dfetch(SampleWorldPositionsTexture, TargetTexelLocation));
	float3 WorldNormal = make_float3(tex1Dfetch(SampleWorldNormalsTexture, TargetTexelLocation));	

	float3 tangent1, tangent2;
	tangent1 = cross(WorldNormal, make_float3(0, 0, 1));
	tangent1 = length(tangent1) < 0.1 ? cross(WorldNormal, make_float3(0, 1, 0)) : tangent1;
	tangent1 = normalize(tangent1);
	tangent2 = normalize(cross(tangent1, WorldNormal));

	for (int index = 0; index < NumDirectionalLights; ++index)
	{
		if (DirectionalLights[index].BakeType == GPULightmass::ALL_BAKED)
		{
			float3 RayInWorldSpace = normalize(-DirectionalLights[index].Direction);

			float3 Normal = WorldNormal;

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
				float3 radiance = DirectionalLights[index].Color;// *make_float3(max(dot(RayInWorldSpace, Normal), 0.0f));

				float3 RayInTangentSpace = WorldToTangent(RayInWorldSpace, tangent1, tangent2, WorldNormal);
				OutLightmapData[TargetTexelLocation].PointLightWorldSpace(radiance, RayInTangentSpace, RayInWorldSpace);
			}
		}
	}


	//point light
	for (int index = 0; index < NumPointLights; ++index)
	{
		if (PointLights[index].BakeType == GPULightmass::ALL_BAKED)
		{
			float3 LightPosition = PointLights[index].WorldPosition;
			float Distance = length(WorldPosition - LightPosition);
			if (Distance < PointLights[index].Radius)
			{
				float3 RayOrigin = WorldPosition + WorldNormal * 0.5f;
				float3 RayInWorldSpace = normalize(LightPosition - WorldPosition);

				HitInfo OutHitInfo;

				rtTrace(
					OutHitInfo,
					make_float4(RayOrigin, 0.01),
					make_float4(RayInWorldSpace, Distance - 0.00001f), true);

				if (OutHitInfo.TriangleIndex == -1)
				{
					float3 radiance = PointLights[index].Color / (Distance * Distance + 1.0f);
					float3 RayInTangentSpace = WorldToTangent(RayInWorldSpace, tangent1, tangent2, WorldNormal);
					OutLightmapData[TargetTexelLocation].PointLightWorldSpace(radiance, RayInTangentSpace, RayInWorldSpace);
				}
			}
		}
	}

	// SpotLights
	for (int index = 0; index < NumSpotLights; ++index)
	{
		if (PointLights[index].BakeType == GPULightmass::ALL_BAKED)
		{
			float3 LightPosition = SpotLights[index].WorldPosition;
			float Distance = length(WorldPosition - LightPosition);
			if (Distance < SpotLights[index].Radius)
			{
				if (dot(normalize(WorldPosition - LightPosition), SpotLights[index].Direction) > SpotLights[index].CosOuterConeAngle)
				{
					float3 RayOrigin = WorldPosition + WorldNormal * 0.5f;
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

						float3 RayInTangentSpace = WorldToTangent(RayInWorldSpace, tangent1, tangent2, WorldNormal);
						float3 radiance = SpotLights[index].Color / (Distance * Distance + 1.0f) * SpotAttenuation;
						OutLightmapData[TargetTexelLocation].IncidentLighting += SpotLights[index].Color / (Distance * Distance + 1.0f) * SpotAttenuation;

						OutLightmapData[TargetTexelLocation].PointLightWorldSpace(radiance, RayInTangentSpace, RayInWorldSpace);
					}
				}
			}
		}
	}
}

__host__ void rtCalculateDirectLighting()
{
	const int Stride = 64;
	dim3 blockDim(Stride, 1);
	dim3 gridDim(divideAndRoundup(LaunchSizeX * LaunchSizeY, Stride), 1);

	CalculateDirectLighting << <gridDim, blockDim >> > ();
}