#pragma once

__global__ void CalculateDirectLighting()
{

	const int TargetTexelLocation = blockIdx.x * blockDim.x + threadIdx.x;

	if (TargetTexelLocation >= BindedSizeX * BindedSizeY)
		return;

	float3 WorldPosition = make_float3(tex1Dfetch(SampleWorldPositionsTexture, TargetTexelLocation));
	float3 WorldNormal = make_float3(tex1Dfetch(SampleWorldNormalsTexture, TargetTexelLocation));

	for (int index = 0; index < NumDirectionalLights; ++index)
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
			OutLightmapData[TargetTexelLocation].IncidentLighting = DirectionalLights[index].Color * make_float3(max(dot(RayInWorldSpace, Normal), 0.0f));
			//OutLightmapData[TargetTexelLocation].IncidentLighting = (Normal);// make_float3(OutHitInfo.TriangleIndex / 5, OutHitInfo.TriangleIndex / 5, OutHitInfo.TriangleIndex / 5);
		}
	}


	//point light
	for (int index = 0; index < NumPointLights; ++index)
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
				OutLightmapData[TargetTexelLocation].IncidentLighting = PointLights[index].Color * make_float3(max(dot(normalize(LightPosition - WorldPosition), WorldNormal), 0.0f)) / (Distance * Distance + 1.0f);
			}
		}
	}

	// SpotLights
	for (int index = 0; index < NumSpotLights; ++index)
	{
		float3 LightPosition = SpotLights[index].WorldPosition;
		float Distance = length(WorldPosition - LightPosition);
		if (Distance < SpotLights[index].Radius)
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
					OutLightmapData[TargetTexelLocation].IncidentLighting = SpotLights[index].Color * make_float3(max(dot(normalize(LightPosition - WorldPosition), WorldNormal), 0.0f)) / (Distance * Distance + 1.0f) * SpotAttenuation;
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