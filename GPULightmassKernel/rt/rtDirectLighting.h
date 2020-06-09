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
}

__host__ void rtCalculateDirectLighting()
{
	const int Stride = 64;
	dim3 blockDim(Stride, 1);
	dim3 gridDim(divideAndRoundup(LaunchSizeX * LaunchSizeY, Stride), 1);

	CalculateDirectLighting << <gridDim, blockDim >> > ();
}