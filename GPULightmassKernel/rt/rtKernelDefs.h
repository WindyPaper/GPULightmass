#pragma once

#include <cuda_runtime.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include <cuda_profiler_api.h>
#include <curand_kernel.h>
#include <cstdio>
#include <linear_math.h>
#include <helper_math.h>
#include <SurfelData.h>

#define USE_CORRELATED_SAMPLING 0

#define USE_JITTERED_SAMPLING 1
#define USE_COSINE_WEIGHTED_HEMISPHERE_SAMPLING 1

#define IRRADIANCE_CACHING_USE_ERROR_THRESHOLD 0
#define IRRADIANCE_CACHING_USE_DISTANCE 1
#define IRRADIANCE_CACHING_VISUALIZE 0
#define IRRADIANCE_CACHING_FORCE_NO_INTERPOLATION 0
#define IRRADIANCE_CACHING_DISTANCE_SCALE 1.0f
const int IRRADIANCE_CACHING_BASE_GRID_SPACING = 32;

struct SamplingGlobalParameters
{
	float FireflyClampingThreshold;
};

__device__ SamplingGlobalParameters GPUSamplingGlobalParameters;

// Old glory params ------------------------------------------------------------------

const int SampleCountOneDimension = 32;

const bool FORCE_SHADOWRAYS = false;

const int TaskBufferSize = 2048 * 2048;

texture<float4, 1, cudaReadModeElementType> BVHTreeNodesTexture;
texture<float4, 1, cudaReadModeElementType> TriangleWoopCoordinatesTexture;
texture<int, 1, cudaReadModeElementType> MappingFromTriangleAddressToIndexTexture;

__constant__ float4* BVHTreeNodes;
__constant__ float4* TriangleWoopCoordinates;
__constant__ int* MappingFromTriangleAddressToIndex;

texture<float4, 1, cudaReadModeElementType> SampleWorldPositionsTexture;
texture<float4, 1, cudaReadModeElementType> SampleWorldNormalsTexture;
texture<float, 1, cudaReadModeElementType> TexelRadiusTexture;

texture<float4, 1, cudaReadModeElementType> SkyLightUpperHemisphereTexture;
texture<float4, 1, cudaReadModeElementType> SkyLightLowerHemisphereTexture;
texture<int, 1, cudaReadModeElementType> SkyLightUpperHemisphereImportantDirectionsTexture;
texture<int, 1, cudaReadModeElementType> SkyLightLowerHemisphereImportantDirectionsTexture;
texture<float4, 1, cudaReadModeElementType> SkyLightUpperHemisphereImportantColorTexture;
texture<float4, 1, cudaReadModeElementType> SkyLightLowerHemisphereImportantColorTexture;

__device__ int SkyLightCubemapNumThetaSteps;
__device__ int SkyLightCubemapNumPhiSteps;

__constant__ float2* VertexTextureUVs;
__constant__ float2* VertexTextureLightmapUVs;
__constant__ int* TriangleMappingIndex;
__constant__ int* TriangleIndexBuffer;

__device__ float3* RasVertexLocalPos;
__device__ float3* RasVertexNormals;
__constant__ float2* RasVertexUVs;
__device__ int* RasTriangleIndexs;
__constant__ float3* RasBBox;
__device__ int RasNumVertices;
__device__ int RasNumTriangles;
__device__ int RasGridElementSize;
__device__ Mat4f* RasViewMat;

__device__ GPULightmass::SurfelData* RasXZPlaneBuffer;

__device__ GPULightmass::LinkListData* RasLinkBuffer;
__device__ int *RasLastIdxNodeBuffer;
__device__ int RasMaxLinkNodeCount;
__device__ int RasCurrLinkCount = 0;
__device__ GPULightmass::SurfelRasIntLinkData *RasIntLightingLinkBuffer;

//__constant__ float4** GatheringRadiosityBuffers;
//__constant__ float4** ShootingRadiosityBuffers;
//__constant__ cudaTextureObject_t* ShootingRadiosityTextures;

//__device__ int *SurfelIndrectedLinkListData;
__device__ GPULightmass::SurfelData* CalculateIndirectedSurfels;
//__device__ int CalculateSurfelsNum;
__device__ GPULightmass::SurfelDirLightingData* SurfelLightingBuffer;
__device__ int *RasSurfelSortOffsetNumBuffer;
__device__ int *SurfelSortLinkBuffer;

__constant__ int BindedSizeX;
__constant__ int BindedSizeY;

namespace GPULightmass
{
struct GatheredLightSample
{
	SHVectorRGB SHVector;
	float SHCorrection;
	float3 IncidentLighting;
	float3 SkyOcclusion;
	float AverageDistance;
	float NumBackfaceHits;

	__device__ __host__ GatheredLightSample& PointLightWorldSpace(const float3 Color, const float3 TangentDirection, const float3 WorldDirection)
	{
		if (TangentDirection.z >= 0.0f)
		{
			SHVector.addIncomingRadiance(Color, 1, WorldDirection);

			SHVector2 SH = SHVector2::basisFunction(TangentDirection);
			SHCorrection += (Color.x * 0.3f + Color.y * 0.59f + Color.z * 0.11f) * (0.282095f * SH.v[0] + 0.325735f * SH.v[2]);
			IncidentLighting += Color * TangentDirection.z;
		}

		return *this;
	}

	__device__ __host__ GatheredLightSample& PointLightWorldSpacePreweighted(const float3 PreweightedColor, const float3 TangentDirection, const float3 WorldDirection)
	{
		if (TangentDirection.z >= 0.0f)
		{
			float3 UnweightedRadiance = PreweightedColor / TangentDirection.z;
			SHVector.addIncomingRadiance(UnweightedRadiance, 1, WorldDirection);

			SHVector2 SH = SHVector2::basisFunction(TangentDirection);
			SHCorrection += getLuminance(UnweightedRadiance) * (0.282095f * SH.v[0] + 0.325735f * SH.v[2]);
			IncidentLighting += PreweightedColor;
		}

		return *this;
	}

	__device__ __host__ GatheredLightSample operator*(float Scalar) const
	{
		GatheredLightSample Result;
		Result.SHVector = SHVector * Scalar;
		Result.SHCorrection = SHCorrection * Scalar;
		Result.IncidentLighting = IncidentLighting * Scalar;
		Result.SkyOcclusion = SkyOcclusion * Scalar;
		Result.AverageDistance = AverageDistance * Scalar;
		Result.NumBackfaceHits = NumBackfaceHits * Scalar;
		return Result;
	}

	__device__ __host__ GatheredLightSample& operator+=(const GatheredLightSample& rhs)
	{
		SHVector += rhs.SHVector;
		SHCorrection += rhs.SHCorrection;
		IncidentLighting += rhs.IncidentLighting;
		SkyOcclusion += rhs.SkyOcclusion;
		AverageDistance += rhs.AverageDistance;
		NumBackfaceHits += rhs.NumBackfaceHits;
		return *this;
	}

	__device__ __host__ GatheredLightSample operator+(const GatheredLightSample& rhs)
	{
		GatheredLightSample Result;
		Result.SHVector = SHVector + rhs.SHVector;
		Result.SHCorrection = SHCorrection + rhs.SHCorrection;
		Result.IncidentLighting = IncidentLighting + rhs.IncidentLighting;
		Result.SkyOcclusion = SkyOcclusion + rhs.SkyOcclusion;
		Result.AverageDistance = AverageDistance + rhs.AverageDistance;
		Result.NumBackfaceHits = NumBackfaceHits + rhs.NumBackfaceHits;
		return Result;
	}

	__device__ __host__ void Reset()
	{
		SHVector.r.reset();
		SHVector.g.reset();
		SHVector.b.reset();
		IncidentLighting = make_float3(0);
		SkyOcclusion = make_float3(0);
		SHCorrection = 0.0f;
		AverageDistance = 0.0f;
		NumBackfaceHits = 0;
	}
};

const int MIXED = 1;
const int ALL_BAKED = 2;

struct DirectionalLight
{
	float3 Color;
	float3 Direction;
	int BakeType;
};

struct PointLight
{
	float3 Color;
	float Radius;
	float3 WorldPosition;
	int BakeType;
};

struct SpotLight
{
	float3 Color;
	float Radius;
	float3 WorldPosition;
	float CosOuterConeAngle;
	float3 Direction;
	float CosInnerConeAngle;
	int BakeType;
};

struct GIVolumeSHData
{
	float4 pos;
	SHVectorRGB SHData;
};

}

__device__ GPULightmass::DirectionalLight* DirectionalLights;
__device__ GPULightmass::PointLight* PointLights;
__device__ GPULightmass::SpotLight* SpotLights;
__device__ int NumDirectionalLights;
__device__ int NumPointLights;
__device__ int NumSpotLights;

__device__ GPULightmass::GatheredLightSample* OutLightmapData;

int LaunchSizeX;
int LaunchSizeY;

__device__ int MappedTexelCounter;

__device__ int* IrradianceWorkspaceBuffer;

__device__ GPULightmass::GIVolumeSHData* BakeGIVolumeSHData;
__device__ int BakeGIVolumeMaxLinkCount;
__device__ int BakeGIVolumeCurrLinkIndex;
__device__ int* BakeGIVolumeLastBuffer;
__device__ GPULightmass::BakeGIVolumeIntLinkData* BakeGIVolumeLinkBuffer;

__align__(16)
struct TaskBuffer
{
	int Size;
	int Buffer[TaskBufferSize];
};


#include "../Radiosity.h"

__device__ int NumTotalSurfaceCaches;
__device__ const SurfaceCacheGPUDataPointers* RadiositySurfaceCaches;
__device__ const cudaTextureObject_t* MaskedCollisionMaps;