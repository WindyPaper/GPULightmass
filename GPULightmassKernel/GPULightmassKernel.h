#pragma once

namespace GPULightmass
{

#ifdef GPULIGHTMASSKERNEL_LIB
#define GPULIGHTMASSKERNEL_API extern "C" __declspec(dllexport)
#else

#define GPULIGHTMASSKERNEL_API extern "C" __declspec(dllimport)

	struct int3
	{
		int x, y, z;
	};

	struct float2
	{
		float x, y;
	};

	struct float3
	{
		float x, y, z;
	};

	struct float4
	{
		float x, y, z, w;
	};

	float3 make_float3(float x, float y, float z)
	{
		float3 f;
		f.x = x;
		f.y = y;
		f.z = z;

		return f;
	}

	float2 make_float2(float x, float y)
	{
		float2 f;
		f.x = x;
		f.y = y;

		return f;
	}

	int3 make_int3(int x, int y, int z)
	{
		int3 f;
		f.x = x;
		f.y = y;
		f.z = z;

		return f;
	}

#endif

struct shvector2
{
	float v[4];
};

struct shvector3
{
	float v[9];
};

struct shvectorrgb
{
	shvector2 r;
	shvector2 g;
	shvector2 b;
};

struct shvectorrgb3
{
	shvector3 r;
	shvector3 g;
	shvector3 b;
};

struct GatheredLightSample
{
	shvectorrgb SHVector;
	float SHCorrection;
	float3 IncidentLighting;
	float3 SkyOcclusion;
	float AverageDistance;
	float NumBackfaceHits;
};

struct VolumetricLightSample
{
	shvectorrgb3 SHVector;
	float3 IncidentLighting;
	float3 SkyOcclusion;
	float MinDistance;
	float BackfacingHitsFraction;
};

enum BakedType
{
	//REAL_TIME = 0, //不需要用，占位
	MIXED = 1,
	ALL_BAKED = 2
};

struct DirectionalLight
{
	float3 Color;
	float3 Direction;
	BakedType type;
};

struct PointLight
{
	float3 Color;
	float Radius;
	float3 WorldPosition;
	BakedType type;
};

struct SpotLight
{
	float3 Color;
	float Radius;
	float3 WorldPosition;
	float CosOuterConeAngle;
	float3 Direction;
	float CosInnerConeAngle;
	BakedType type;
};

typedef void(*GPULightmassLogHandler)(const wchar_t* message);

GPULIGHTMASSKERNEL_API void SetLogHandler(GPULightmassLogHandler LogHandler);

GPULIGHTMASSKERNEL_API void ImportAggregateMesh(
	const int NumVertices,
	const int NumTriangles,
	const float3 VertexWorldPositionBuffer[],
	const float2 VertexTextureUVBuffer[],
	const float2 VertexLightmapUVBuffer[],
	const int3 TriangleIndexBuffer[],
	const int TriangleMaterialIndex[],
	const int TriangleTextureMappingIndex[]
);

GPULIGHTMASSKERNEL_API void ImportMaterialMaps(
	const int NumMaterials,
	const int SizeXY,
	float** MapData
);

GPULIGHTMASSKERNEL_API void ImportSurfaceCache(
	const int ID,
	const int SurfaceCacheSizeX,
	const int SurfaceCacheSizeY,
	const float4 WorldPositionMap[],
	const float4 WorldNormalMap[],
	const float4 ReflectanceMap[],
	const float4 EmissiveMap[]
);

GPULIGHTMASSKERNEL_API void RunRadiosity(int NumBounces, int NumSamplesFirstPass);

GPULIGHTMASSKERNEL_API void SetTotalTexelsForProgressReport(
	const size_t NumTotalTexels
);

GPULIGHTMASSKERNEL_API void CalculateIndirectLightingTextureMapping(
	const size_t NumTexelsInCurrentBatch,
	const int CachedSizeX,
	const int CachedSizeY,
	const int NumSamples,
	const float4 WorldPositionMap[],
	const float4 WorldNormalMap[],
	const float TexelRadiusMap[],
	GatheredLightSample OutLightmapData[]
);

GPULIGHTMASSKERNEL_API void ImportSkyLightCubemap(
	const int NumThetaSteps,
	const int NumPhiSteps,
	float4 UpperHemisphereCubemap[],
	float4 LowerHemisphereCubemap[]
);

GPULIGHTMASSKERNEL_API void ImportPunctualLights(
	const int NumDirectionalLights,
	const DirectionalLight DirectionalLights[],
	const int NumPointLights,
	const PointLight PointLights[],
	const int NumSpotLights,
	const SpotLight SpotLights[]
);

GPULIGHTMASSKERNEL_API void SetGlobalSamplingParameters(
	float FireflyClampingThreshold
);

GPULIGHTMASSKERNEL_API void PreallocRadiositySurfaceCachePointers(const int Num);

GPULIGHTMASSKERNEL_API void CalculateVolumetricLightmapBrickSamples(
	const int BrickSize,
	const float3 WorldBrickMin,
	const float3 WorldChildCellSize,
	VolumetricLightSample InOutVolumetricBrickUpperSamples[],
	VolumetricLightSample InOutVolumetricBrickLowerSamples[]
);

GPULIGHTMASSKERNEL_API void CalculateVolumeSampleList(
	const int NumSamples,
	const float3 WorldPositions[],
	VolumetricLightSample InOutVolumetricBrickUpperSamples[],
	VolumetricLightSample InOutVolumetricBrickLowerSamples[]
);

GPULIGHTMASSKERNEL_API void ImportDirectLights(
	const int NumDirectionalLights,
	const DirectionalLight DirectionalLights[],
	const int NumPointLights,
	const PointLight PointLights[],
	const int NumSpotLights,
	const SpotLight SpotLights[]
);

GPULIGHTMASSKERNEL_API void CalculateDirectLightingAndShadow(
	const size_t NumTexelsInCurrentBatch,
	const int CachedSizeX,
	const int CachedSizeY,
	const int NumSamples,
	const float4 WorldPositionMap[],
	const float4 WorldNormalMap[],
	const float TexelRadiusMap[],
	GatheredLightSample OutLightmapData[]
);

//direct + indirected
GPULIGHTMASSKERNEL_API void CalculateAllLightingAndShadow(
	const size_t NumTexelsInCurrentBatch,
	const int CachedSizeX,
	const int CachedSizeY,
	const int NumSamples,
	const float4 WorldPositionMap[],
	const float4 WorldNormalMap[],
	const float TexelRadiusMap[],
	GatheredLightSample OutLightmapData[]
);

//To surface element
struct SurfelData;
GPULIGHTMASSKERNEL_API void RasterizeModelToSurfel(
	const int GridElementSize,
	const int NumVertices,
	const int NumTriangles,
	const float3 VertexLocalPositionBuffer[],
	const float2 VertexTextureUVBuffer[],
	const int3 TriangleIndexBuffer[],
	const int TriangleTextureMappingIndex[],
	const float3 BBox[],
	int OutNumberSurfel[],
	SurfelData *OutSurfelData
);

}
