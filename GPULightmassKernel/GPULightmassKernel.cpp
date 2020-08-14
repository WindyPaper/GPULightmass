#include <memory>
#include <vector>
#include <algorithm>
#include <time.h>

#define GPULIGHTMASSKERNEL_LIB

#include <cuda_runtime.h>
#include "helper_math.h"
#include "GPULightmassKernel.h"

#include "rt/rtDebugFunc.h"
#include "HostFunc.h"

#include "Radiosity.h"
#include "ProgressReport.h"
#include "BVH/EmbreeBVHBuilder.h"

//#include "rt/rtDirectLighting.h"
#include "SurfelData.h"

__host__ void rtBindMaskedCollisionMaps(
	int NumMaps,
	cudaTextureObject_t* InMaps
);

__host__ void rtLaunchVolumetric(
	const int BrickSize,
	const float3 WorldBrickMin,
	const float3 WorldChildCellSize,
	GPULightmass::VolumetricLightSample InOutVolumetricBrickUpperSamples[],
	GPULightmass::VolumetricLightSample InOutVolumetricBrickLowerSamples[]
);

__host__ void rtLaunchVolumeSamples(
	const int NumSamples,
	const float3 WorldPositions[],
	GPULightmass::VolumetricLightSample InOutVolumetricBrickUpperSamples[],
	GPULightmass::VolumetricLightSample InOutVolumetricBrickLowerSamples[]
);

__host__ void rtBindPunctualLights(
	const int InNumDirectionalLights,
	const GPULightmass::DirectionalLight* InDirectionalLights,
	const int InNumPointLights,
	const GPULightmass::PointLight* InPointLights,
	const int InNumSpotLights,
	const GPULightmass::SpotLight* InSpotLights
);

__host__ void rtBindSampleData(
	const float4* SampleWorldPositions,
	const float4* SampleWorldNormals,
	const float* TexelRadius,
	GPULightmass::GatheredLightSample* InOutLightmapData,
	const int InSizeX,
	const int InSizeY
);

__host__ void rtBindRasterizeData(
	const float3 *VertexData,
	const float2 *UVs,
	const int *TriangleIndex,
	const float3 *Bbox,
	const int NumVertices,
	const int NumTriangles,
	const int GridElementSize
);

__host__ void rtCalculateDirectLighting();

__host__ void rtSetGlobalSamplingParameters(
	float FireflyClampingThreshold
);

__host__ void rtRasterizeModel(const int NumVertices, const int NumTriangles);

void WriteHDR(std::string fileName, const float4* buffer, int Width, int Height);

#include "StringUtils.h"

namespace GPULightmass
{

GPULightmassLogHandler GLogHandler = nullptr;

void LOG(const char* format, ...)
{
	if (GLogHandler == nullptr) return;
	char dest[1024 * 16];
	va_list argptr;
	va_start(argptr, format);
	vsprintf(dest, format, argptr);
	GLogHandler((L"GPULightmass Kernel: " + RStringUtils::Widen(dest)).c_str());
	OutputDebugStringW((L"GPULightmass Kernel: " + RStringUtils::Widen(dest)).c_str());
	va_end(argptr);
}

GPULIGHTMASSKERNEL_API void SetLogHandler(GPULightmassLogHandler LogHandler)
{
	GLogHandler = LogHandler;
	StartProgressReporter();
}

class ScopedTimer
{
private:
	clock_t startTime;
	const char* timerName;
public:
	ScopedTimer(const char* _timerName)
		:timerName(_timerName), startTime(clock())
	{
	}

	~ScopedTimer()
	{
		LOG("%s finished, %dMS", timerName, clock() - startTime);
	}
};

#define SCOPED_TIMER(NAME) ScopedTimer timer##__COUNTER__(NAME)

#define DEBUG_ENABLE_BVH_CACHING

GPULIGHTMASSKERNEL_API void ImportAggregateMesh(
	const int NumVertices,
	const int NumTriangles,
	const float3 VertexWorldPositionBuffer[],
	const float2 VertexTextureUVBuffer[],
	const float2 VertexLightmapUVBuffer[],
	const int3 TriangleIndexBuffer[],
	const int TriangleMaterialIndex[],
	const int TriangleTextureMappingIndex[]
)
{
	LOG("Importing mesh: %d vertices, %d triangles", NumVertices, NumTriangles);

	BVH2Node* root;

	{
		EmbreeBVHBuilder builder(NumVertices, NumTriangles, VertexWorldPositionBuffer, TriangleIndexBuffer);
		{
			SCOPED_TIMER("Embree SBVH Construction");
			root = builder.BuildBVH2();
		}

		std::vector<float4> nodeData;
		std::vector<float4> woopifiedTriangles;
		std::vector<int> triangleIndices;

		{
			SCOPED_TIMER("Convert to CudaBVH");
			builder.ConvertToCUDABVH2(root, TriangleMaterialIndex, nodeData, woopifiedTriangles, triangleIndices);
		}

		{
			SCOPED_TIMER("Bind CudaBVH");
			BindBVHData(nodeData, woopifiedTriangles, triangleIndices);
		}
	}
	
	std::unique_ptr<float2[]> verifiedVertexLightmapUVBuffer{ new float2[NumVertices]() };

	for (int vertex = 0; vertex < NumVertices; vertex++)
	{
		verifiedVertexLightmapUVBuffer[vertex] = clamp(VertexLightmapUVBuffer[vertex], make_float2(0.0f), make_float2(1.0f));
	}

	BindParameterizationData(NumVertices, NumTriangles, VertexTextureUVBuffer, verifiedVertexLightmapUVBuffer.get(), TriangleIndexBuffer, TriangleTextureMappingIndex);
}

GPULIGHTMASSKERNEL_API void ImportMaterialMaps(
	const int NumMaterials,
	const int SizeXY,
	float** MapData
)
{
	std::vector<cudaTextureObject_t> textureMaps;
	textureMaps.resize(NumMaterials);

	for (int i = 0; i < NumMaterials; i++)
	{
		cudaArray_t array;
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
		cudaCheck(cudaMallocArray(&array, &channelDesc, SizeXY, SizeXY));
		cudaCheck(cudaMemcpyToArray(array, 0, 0, MapData[i], SizeXY * SizeXY * sizeof(float), cudaMemcpyHostToDevice));

		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = array;

		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.addressMode[0] = cudaAddressModeWrap;
		texDesc.addressMode[1] = cudaAddressModeWrap;
		texDesc.filterMode = cudaFilterModeLinear;
		texDesc.readMode = cudaReadModeElementType;
		texDesc.normalizedCoords = 1;

		cudaCheck(cudaCreateTextureObject(&textureMaps[i], &resDesc, &texDesc, NULL));
	}

	LOG("%d masked collision maps imported", NumMaterials);

	rtBindMaskedCollisionMaps(textureMaps.size(), textureMaps.data());
}

GPULIGHTMASSKERNEL_API void ImportSurfaceCache(
	const int ID,
	const int SurfaceCacheSizeX,
	const int SurfaceCacheSizeY,
	const float4 WorldPositionMap[],
	const float4 WorldNormalMap[],
	const float4 ReflectanceMap[],
	const float4 EmissiveMap[]
)
{
	CreateSurfaceCacheSampleData(ID, SurfaceCacheSizeX, SurfaceCacheSizeY, WorldPositionMap, WorldNormalMap, ReflectanceMap, EmissiveMap);
}

GPULIGHTMASSKERNEL_API void RunRadiosity(int NumBounces, int NumSamplesFirstPass)
{
	LaunchRadiosityLoop(NumBounces, NumSamplesFirstPass);
}

GPULIGHTMASSKERNEL_API void SetTotalTexelsForProgressReport(
	const size_t NumTotalTexels
)
{
	SetTotalTexels(NumTotalTexels);
}

GPULIGHTMASSKERNEL_API void SetGlobalSamplingParameters(
	float FireflyClampingThreshold
)
{
	rtSetGlobalSamplingParameters(FireflyClampingThreshold);
}

GPULIGHTMASSKERNEL_API void CalculateIndirectLightingTextureMapping(
	const size_t NumTexelsInCurrentBatch,
	const int CachedSizeX,
	const int CachedSizeY,
	const int NumSamples,
	const float4 WorldPositionMap[],
	const float4 WorldNormalMap[],
	const float TexelRadiusMap[],
	GatheredLightSample OutLightmapData[]
)
{
	CalculateLighting(WorldPositionMap, WorldNormalMap, TexelRadiusMap, OutLightmapData, CachedSizeX, CachedSizeY, NumSamples);
	ReportCurrentFinishedTexels(NumTexelsInCurrentBatch);
}

struct LuminanceSortingEntry
{
	float Luminance;
	int LinearIndex;
};

GPULIGHTMASSKERNEL_API void ImportSkyLightCubemap(
	const int NumThetaSteps,
	const int NumPhiSteps,
	float4 UpperHemisphereCubemap[],
	float4 LowerHemisphereCubemap[]
)
{
	std::vector<float4> UpperHemisphereCubemapWithBoundaryCheck(UpperHemisphereCubemap, UpperHemisphereCubemap + NumThetaSteps * NumPhiSteps);
	std::vector<float4> LowerHemisphereCubemapWithBoundaryCheck(LowerHemisphereCubemap, LowerHemisphereCubemap + NumThetaSteps * NumPhiSteps);
	std::vector<LuminanceSortingEntry> UpperHemisphereSortingEntries;
	std::vector<LuminanceSortingEntry> LowerHemisphereSortingEntries;

	for (int i = 0; i < NumThetaSteps * NumPhiSteps; i++)
	{
		UpperHemisphereSortingEntries.push_back(LuminanceSortingEntry { getLuminance(make_float3(UpperHemisphereCubemapWithBoundaryCheck.at(i))), i });
		LowerHemisphereSortingEntries.push_back(LuminanceSortingEntry { getLuminance(make_float3(LowerHemisphereCubemapWithBoundaryCheck.at(i))), i });
	}

	std::sort(UpperHemisphereSortingEntries.begin(), UpperHemisphereSortingEntries.end(), [](const LuminanceSortingEntry& a, const LuminanceSortingEntry& b) { return a.Luminance > b.Luminance; });
	std::sort(LowerHemisphereSortingEntries.begin(), LowerHemisphereSortingEntries.end(), [](const LuminanceSortingEntry& a, const LuminanceSortingEntry& b) { return a.Luminance > b.Luminance; });

	std::vector<int> UpperHemisphereImportantDirections;
	std::vector<int> LowerHemisphereImportantDirections;
	std::vector<float4> UpperHemisphereImportantColor;
	std::vector<float4> LowerHemisphereImportantColor;

	for (int i = 0; i < 16; i++)
	{
		UpperHemisphereImportantDirections.push_back(UpperHemisphereSortingEntries[i].LinearIndex);
		UpperHemisphereImportantColor.push_back(UpperHemisphereCubemapWithBoundaryCheck.at(UpperHemisphereSortingEntries[i].LinearIndex));

		LowerHemisphereImportantDirections.push_back(LowerHemisphereSortingEntries[i].LinearIndex);
		LowerHemisphereImportantColor.push_back(LowerHemisphereCubemapWithBoundaryCheck.at(LowerHemisphereSortingEntries[i].LinearIndex));
	}

#if 0
	for (int i = 0; i < 16; i++)
	{
		UpperHemisphereCubemapWithBoundaryCheck[UpperHemisphereSortingEntries[i].LinearIndex] = make_float4(0);
		LowerHemisphereCubemapWithBoundaryCheck[LowerHemisphereSortingEntries[i].LinearIndex] = make_float4(0);
	}
#endif

	BindSkyLightCubemap(NumThetaSteps, NumPhiSteps, UpperHemisphereCubemapWithBoundaryCheck.data(), LowerHemisphereCubemapWithBoundaryCheck.data(), UpperHemisphereImportantDirections.data(), LowerHemisphereImportantDirections.data(), UpperHemisphereImportantColor.data(), LowerHemisphereImportantColor.data());
}

GPULIGHTMASSKERNEL_API void ImportPunctualLights(
	const int NumDirectionalLights,
	const DirectionalLight DirectionalLights[],
	const int NumPointLights,
	const PointLight PointLights[],
	const int NumSpotLights,
	const SpotLight SpotLights[]
)
{
	DirectionalLight* cudaDirectionalLights;
	PointLight* cudaPointLights;
	SpotLight* cudaSpotLights;

	cudaCheck(cudaMalloc(&cudaDirectionalLights, sizeof(DirectionalLight) * NumDirectionalLights));
	cudaCheck(cudaMalloc(&cudaPointLights, sizeof(PointLight) * NumPointLights));
	cudaCheck(cudaMalloc(&cudaSpotLights, sizeof(SpotLight) * NumSpotLights));

	cudaCheck(cudaMemcpy(cudaDirectionalLights, DirectionalLights, sizeof(DirectionalLight) * NumDirectionalLights, cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(cudaPointLights, PointLights, sizeof(PointLight) * NumPointLights, cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(cudaSpotLights, SpotLights, sizeof(SpotLight) * NumSpotLights, cudaMemcpyHostToDevice));

	rtBindPunctualLights(NumDirectionalLights, cudaDirectionalLights, NumPointLights, cudaPointLights, NumSpotLights, cudaSpotLights);

	LOG("%d directional lights, %d point lights and %d spot lights imported", NumDirectionalLights, NumPointLights, NumSpotLights);
}

GPULIGHTMASSKERNEL_API void PreallocRadiositySurfaceCachePointers(const int Num)
{
	PreallocSurfaceCachePointers(Num);
}

GPULIGHTMASSKERNEL_API void CalculateVolumetricLightmapBrickSamples(
	const int BrickSize,
	const float3 WorldBrickMin,
	const float3 WorldChildCellSize,
	VolumetricLightSample InOutVolumetricBrickUpperSamples[],
	VolumetricLightSample InOutVolumetricBrickLowerSamples[]
)
{
	VolumetricLightSample* cudaVolumetricBrickUpperSamples;
	VolumetricLightSample* cudaVolumetricBrickLowerSamples;

	cudaCheck(cudaMalloc(&cudaVolumetricBrickUpperSamples, BrickSize * BrickSize * BrickSize * sizeof(VolumetricLightSample)));
	cudaCheck(cudaMalloc(&cudaVolumetricBrickLowerSamples, BrickSize * BrickSize * BrickSize * sizeof(VolumetricLightSample)));

	rtLaunchVolumetric(BrickSize, WorldBrickMin, WorldChildCellSize, cudaVolumetricBrickUpperSamples, cudaVolumetricBrickLowerSamples);

	cudaMemcpy(InOutVolumetricBrickUpperSamples, cudaVolumetricBrickUpperSamples, BrickSize * BrickSize * BrickSize * sizeof(VolumetricLightSample), cudaMemcpyDeviceToHost);
	cudaMemcpy(InOutVolumetricBrickLowerSamples, cudaVolumetricBrickLowerSamples, BrickSize * BrickSize * BrickSize * sizeof(VolumetricLightSample), cudaMemcpyDeviceToHost);

	cudaFree(cudaVolumetricBrickUpperSamples);
	cudaFree(cudaVolumetricBrickLowerSamples);
}

GPULIGHTMASSKERNEL_API void CalculateVolumeSampleList(
	const int NumSamples,
	const float3 WorldPositions[],
	VolumetricLightSample InOutVolumetricBrickUpperSamples[],
	VolumetricLightSample InOutVolumetricBrickLowerSamples[]
)
{
	float3* cudaWorldPositions;

	cudaCheck(cudaMalloc(&cudaWorldPositions, NumSamples * sizeof(float3)));
	cudaMemcpy(cudaWorldPositions, WorldPositions, NumSamples * sizeof(float3), cudaMemcpyHostToDevice);

	VolumetricLightSample* cudaVolumetricBrickUpperSamples;
	VolumetricLightSample* cudaVolumetricBrickLowerSamples;

	cudaCheck(cudaMalloc(&cudaVolumetricBrickUpperSamples, NumSamples * sizeof(VolumetricLightSample)));
	cudaCheck(cudaMalloc(&cudaVolumetricBrickLowerSamples, NumSamples * sizeof(VolumetricLightSample)));

	rtLaunchVolumeSamples(NumSamples, cudaWorldPositions, cudaVolumetricBrickUpperSamples, cudaVolumetricBrickLowerSamples);

	cudaMemcpy(InOutVolumetricBrickUpperSamples, cudaVolumetricBrickUpperSamples, NumSamples * sizeof(VolumetricLightSample), cudaMemcpyDeviceToHost);
	cudaMemcpy(InOutVolumetricBrickLowerSamples, cudaVolumetricBrickLowerSamples, NumSamples * sizeof(VolumetricLightSample), cudaMemcpyDeviceToHost);

	cudaFree(cudaWorldPositions);
	cudaFree(cudaVolumetricBrickUpperSamples);
	cudaFree(cudaVolumetricBrickLowerSamples);
}

GPULIGHTMASSKERNEL_API void ImportDirectLights(const int NumDirectionalLights, const DirectionalLight DirectionalLights[], const int NumPointLights, const PointLight PointLights[], const int NumSpotLights, const SpotLight SpotLights[])
{
	DirectionalLight *cudaDirectionalLights;
	PointLight *cudaPointLights;
	SpotLight *cudaSpotLights;

	cudaCheck(cudaMalloc(&cudaDirectionalLights, NumDirectionalLights * sizeof(DirectionalLight)));
	cudaCheck(cudaMalloc(&cudaPointLights, NumPointLights * sizeof(PointLight)));
	cudaCheck(cudaMalloc(&cudaSpotLights, NumSpotLights * sizeof(SpotLight)));

	cudaCheck(cudaMemcpy(cudaDirectionalLights, DirectionalLights, sizeof(DirectionalLight) * NumDirectionalLights, cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(cudaPointLights, PointLights, sizeof(PointLight) * NumPointLights, cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(cudaSpotLights, SpotLights, sizeof(SpotLight) * NumSpotLights, cudaMemcpyHostToDevice));

	rtBindPunctualLights(NumDirectionalLights, cudaDirectionalLights, NumPointLights, cudaPointLights, NumSpotLights, cudaSpotLights);

	LOG("Add Direct lights. %d directional lights, %d point lights, %d spot lights", NumDirectionalLights, NumPointLights, NumSpotLights);
}

GPULIGHTMASSKERNEL_API void CalculateDirectLightingAndShadow(const size_t NumTexelsInCurrentBatch, const int CachedSizeX, const int CachedSizeY, const int NumSamples, const float4 WorldPositionMap[], const float4 WorldNormalMap[], const float TexelRadiusMap[], GatheredLightSample OutLightmapData[])
{
	float4* cudaSampleWorldPositions;
	float4* cudaSampleWorldNormals;
	float*	cudaTexelRadius;
	GPULightmass::GatheredLightSample* cudaOutLightmapData;

	cudaCheck(cudaMalloc((void**)&cudaSampleWorldPositions, CachedSizeX * CachedSizeY * sizeof(float4)));
	cudaCheck(cudaMemcpyAsync(cudaSampleWorldPositions, WorldPositionMap, CachedSizeX * CachedSizeY * sizeof(float4), cudaMemcpyHostToDevice));

	cudaCheck(cudaMalloc((void**)&cudaSampleWorldNormals, CachedSizeX * CachedSizeY * sizeof(float4)));
	cudaCheck(cudaMemcpyAsync(cudaSampleWorldNormals, WorldNormalMap, CachedSizeX * CachedSizeY * sizeof(float4), cudaMemcpyHostToDevice));

	cudaCheck(cudaMalloc((void**)&cudaTexelRadius, CachedSizeX * CachedSizeY * sizeof(float)));
	cudaCheck(cudaMemcpyAsync(cudaTexelRadius, TexelRadiusMap, CachedSizeX * CachedSizeY * sizeof(float), cudaMemcpyHostToDevice));

	cudaCheck(cudaMalloc((void**)&cudaOutLightmapData, CachedSizeX * CachedSizeY * sizeof(GPULightmass::GatheredLightSample)));

	rtBindSampleData(cudaSampleWorldPositions, cudaSampleWorldNormals, cudaTexelRadius, cudaOutLightmapData, CachedSizeX, CachedSizeY);

	cudaCheck(cudaMemset(cudaOutLightmapData, 0, CachedSizeX * CachedSizeY * sizeof(GPULightmass::GatheredLightSample)));

	//float MRaysPerSecond = 1.0f;
	//float time = rtTimedLaunch(MRaysPerSecond, NumSamples);
	rtCalculateDirectLighting();

	cudaPostKernelLaunchCheck

	cudaCheck(cudaMemcpyAsync(OutLightmapData, cudaOutLightmapData, CachedSizeX * CachedSizeY * sizeof(GPULightmass::GatheredLightSample), cudaMemcpyDeviceToHost));

	cudaCheck(cudaFree(cudaOutLightmapData));
	cudaCheck(cudaFree(cudaSampleWorldPositions));
	cudaCheck(cudaFree(cudaSampleWorldNormals));
	cudaCheck(cudaFree(cudaTexelRadius));
}

GPULIGHTMASSKERNEL_API void CalculateAllLightingAndShadow(
	const size_t NumTexelsInCurrentBatch,
	const int CachedSizeX,
	const int CachedSizeY,
	const int NumSamples,
	const float4 WorldPositionMap[],
	const float4 WorldNormalMap[],
	const float TexelRadiusMap[],
	GatheredLightSample OutLightmapData[]
)
{
	CalculateAllBakedLighting(WorldPositionMap, WorldNormalMap, TexelRadiusMap, OutLightmapData, CachedSizeX, CachedSizeY, NumSamples);
	ReportCurrentFinishedTexels(NumTexelsInCurrentBatch);
}

GPULIGHTMASSKERNEL_API void RasterizeModelToSurfel(const int GridElementSize, const int NumVertices, const int NumTriangles, const float3 VertexLocalPositionBuffer[], const float2 VertexTextureUVBuffer[], const int3 TriangleIndexBuffer[], const int TriangleTextureMappingIndex[], const float3 BBox[], int OutNumberSurfel[], float4 OutWorldPosition[], float4 OutWorldNormal[], float4 OutAlbedoAndTransparent[])
{
	float3 *cudaLocalPos;
	float2 *cudaUVs;
	int *cudaTriangleIndex;
	float3 *cudaBBox; //MIN - MAX	
	cudaCheck(cudaMalloc((void**)&cudaLocalPos, NumVertices * sizeof(float3)));
	cudaCheck(cudaMalloc((void**)&cudaUVs, NumVertices * sizeof(float2)));
	cudaCheck(cudaMalloc((void**)&cudaTriangleIndex, NumTriangles * 3 * sizeof(int)));
	cudaCheck(cudaMalloc((void**)&cudaBBox, 2 * sizeof(float3)));
	
	float3 maxBBox[2];
	maxBBox[0] = make_float3(std::floor(BBox[0].x / GridElementSize), std::floor(BBox[0].y / GridElementSize), std::floor(BBox[0].z / GridElementSize));
	maxBBox[1] = make_float3(std::floor(BBox[1].x / GridElementSize), std::floor(BBox[1].y / GridElementSize), std::floor(BBox[1].z / GridElementSize));

	cudaCheck(cudaMemcpyAsync(cudaLocalPos, VertexLocalPositionBuffer, NumVertices * sizeof(float3), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpyAsync(cudaUVs, VertexTextureUVBuffer, NumVertices * sizeof(float2), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpyAsync(cudaTriangleIndex, TriangleIndexBuffer, NumTriangles * 3 * sizeof(int), cudaMemcpyHostToDevice));	
	cudaCheck(cudaMemcpyAsync(cudaBBox, maxBBox, 2 * sizeof(float3), cudaMemcpyHostToDevice));

	//buffer
	SurfelData *cudaYZPlaneBuffer;
	int YZNumBufferSize = ((int)BBox[1].y - (int)BBox[0].y) * ((int)BBox[1].z - (int)BBox[0].z);
	cudaCheck(cudaMalloc((void**)&cudaYZPlaneBuffer, YZNumBufferSize * sizeof(SurfelData)));
	cudaCheck(cudaMemset((void**)&cudaYZPlaneBuffer, 0, YZNumBufferSize * sizeof(SurfelData)));
	
	rtBindRasterizeData(cudaLocalPos, cudaUVs, cudaTriangleIndex, cudaBBox, NumVertices, NumTriangles, GridElementSize);


	rtRasterizeModel(NumVertices, NumTriangles);
}

}
