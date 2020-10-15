#include <memory>
#include <vector>
#include <algorithm>
#include <time.h>
#include <fstream>
#include <unordered_map>

#define GPULIGHTMASSKERNEL_LIB

#include <cuda_runtime.h>
#include "helper_math.h"
#include "GPULightmassKernel.h"

#include "rt/rtDebugFunc.h"
#include "HostFunc.h"

#include "Radiosity.h"
#include "ProgressReport.h"
#include "BVH/EmbreeBVHBuilder.h"
#include "HashCode.h"

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
	const float3* VertexData,
	const float3* VertexNormal,
	const float2* UVs,
	const int* TriangleIndex,
	const float3* Bbox,
	const int NumVertices,
	const int NumTriangles,
	const int GridElementSize,
	const Mat4f* ViewMat
);

__host__ void rtBindRasterizeBufferData(
	const GPULightmass::SurfelData* XZPlane
);

__host__ void rtCalculateDirectLighting();

__host__ void rtSetGlobalSamplingParameters(
	float FireflyClampingThreshold
);

__host__ void rtRasterizeModel(const int NumVertices, const int NumTriangles);

__host__ void rtBindSurfelLinkData(
	const int MaxLinkNodeCount,
	const GPULightmass::LinkListData* LinkBuffer,
	const int* LastNodeIdxBuffer
);

__host__ void rtBindSurfelIndirectedLightingData(
	const GPULightmass::SurfelData *SurfelData,
	const int CalSurfelsNum,
	const int GridElementSize
);

__host__ void rtSurfelDirectLighting(const int SurfelNum);

__host__ void rtBindSurfelDirLightData(
	const GPULightmass::SurfelDirLightingData* SurfelDirData
);

__host__ void rtBindSurfelIndirectedLightingDirData(
	const Mat4f* ViewMat,
	const float3* bbox,
	const int* LastLinkData,
	const int* LinkBufferData
);

__host__ void rtSurfelIndirectedLighting(const int SurfelNum);

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

GPULIGHTMASSKERNEL_API bool CreateCache(const int NumVertices, const int NumTriangles, const float3 VertexLocalPositionBuffer[], const int3 TriangleIndexBuffer[], const float3 BBox[])
{
	std::ofstream ofs;
	ofs.open("./model_cache.bin", std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);
	ofs << NumVertices;
	ofs.write((const char*)VertexLocalPositionBuffer, sizeof(float3) * NumVertices);
	ofs << NumTriangles;
	ofs.write((const char*)TriangleIndexBuffer, sizeof(int3) * NumTriangles);
	ofs.write((const char*)BBox, sizeof(float3) * 2);
	ofs.flush();
	ofs.close();

	return true;
}

float3 interplate_float3(const float3 &v0, const float3 &v1, const float3 &v2, const float3 &coeff)
{
	return make_float3(
		dot(make_float3(v0.x, v1.x, v2.x), coeff),
		dot(make_float3(v0.y, v1.y, v2.y), coeff),
		dot(make_float3(v0.z, v1.z, v2.z), coeff)
	);
}

//void add_surfel_data_output(const float3 VertexLocalPositionBuffer[], const float3 VertexLocalNormalBuffer[], const int3 TriangleIndexBuffer[],
//	const 

void GenerateSurfelDirectional(const Mat4f &CamMat, const int GridElementSize, const int NumVertices, const int NumTriangles,
	const float3 VertexLocalPositionBuffer[], const float3 VertexLocalNormalBuffer[], const float2 VertexTextureUVBuffer[], const int3 TriangleIndexBuffer[], const int TriangleTextureMappingIndex[], const float3 BBox[],
	int *OutNumberSurfel, GPULightmass::SurfelData **OutSurfelData)
{
	float3 *cudaLocalPos;
	float3 *cudaNormal;
	float2 *cudaUVs;
	int *cudaTriangleIndex;
	float3 *cudaBBox; //MIN - MAX	
	cudaCheck(cudaMalloc((void**)&cudaLocalPos, NumVertices * sizeof(float3)));
	cudaCheck(cudaMalloc((void**)&cudaNormal, NumVertices * sizeof(float3)));
	cudaCheck(cudaMalloc((void**)&cudaUVs, NumVertices * sizeof(float2)));
	cudaCheck(cudaMalloc((void**)&cudaTriangleIndex, NumTriangles * 3 * sizeof(int)));
	cudaCheck(cudaMalloc((void**)&cudaBBox, 2 * sizeof(float3)));

	//bbox to camera and ortho projection to determine plane size	
	Vec4f camera_base_bbox[2];
	camera_base_bbox[0] = CamMat * Vec4f(BBox[0].x, BBox[0].y, BBox[0].z, 1.0f);
	camera_base_bbox[1] = CamMat * Vec4f(BBox[1].x, BBox[1].y, BBox[1].z, 1.0f);
	float3 camera_base_max_box = make_float3(
		std::max(camera_base_bbox[0].x, camera_base_bbox[1].x),
		std::max(camera_base_bbox[0].y, camera_base_bbox[1].y),
		std::max(camera_base_bbox[0].z, camera_base_bbox[1].z));
	float3 camera_base_min_box = make_float3(
		std::min(camera_base_bbox[0].x, camera_base_bbox[1].x),
		std::min(camera_base_bbox[0].y, camera_base_bbox[1].y),
		std::min(camera_base_bbox[0].z, camera_base_bbox[1].z));
	//Mat4f ortho_m;

	float3 maxBBox[2];
	maxBBox[0] = make_float3(
		std::floor(camera_base_min_box.x / GridElementSize),
		std::floor(camera_base_min_box.y / GridElementSize),
		std::floor(camera_base_min_box.z / GridElementSize));
	maxBBox[1] = make_float3(
		std::ceil(camera_base_max_box.x / GridElementSize),
		std::ceil(camera_base_max_box.y / GridElementSize),
		std::ceil(camera_base_max_box.z / GridElementSize));

	cudaCheck(cudaMemcpy(cudaLocalPos, VertexLocalPositionBuffer, NumVertices * sizeof(float3), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(cudaNormal, VertexLocalNormalBuffer, NumVertices * sizeof(float3), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(cudaUVs, VertexTextureUVBuffer, NumVertices * sizeof(float2), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(cudaTriangleIndex, TriangleIndexBuffer, NumTriangles * 3 * sizeof(int), cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(cudaBBox, maxBBox, 2 * sizeof(float3), cudaMemcpyHostToDevice));	

	//GPULightmass::SurfelData* cudaXZPlaneBuffer; // map to camera
	int XZNumBufferSize = (int(maxBBox[1].x) - int(maxBBox[0].x)) * (int(maxBBox[1].z) - int(maxBBox[0].z));
	//cudaCheck(cudaMalloc(&cudaXZPlaneBuffer, XZNumBufferSize * sizeof(GPULightmass::SurfelData)));
	//cudaCheck(cudaMemset(cudaXZPlaneBuffer, 0, XZNumBufferSize * sizeof(GPULightmass::SurfelData)));

	int LinkBufferSize = XZNumBufferSize * ((int)maxBBox[1].y - (int)maxBBox[0].y);
	GPULightmass::LinkListData* pLinkBuffer = new LinkListData[LinkBufferSize];

	//init
	int* pLastIdxBuffer = new int[XZNumBufferSize];
	for (int i = 0; i < XZNumBufferSize; ++i)
	{
		pLastIdxBuffer[i] = -1;
	}

	GPULightmass::LinkListData* cudaLinkBuffer;
	int *cudaLastIdxBuffer;
	cudaCheck(cudaMalloc(&cudaLinkBuffer, sizeof(GPULightmass::LinkListData) * LinkBufferSize));
	cudaCheck(cudaMalloc(&cudaLastIdxBuffer, sizeof(int) * XZNumBufferSize));
	cudaCheck(cudaMemcpy(cudaLinkBuffer, pLinkBuffer, sizeof(GPULightmass::LinkListData) * LinkBufferSize, cudaMemcpyHostToDevice));
	cudaCheck(cudaMemcpy(cudaLastIdxBuffer, pLastIdxBuffer, sizeof(int) * XZNumBufferSize, cudaMemcpyHostToDevice));
	rtBindSurfelLinkData(LinkBufferSize, cudaLinkBuffer, cudaLastIdxBuffer);

	Mat4f* cudaViewMat;
	cudaCheck(cudaMalloc(&cudaViewMat, sizeof(Mat4f)));
	cudaCheck(cudaMemcpy(cudaViewMat, &CamMat, sizeof(Mat4f), cudaMemcpyHostToDevice));
	rtBindRasterizeData(cudaLocalPos, cudaNormal, cudaUVs, cudaTriangleIndex, cudaBBox, NumVertices, NumTriangles, GridElementSize, cudaViewMat);

	rtRasterizeModel(NumVertices, NumTriangles);

	// 可以组织数据了	
	cudaCheck(cudaMemcpy(pLastIdxBuffer, cudaLastIdxBuffer, sizeof(int) * XZNumBufferSize, cudaMemcpyDeviceToHost));
	cudaCheck(cudaMemcpy(pLinkBuffer, cudaLinkBuffer, sizeof(GPULightmass::LinkListData) * LinkBufferSize, cudaMemcpyDeviceToHost));
	std::vector<GPULightmass::SurfelData> outSurfelDataVec;
	outSurfelDataVec.reserve(XZNumBufferSize);
	for (int i = 0; i < XZNumBufferSize; ++i)
	{
		int curr_idx = pLastIdxBuffer[i];

		while (curr_idx != -1)
		{
			const GPULightmass::LinkListData &link_surf_data = pLinkBuffer[curr_idx];

			const SurfelLinkData &compress_link_data = link_surf_data.data;
			const float3 coeffi = make_float3(compress_link_data.uvw, 1.0f - compress_link_data.uvw.x - compress_link_data.uvw.y);

			int3 idxs = TriangleIndexBuffer[compress_link_data.triangle_index];
			float3 local_pos0 = VertexLocalPositionBuffer[idxs.x];
			float3 local_pos1 = VertexLocalPositionBuffer[idxs.y];
			float3 local_pos2 = VertexLocalPositionBuffer[idxs.z];
			float3 out_interplate_pos = interplate_float3(local_pos0, local_pos1, local_pos2, coeffi);

			float3 local_normal0 = VertexLocalNormalBuffer[idxs.x];
			float3 local_normal1 = VertexLocalNormalBuffer[idxs.y];
			float3 local_normal2 = VertexLocalNormalBuffer[idxs.z];
			float3 out_interplate_normal = interplate_float3(local_normal0, local_normal1, local_normal2, coeffi);

			GPULightmass::SurfelData out_surfel_data;
			out_surfel_data.pos = make_float4(out_interplate_pos, 1.0f);
			out_surfel_data.normal = make_float4(out_interplate_normal, 1.0f);
			out_surfel_data.diff_alpha = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
			outSurfelDataVec.push_back(out_surfel_data);

			curr_idx = link_surf_data.prev_index;
		}
	}

	//copy mem to host
	int output_surfel_data_count = outSurfelDataVec.size();
	(*OutSurfelData) = new GPULightmass::SurfelData[output_surfel_data_count];
	memcpy(*OutSurfelData, &outSurfelDataVec[0], sizeof(GPULightmass::SurfelData) * output_surfel_data_count);
	*OutNumberSurfel = output_surfel_data_count;

	//cudaCheck(cudaFree(cudaXZPlaneBuffer));

	delete[] pLinkBuffer;
	delete[] pLastIdxBuffer;
	cudaCheck(cudaFree(cudaLinkBuffer));
	cudaCheck(cudaFree(cudaLastIdxBuffer));

	cudaCheck(cudaFree(cudaUVs));
	cudaCheck(cudaFree(cudaTriangleIndex));
	cudaCheck(cudaFree(cudaBBox));
	cudaCheck(cudaFree(cudaNormal));
	cudaCheck(cudaFree(cudaLocalPos));
	cudaCheck(cudaFree(cudaViewMat));
}

GPULIGHTMASSKERNEL_API void RasterizeModelToSurfel(const int GridElementSize, const int NumVertices, const int NumTriangles, 
	const float3 VertexLocalPositionBuffer[], const float3 VertexLocalNormalBuffer[], const float2 VertexTextureUVBuffer[], const int3 TriangleIndexBuffer[], const int TriangleTextureMappingIndex[], const float3 BBox[],
	int OutNumberSurfel[], GPULightmass::SurfelData *OutSurfelData)
{
	CreateCache(NumVertices, NumTriangles, VertexLocalPositionBuffer, TriangleIndexBuffer, BBox);
	
	std::unordered_map<int, GPULightmass::SurfelData> compress_data_hash;
	std::unordered_map<int, GPULightmass::SurfelData>::iterator compress_data_hash_iter;

	
	Mat4f camera_up_m;
	camera_up_m.cameraMatrix(Vec3f(0.0f, 0.0f, -1.0f), Vec3f(0.0f, 0.0f, 0.0f));
	GPULightmass::SurfelData *pUpCameraRasData = NULL;
	int UpCameraCount = 0;
	GenerateSurfelDirectional(camera_up_m, GridElementSize, NumVertices, NumTriangles, VertexLocalPositionBuffer, VertexLocalNormalBuffer, VertexTextureUVBuffer,
		TriangleIndexBuffer, TriangleTextureMappingIndex, BBox, &UpCameraCount, &pUpCameraRasData);
	
	for (int i = 0; i < UpCameraCount; ++i)
	{
		int hash_key = F3ToIntKey(make_float3(pUpCameraRasData[i].pos));
		compress_data_hash_iter = compress_data_hash.find(hash_key);
		if (compress_data_hash_iter == compress_data_hash.end())
		{
			compress_data_hash.insert(std::pair<int, GPULightmass::SurfelData>(hash_key, pUpCameraRasData[i]));
		}		
	}
	

	Mat4f camera_up_l;
	camera_up_l.cameraMatrix(Vec3f(1.0f, 0.0f, 0.0f), Vec3f(0.0f, 0.0f, 0.0f));
	GPULightmass::SurfelData *pLeftCameraRasData = NULL;
	int LeftCameraCount = 0;
	GenerateSurfelDirectional(camera_up_l, GridElementSize, NumVertices, NumTriangles, VertexLocalPositionBuffer, VertexLocalNormalBuffer, VertexTextureUVBuffer,
		TriangleIndexBuffer, TriangleTextureMappingIndex, BBox, &LeftCameraCount, &pLeftCameraRasData);

	for (int i = 0; i < LeftCameraCount; ++i)
	{
		int hash_key = F3ToIntKey(make_float3(pLeftCameraRasData[i].pos));
		compress_data_hash_iter = compress_data_hash.find(hash_key);
		if (compress_data_hash_iter == compress_data_hash.end())
		{
			compress_data_hash.insert(std::pair<int, GPULightmass::SurfelData>(hash_key, pLeftCameraRasData[i]));
		}
	}

	Mat4f camera_up_f;
	camera_up_f.cameraMatrix(Vec3f(0.0f, 1.0f, 0.0f), Vec3f(0.0f, 0.0f, 0.0f));
	GPULightmass::SurfelData *pFCameraRasData = NULL;
	int ForwardCameraCount = 0;
	GenerateSurfelDirectional(camera_up_f, GridElementSize, NumVertices, NumTriangles, VertexLocalPositionBuffer, VertexLocalNormalBuffer, VertexTextureUVBuffer,
		TriangleIndexBuffer, TriangleTextureMappingIndex, BBox, &ForwardCameraCount, &pFCameraRasData);	
	
	for (int i = 0; i < ForwardCameraCount; ++i)
	{
		int hash_key = F3ToIntKey(make_float3(pFCameraRasData[i].pos));
		compress_data_hash_iter = compress_data_hash.find(hash_key);
		if (compress_data_hash_iter == compress_data_hash.end())
		{
			compress_data_hash.insert(std::pair<int, GPULightmass::SurfelData>(hash_key, pFCameraRasData[i]));
		}
	}

	//int before_num = UpCameraCount + LeftCameraCount + ForwardCameraCount;
	OutNumberSurfel[0] = compress_data_hash.size();

	int SurfelDataElemSize = sizeof(GPULightmass::SurfelData);
	int SurfelIdx = 0;
	for (compress_data_hash_iter = compress_data_hash.begin(); compress_data_hash_iter != compress_data_hash.end(); ++compress_data_hash_iter)
	{
		OutSurfelData[SurfelIdx] = compress_data_hash_iter->second;
		SurfelIdx++;
	}

	CalculateSurfelIndirectedLighting(OutSurfelData, OutNumberSurfel[0], GridElementSize);

	delete[] pUpCameraRasData;
	delete[] pLeftCameraRasData;
	delete[] pFCameraRasData;
}

void GetBBox(const SurfelData *pSurfelData, const int SurfelNum, float3 OutBBox[2])
{
	OutBBox[0] = make_float3(std::numeric_limits<float>::max());
	OutBBox[1] = make_float3(std::numeric_limits<float>::min());

	for (int i = 0; i < SurfelNum; ++i)
	{
		OutBBox[0].x = std::min(OutBBox[0].x, pSurfelData[i].pos.x);
		OutBBox[0].y = std::min(OutBBox[0].y, pSurfelData[i].pos.y);
		OutBBox[0].z = std::min(OutBBox[0].z, pSurfelData[i].pos.z);

		OutBBox[1].x = std::max(OutBBox[1].x, pSurfelData[i].pos.x);
		OutBBox[1].y = std::max(OutBBox[1].y, pSurfelData[i].pos.y);
		OutBBox[1].z = std::max(OutBBox[1].z, pSurfelData[i].pos.z);
	}
}

GPULIGHTMASSKERNEL_API void CalculateSurfelIndirectedLighting(SurfelData *InOutSurfelData, const int SurfelNum, const int GridElementSize)
{
	SurfelData *cudaSurfelData;
	cudaCheck(cudaMalloc(&cudaSurfelData, SurfelNum * sizeof(SurfelData)));
	cudaCheck(cudaMemcpy(cudaSurfelData, InOutSurfelData, SurfelNum * sizeof(SurfelData), cudaMemcpyHostToDevice));	

	float3 BBox[2];
	GetBBox(InOutSurfelData, SurfelNum, BBox);

	rtBindSurfelIndirectedLightingData(cudaSurfelData, SurfelNum, GridElementSize);

	//First pass, direct lighting
	rtSurfelDirectLighting(SurfelNum);

	//Create Link buffer and radiance buffer
	SurfelDirLightingData SurfelDirLightingBuffer;
	SurfelDirLightingBuffer.SurfelNum = SurfelNum;
	cudaCheck(cudaMalloc(&SurfelDirLightingBuffer.radiance[0], sizeof(float4) * SurfelNum));
	cudaCheck(cudaMemset(SurfelDirLightingBuffer.radiance[0], 0, sizeof(float4) * SurfelNum));
	cudaCheck(cudaMalloc(&SurfelDirLightingBuffer.radiance[1], sizeof(float4) * SurfelNum));
	cudaCheck(cudaMemset(SurfelDirLightingBuffer.radiance[1], 0, sizeof(float4) * SurfelNum));
	//cudaCheck(cudaMalloc(&SurfelDirLightingBuffer.LinkIndexBuf, sizeof(int) * SurfelNum));
	int* pInitLastIdxBuffer = new int[SurfelNum];
	for (int i = 0; i < SurfelNum; ++i)
	{
		pInitLastIdxBuffer[i] = -1;
	}
	int* cudaLinkBufferData;
	cudaCheck(cudaMalloc(&cudaLinkBufferData, sizeof(int) * SurfelNum));		
	
	SurfelDirLightingData *cudaSurfelDirLightingBuffer;
	cudaCheck(cudaMalloc(&cudaSurfelDirLightingBuffer, sizeof(SurfelDirLightingData)));
	cudaCheck(cudaMemcpy(cudaSurfelDirLightingBuffer, &SurfelDirLightingBuffer, sizeof(SurfelDirLightingData), cudaMemcpyHostToDevice));
	rtBindSurfelDirLightData(cudaSurfelDirLightingBuffer);

	Mat4f *cudaViewMat;
	cudaCheck(cudaMalloc(&cudaViewMat, sizeof(Mat4f)));
	float3 *cudaBBox;
	cudaCheck(cudaMalloc(&cudaBBox, sizeof(float3) * 2));

	//semi-spherical sampling
	const int PassNum = 5;
	const int NumThetaStep = 4;
	const int NumPhiStep = 4;
	const float ThetaStep = 3.1415f / 2.0f / (NumThetaStep + 2);
	const float PhiStep = 3.1415f * 2.0f / NumPhiStep;

	for (int i = 0; i < NumThetaStep; ++i)
	{
		for (int j = 0; j < NumPhiStep; ++j)
		{
			float theta = ThetaStep * (i + 1);
			float phi = PhiStep * j;

			float x = std::sin(theta) * std::cos(phi);
			float y = std::sin(theta) * std::sin(phi);
			float z = std::cos(theta);

			Mat4f dir;
			dir.cameraMatrix(Vec3f(-x, -y, -z), Vec3f(0.0f, 0.0f, 0.0f));
			printf("dir = %f, %f, %f\n", -x, -y, -z);

			//transform bbox to camera
			float3 BBoxBaseOnCam[2];
			Vec4f tmin_b = dir * Vec4f(BBox[0].x, BBox[0].y, BBox[0].z, 1.0f);
			Vec4f tmax_b = dir * Vec4f(BBox[1].x, BBox[1].y, BBox[1].z, 1.0f);
			float3 min_b = make_float3(
				std::min(tmin_b.x, tmax_b.x),
				std::min(tmin_b.y, tmax_b.y),
				std::min(tmin_b.z, tmax_b.z));
			float3 max_b = make_float3(
				std::max(tmin_b.x, tmax_b.x),
				std::max(tmin_b.y, tmax_b.y),
				std::max(tmin_b.z, tmax_b.z));

			BBoxBaseOnCam[0] = min_b / GridElementSize;
			BBoxBaseOnCam[1] = max_b / GridElementSize;

			cudaCheck(cudaMemcpy(cudaViewMat, &dir, sizeof(Mat4f), cudaMemcpyHostToDevice));
			cudaCheck(cudaMemcpy(cudaBBox, BBoxBaseOnCam, sizeof(float3) * 2, cudaMemcpyHostToDevice));

			int XZNumBufferSize = (int(BBoxBaseOnCam[1].x) - int(BBoxBaseOnCam[0].x)) * (int(BBoxBaseOnCam[1].z) - int(BBoxBaseOnCam[0].z));
			int* SurfelLightingLastLink = new int[XZNumBufferSize];
			for (int nl = 0; nl < XZNumBufferSize; ++nl)
			{
				SurfelLightingLastLink[nl] = -1;
			}
			int* cudaSurfelLightingLastLink;
			cudaCheck(cudaMalloc(&cudaSurfelLightingLastLink, sizeof(int) * XZNumBufferSize));
			cudaCheck(cudaMemcpy(cudaSurfelLightingLastLink, SurfelLightingLastLink, sizeof(int) * XZNumBufferSize, cudaMemcpyHostToDevice));

			cudaCheck(cudaMemcpy(cudaLinkBufferData, pInitLastIdxBuffer, sizeof(int) * SurfelNum, cudaMemcpyHostToDevice));

			rtBindSurfelIndirectedLightingDirData(cudaViewMat, cudaBBox, cudaSurfelLightingLastLink, cudaLinkBufferData);

			rtSurfelIndirectedLighting(SurfelNum);

			cudaCheck(cudaFree(cudaSurfelLightingLastLink));
			delete[] SurfelLightingLastLink;
		}
	}



	cudaCheck(cudaMemcpy(InOutSurfelData, cudaSurfelData, SurfelNum * sizeof(SurfelData), cudaMemcpyDeviceToHost));

	delete[] pInitLastIdxBuffer;
	cudaCheck(cudaFree(cudaSurfelData));
	cudaCheck(cudaFree(SurfelDirLightingBuffer.radiance[0]));
	cudaCheck(cudaFree(SurfelDirLightingBuffer.radiance[1]));
	cudaCheck(cudaFree(cudaLinkBufferData));
	cudaCheck(cudaFree(cudaViewMat));
	cudaCheck(cudaFree(cudaBBox));
}

}
