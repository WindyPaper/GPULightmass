#include <memory>
#include <vector>
#include <algorithm>
#include <time.h>

#define GPULIGHTMASSKERNEL_LIB

#include <cuda_runtime.h>
#include "GPULightmassKernel.h"

#include <helper_math.h>
#include "SurfelData.h"
#include <stdio.h>
#include <memory>
#include <fstream>

//only for unit test
int main()
{	
	const int vertices_num = 3;
	const int triangle_num = 1;
	float3 vertex_pos[vertices_num];
	vertex_pos[0] = make_float3(0, -50, -50);
	vertex_pos[1] = make_float3(0, 50, 50);
	vertex_pos[2] = make_float3(0, -50, 50);

	float2 uvs[vertices_num];
	uvs[0] = make_float2(0, 0);
	uvs[1] = make_float2(1, 1);
	uvs[2] = make_float2(0, 1);

	int3 index[triangle_num];
	index[0] = make_int3(0, 1, 2);

	float3 bbox[2];
	bbox[0] = make_float3(0, -50, -50);
	bbox[1] = make_float3(0, 50, 50);

	const int surfel_num = 100;
	GPULightmass::SurfelData *pSurfData = new GPULightmass::SurfelData[surfel_num];
	memset(pSurfData, 0, sizeof(GPULightmass::SurfelData) * surfel_num);
	int out_surf_num[1];
	out_surf_num[0] = surfel_num;

	GPULightmass::RasterizeModelToSurfel(10, vertices_num, triangle_num, vertex_pos, uvs, index, &(index[0].x), bbox, out_surf_num, pSurfData);

	typedef unsigned char RGB[3];
	RGB pixel[100];
	memset(pixel, 0, sizeof(pixel));

	for (int i = 0; i < out_surf_num[0]; ++i)
	{
		/*if(pSurfData[i].pos.x != 0 || pSurfData[i].pos.y != 0 || pSurfData[i].pos.z != 0)
		{
			printf("x = %f, y = %f, z = %f\n", pSurfData[i].pos.x, pSurfData[i].pos.y, pSurfData[i].pos.z);
		}*/

		pixel[i][0] = (unsigned char)((pSurfData[i].pos.x / 50 * 0.5 + 0.5) * 255);
		pixel[i][1] = (unsigned char)((pSurfData[i].pos.y / 50 * 0.5 + 0.5) * 255);
		pixel[i][2] = (unsigned char)((pSurfData[i].pos.z / 50 * 0.5 + 0.5) * 255);
	}

	const int w = 10;
	const int h = 10;
	std::ofstream ofs;
	ofs.open("./raster2d.ppm", std::ios_base::binary);
	ofs << "P6\n" << w << " " << h << "\n255\n";
	ofs.write((char*)pixel, w * h * 3);
	ofs.close();

	delete[] pSurfData;

	return 0;
}