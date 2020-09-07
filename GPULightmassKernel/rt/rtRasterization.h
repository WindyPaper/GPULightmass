#pragma once

__device__ bool pointInTriangle(float3 A, float3 B, float3 C, float3 P)
{
	// Prepare our barycentric variables
	float3 u = B - A;
	float3 v = C - A;
	float3 w = P - A;

	float3 vCrossW = cross(v, w);
	float3 vCrossU = cross(v, u);

	// Test sign of r
	if (dot(vCrossW, vCrossU) < 0)
	{
		return false;
	}

	float3 uCrossW = cross(u, w);
	float3 uCrossV = cross(u, v);

	// Test sign of t
	if (dot(uCrossW, uCrossV) < 0)
	{
		return false;
	}		

	// At this point, we know that r and t and both > 0.
	// Therefore, as long as their sum is <= 1, each must be less <= 1
	float denom = length(uCrossV);
	float r = length(vCrossW) / denom;
	float t = length(uCrossW) / denom;

	return (r + t <= 1);
}

__device__ float2 toBarycentric(
	const float2 p1, const float2 p2, const float2 p3, const float2 p)
{
	// http://www.blackpawn.com/texts/pointinpoly/
	// Compute vectors
	float2 v0 = p1 - p3;

	float2 v1 = p2 - p3;
	float2 v2 = p - p3;

	// Compute dot products
	float dot00 = dot(v0, v0);
	float dot01 = dot(v0, v1);
	float dot02 = dot(v0, v2);
	float dot11 = dot(v1, v1);
	float dot12 = dot(v1, v2);
	// Compute barycentric coordinates
	float invDenom = 1.0f / (dot00 * dot11 - dot01 * dot01);

	float2 out_uv;
	out_uv.x = (dot11 * dot02 - dot01 * dot12) * invDenom;
	out_uv.y = (dot00 * dot12 - dot01 * dot02) * invDenom;
	return out_uv;
}

__device__ float4 GetBBox(const float2 &p0, const float2 &p1, const float2 &p2) //MIN, MAX
{	
	float2 min_max_value_x = make_float2(
		min(min(p0.x, p1.x), p2.x),
		max(max(p0.x, p1.x), p2.x)
	);

	float2 min_max_value_y = make_float2(
		min(min(p0.y, p1.y), p2.y),
		max(max(p0.y, p1.y), p2.y)
	);

	return make_float4(floor(min_max_value_x.x), floor(min_max_value_y.x),
		ceil(min_max_value_x.y), ceil(min_max_value_y.y));
}

__device__ float edgeFunction(const float2 &a, const float2 &b, const float2 &c)
{
	return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
}

__device__ float3 interplate_float3(const float3 &v0, const float3 &v1, const float3 &v2, const float3 &coeff)
{
	return make_float3(
		dot(make_float3(v0.x, v1.x, v2.x), coeff),
		dot(make_float3(v0.y, v1.y, v2.y), coeff),
		dot(make_float3(v0.z, v1.z, v2.z), coeff)
	);
}

__device__ void interplate_triangle_buffer(const int w, const int h, const int2 &lb, const int2 &rt, 
	const float2 &p0, const float2 &p1, const float2 &p2, 
	const int index0, const int index1, const int index2, GPULightmass::SurfelData* surfel_data)
{
	float area = edgeFunction(p0, p1, p2);

	//int w = RasBBox[1].x - RasBBox[0].x;
	//int h = RasBBox[1].y - RasBBox[0].y;

	printf("(%d, %d) (%d, %d) w = %d, h = %d, (%f, %f), (%f, %f), (%f, %f)\n", 
		lb.x, lb.y, rt.x, rt.y, w, h, p0.x, p0.y, p1.x, p1.y, p2.x, p2.y);

	for (int i = lb.y; i <= rt.y; ++i)
	{
		for (int j = lb.x; j <= rt.x; ++j)
		{
			float2 p = { j + 0.5f, i + 0.5f };
			float w0 = edgeFunction(p1, p2, p);
			float w1 = edgeFunction(p2, p0, p);
			float w2 = edgeFunction(p0, p1, p);
			if (w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f) {
				w0 /= area;
				w1 /= area;
				w2 /= area;
				
				float3 local_pos0 = RasVertexLocalPos[index0];
				float3 local_pos1 = RasVertexLocalPos[index1];
				float3 local_pos2 = RasVertexLocalPos[index2];
				float3 out_interplate_pos = interplate_float3(local_pos0, local_pos1, local_pos2, make_float3(w0, w1, w2));

				float3 local_normal0 = RasVertexNormals[index0];
				float3 local_normal1 = RasVertexNormals[index1];
				float3 local_normal2 = RasVertexNormals[index2];
				float3 out_interplate_normal = interplate_float3(local_normal0, local_normal1, local_normal2, make_float3(w0, w1, w2));

				//get output index
				int index_out_texel = (i - RasBBox[0].z) * w + (j - RasBBox[0].x);
				surfel_data[index_out_texel].pos = make_float4(out_interplate_pos, 1.0f);
				surfel_data[index_out_texel].normal = make_float4(out_interplate_normal);

				printf("write pixel (%d, %d), data index %d\n", j, i, index_out_texel);
			}
			else
			{
				//printf("No pixel (%d, %d)\n", j, i);
			}
		}
	}
}

__device__ void transform_vertex(const Mat4f& m, float3& p)
{
	const Vec4f p4f = Vec4f(p.x, p.y, p.z, 1.0f);
	Vec4f t1 = m * p4f;
	p = make_float3(t1.x / t1.w, t1.y / t1.w, t1.z / t1.w);
}

__device__ void toOrthoProjectSpace( float3& p0, float3& p1, float3& p2)
{
	Mat4f ortho_m;
	ortho_m.orthoMatrix(RasBBox[0].z, RasBBox[1].z, RasBBox[0].x, RasBBox[1].x, RasBBox[0].y, RasBBox[1].y);
	//Mat4f vp_mat = ortho_m * (*RasViewMat);
	Mat4f vp_mat = (*RasViewMat);//先不归一化

	transform_vertex(vp_mat, p0);
	transform_vertex(vp_mat, p1);
	transform_vertex(vp_mat, p2);
}

__global__ void rtVertexTransform()
{
	int vertex_id = blockIdx.x * blockDim.x + threadIdx.x;

	if (vertex_id < RasNumVertices)
	{

	}
}

__global__ void PlaneRasterization()
{
	int triangle_index = blockIdx.x * blockDim.x + threadIdx.x;	

	if (triangle_index < RasNumTriangles)
	{
		int index_0 = RasTriangleIndexs[triangle_index * 3];
		int index_1 = RasTriangleIndexs[triangle_index * 3 + 1];
		int index_2 = RasTriangleIndexs[triangle_index * 3 + 2];

		float3 p0 = RasVertexLocalPos[index_0];
		float3 p1 = RasVertexLocalPos[index_1];
		float3 p2 = RasVertexLocalPos[index_2];

		toOrthoProjectSpace(p0, p1, p2);

		float2 p0_on_plane = make_float2(p0.x, p0.z) / RasGridElementSize;
		float2 p1_on_plane = make_float2(p1.x, p1.z) / RasGridElementSize;
		float2 p2_on_plane = make_float2(p2.x, p2.z) / RasGridElementSize;
		
		float4 min_max_value = GetBBox(p0_on_plane, p1_on_plane, p2_on_plane);

		printf("(%f, %f, %f), (%f, %f, %f), (%f, %f, %f), bbox = (%f, %f), (%f, %f)\n\n",
			p0.x, p0.y, p0.z, p1.x, p1.y, p1.z, p2.x, p2.y, p2.z, min_max_value.x, min_max_value.y, min_max_value.z, min_max_value.w);

		int w = RasBBox[1].x - RasBBox[0].x;
		int h = RasBBox[1].z - RasBBox[0].z;
		interplate_triangle_buffer(w, h, make_int2(min_max_value.x, min_max_value.y), make_int2(min_max_value.z, min_max_value.w),
			p0_on_plane, p1_on_plane, p2_on_plane, index_0, index_1, index_2, RasXZPlaneBuffer);
		//RasYZPlaneBuffer[0].pos = make_float3(100.0f, 100.0f, 100.0f);
	}
}


__host__ void rtRasterizeModel(const int NumVertices, const int NumTriangles)
{
	/*const int Stride = 64;
	dim3 blockDim(Stride, 1);
	dim3 gridDim(divideAndRoundup(LaunchSizeX * LaunchSizeY, Stride), 1);

	CalculateDirectLighting << <gridDim, blockDim >> > ();*/

	//vertex transform
	/*int NumVerticesSqrt = std::ceil(std::sqrtf(NumVertices));
	dim3 blockDimVertices(NumVerticesSqrt);
	dim3 gridDimVertices(NumVerticesSqrt);
	rtVertexTransform << <gridDimVertices, blockDimVertices >> > ();*/

	const int Stride = 64;
	dim3 blockDim(Stride, 1);
	dim3 gridDim(divideAndRoundup(NumTriangles, Stride), 1);
	PlaneRasterization << <gridDim, blockDim >> > ();
}