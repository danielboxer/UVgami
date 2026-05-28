#include "cusimp.h"
#include "thrust/device_ptr.h"
#include "thrust/sort.h"
#include "thrust/fill.h"
#include <math.h>
#include <thrust/shuffle.h>
#include <thrust/random.h>

#include <easy/profiler.h>

namespace cusimp
{
    const int BLOCK_SIZE = 512;
    const float COST_RANGE = 10.0;



    bool check_cuda_result(cudaError_t code, const char *file, int line)
    {
        if (code == cudaSuccess)
            return true;

        fprintf(stderr, "CUDA error %u: %s (%s:%d)\n", unsigned(code), cudaGetErrorString(code), file,
                line);
        return false;
    }

#define CHECK_CUDA(code) check_cuda_result(code, __FILE__, __LINE__);

    template <typename T>
    inline __device__ __host__ T min(T a, T b) { return a < b ? a : b; }
    template <typename T>
    inline __device__ __host__ T max(T a, T b) { return a > b ? a : b; }
    template <typename T>
    inline __device__ __host__ T clamp(T x, T a, T b) { return min(max(a, x), b); }

    void CUSimp::ensure_temp_storage_size(size_t size)
    {
        if (size > allocated_temp_storage_size)
        {
            allocated_temp_storage_size = size_t(size + size / 5);
            // fprintf(stderr, "allocated_temp_storage_size %ld\n", allocated_temp_storage_size);
            CHECK_CUDA(cudaFreeAsync(temp_storage, stream));
            CHECK_CUDA(cudaMallocAsync((void **)&temp_storage, allocated_temp_storage_size, stream));
        }
    }

    void CUSimp::ensure_pts_storage_size(size_t point_count)
    {
        if (point_count > allocated_pts)
        {
            allocated_pts = size_t(point_count + point_count / 5);
            // fprintf(stderr, "allocated_pts %ld\n", allocated_pts);
            CHECK_CUDA(cudaFreeAsync(points, stream));
            CHECK_CUDA(
                cudaMallocAsync((void **)&points, (allocated_pts + 1) * sizeof(Vertex<float>), stream));

            CHECK_CUDA(cudaFreeAsync(pts_occ, stream));
            CHECK_CUDA(
                cudaMallocAsync((void **)&pts_occ, (allocated_pts + 1) * sizeof(int), stream));

            CHECK_CUDA(cudaFreeAsync(pts_map, stream));
            CHECK_CUDA(
                cudaMallocAsync((void **)&pts_map, (allocated_pts + 1) * sizeof(int), stream));
        }
    }

    void CUSimp::ensure_tris_storage_size(size_t tris_count)
    {
        if (tris_count > allocated_tris)
        {
            allocated_tris = size_t(tris_count + tris_count / 5);
            // fprintf(stderr, "allocated_tris %ld\n", allocated_tris);
            CHECK_CUDA(cudaFreeAsync(triangles, stream));
            CHECK_CUDA(
                cudaMallocAsync((void **)&triangles, (allocated_tris + 1) * sizeof(Triangle<int>), stream));
        }
    }

    void CUSimp::ensure_near_count_storage_size(size_t point_count)
    {
        if (point_count > allocated_near_count)
        {
            allocated_near_count = size_t(point_count + point_count / 5);
            // fprintf(stderr, "allocated_near_count %ld\n", allocated_near_count);
            CHECK_CUDA(cudaFreeAsync(first_near_tris, stream));
            CHECK_CUDA(
                cudaMallocAsync((void **)&first_near_tris, (allocated_near_count + 1) * sizeof(int), stream));
        }
    }

    void CUSimp::ensure_near_tris_storage_size(size_t near_tri_count)
    {
        if (near_tri_count > allocated_near_tris)
        {
            allocated_near_tris = size_t(near_tri_count + near_tri_count / 5);
            // fprintf(stderr, "allocated_near_tris %ld\n", allocated_near_tris);
            CHECK_CUDA(cudaFreeAsync(near_tris, stream));
            CHECK_CUDA(
                cudaMallocAsync((void **)&near_tris, (allocated_near_tris + 1) * sizeof(int), stream));
        }
    }

    void CUSimp::ensure_near_offset_storage_size(size_t point_count)
    {
        if (point_count > allocated_near_offset)
        {
            allocated_near_offset = size_t(point_count + point_count / 5);
            // fprintf(stderr, "allocated_near_offset %ld\n", allocated_near_offset);
            CHECK_CUDA(cudaFreeAsync(near_offset, stream));
            CHECK_CUDA(
                cudaMallocAsync((void **)&near_offset, (allocated_near_offset + 1) * sizeof(int), stream));
        }
    }

    void CUSimp::ensure_edge_count_storage_size(size_t tris_count)
    {
        if (tris_count > allocated_edge_count)
        {
            allocated_edge_count = size_t(tris_count + tris_count / 5);
            // fprintf(stderr, "allocated_edge_count %ld\n", allocated_edge_count);
            CHECK_CUDA(cudaFreeAsync(first_edge, stream));
            CHECK_CUDA(
                cudaMallocAsync((void **)&first_edge, (allocated_edge_count + 1) * sizeof(int), stream));
        }
    }

    void CUSimp::ensure_edge_storage_size(size_t edge_count)
    {
        if (edge_count > allocated_edge)
        {
            allocated_edge = size_t(edge_count + edge_count / 5);
            // fprintf(stderr, "allocated_edge %ld\n", allocated_edge);
            CHECK_CUDA(cudaFreeAsync(edges, stream));
            CHECK_CUDA(
                cudaMallocAsync((void **)&edges, (allocated_edge + 1) * sizeof(Edge<int>), stream));
        }
    }

    void CUSimp::ensure_vert_Q_storage_size(size_t point_count)
    {
        if (point_count > allocated_vert_Q)
        {
            allocated_vert_Q = size_t(point_count + point_count / 5);
            // fprintf(stderr, "allocated_vert_Q %ld\n", allocated_vert_Q);
            CHECK_CUDA(cudaFreeAsync(vert_Q, stream));
            CHECK_CUDA(
                cudaMallocAsync((void **)&vert_Q, (allocated_vert_Q + 1) * sizeof(Mat4x4<float>), stream));
        }
    }

    void CUSimp::ensure_edge_cost_storage_size(size_t edge_count)
    {
        if (edge_count > allocated_edge_cost)
        {
            allocated_edge_cost = size_t(edge_count + edge_count / 5);
            // fprintf(stderr, "allocated_edge_cost %ld\n", allocated_edge_cost);
            CHECK_CUDA(cudaFreeAsync(edge_cost, stream));
            CHECK_CUDA(
                cudaMallocAsync((void **)&edge_cost, (allocated_edge_cost + 1) * sizeof(uint32_t), stream));
        }
    }

    __host__ void CUSimp::ensure_fixed_vertex_mask_storage_size(size_t point_count)
    {
        if (point_count > allocated_fixed_vert_mask)
            allocated_fixed_vert_mask = size_t(point_count + point_count / 5);

        CHECK_CUDA(cudaFreeAsync(fixed_vert_mask, stream));
        CHECK_CUDA(cudaMallocAsync((void **)&fixed_vert_mask,
                            (allocated_fixed_vert_mask + 1) * sizeof(bool), stream));
    }

    __host__ void CUSimp::ensure_fixed_vertices_storage_size(size_t n_fixed)
    {
        if (n_fixed > allocated_fixed_vertices)
            allocated_fixed_vertices = size_t(n_fixed + n_fixed / 5);

        CHECK_CUDA(cudaFreeAsync(fixed_vertices, stream));
        CHECK_CUDA(cudaMallocAsync((void **)&fixed_vertices,
                            (allocated_fixed_vertices + 1) * sizeof(int), stream));
    }

    __global__ void create_fixed_vertex_mask_kernel(CUSimp sp)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= sp.n_fixed_vertices) return;

        int v = sp.fixed_vertices[idx];
        sp.fixed_vert_mask[v] = true;
    }

    __host__ void CUSimp::ensure_boundary_links_storage_size(size_t n_pts)
    {
        if (n_pts > allocated_boundary_links)
        {
            if (boundary_next)  cudaFreeAsync(boundary_next, stream);
            if (boundary_prev)  cudaFreeAsync(boundary_prev, stream);

            CHECK_CUDA(cudaMallocAsync(&boundary_next, n_pts * sizeof(int), stream));
            CHECK_CUDA(cudaMallocAsync(&boundary_prev, n_pts * sizeof(int), stream));

            allocated_boundary_links = n_pts;
        }
    }
    __global__ void create_boundary_links_kernel(CUSimp sp)
    {
        int eIdx = blockIdx.x * blockDim.x + threadIdx.x;
        if (eIdx >= sp.n_boundary_edges) return;

        Edge<int> e = sp.boundary_edges[eIdx];
        int u = e.u, v = e.v;

        // We assume a manifold boundary ⇒ each vertex sees at most two edges.
        // Atomic index 0 = 'next', index 1 = 'prev'
        // --------------------------------------------------------------
        auto insert = [] __device__ (int vert, int nbr,
                                    int *next, int *prev)
        {
            // slot = atomicCAS(next+vert, -1, nbr)  tries to write into 'next'
            // if 'next' already occupied, write into 'prev'
            if (atomicCAS(next + vert, -1, nbr) != -1)
                atomicCAS(prev + vert, -1, nbr);
        };

        insert(u, v, sp.boundary_next, sp.boundary_prev);
        insert(v, u, sp.boundary_next, sp.boundary_prev);
    }

    void CUSimp::ensure_boundary_edges_storage_size(size_t edge_count)
    {
        if (edge_count > allocated_boundary_edges)
        {
            allocated_boundary_edges = size_t(edge_count + edge_count / 5);
        }
        CHECK_CUDA(cudaFreeAsync(boundary_edges, stream));
        CHECK_CUDA(
            cudaMallocAsync((void **)&boundary_edges, (allocated_boundary_edges + 1) * sizeof(Edge<int>), stream));
    }

    void CUSimp::ensure_boundary_vertex_mask_storage_size(size_t point_count)
    {
        if (point_count > allocated_boundary_vert_mask)
        {
            allocated_boundary_vert_mask = size_t(point_count + point_count / 5);
        }
        CHECK_CUDA(cudaFreeAsync(boundary_vert_mask, stream));
        CHECK_CUDA(
            cudaMallocAsync((void **)&boundary_vert_mask, (allocated_boundary_vert_mask + 1) * sizeof(bool), stream));
    }

    __global__ void create_boundary_vertex_mask_kernel(CUSimp sp)
    {
        int edge_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (edge_index >= sp.n_boundary_edges)
            return;

        Edge<int> edge = sp.boundary_edges[edge_index];
        sp.boundary_vert_mask[edge.u] = true;
        sp.boundary_vert_mask[edge.v] = true;
    }


    void CUSimp::ensure_tri_min_cost_storage_size(size_t tri_count)
    {
        if (tri_count > allocated_tri_min_cost)
        {
            allocated_tri_min_cost = size_t(tri_count + tri_count / 5);
            // fprintf(stderr, "allocated_tri_min_cost %ld\n", allocated_tri_min_cost);
            CHECK_CUDA(cudaFreeAsync(tri_min_cost, stream));
            CHECK_CUDA(
                cudaMallocAsync((void **)&tri_min_cost, (allocated_tri_min_cost + 1) * sizeof(uint64_cu), stream));
        }
    }

    __global__ void count_near_tris_kernel(CUSimp sp)
    {
        int tri_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (tri_index >= sp.n_tris)
            return;

        Triangle<int> tri = sp.triangles[tri_index];
        if (tri.i == -1 && tri.j == -1 && tri.k == -1)
            return;

        atomicAdd(&sp.first_near_tris[tri.i], 1);
        atomicAdd(&sp.first_near_tris[tri.j], 1);
        atomicAdd(&sp.first_near_tris[tri.k], 1);
    }

    __global__ void create_near_tris_kernel(CUSimp sp)
    {
        int tri_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (tri_index >= sp.n_tris)
            return;

        Triangle<int> tri = sp.triangles[tri_index];
        if (tri.i == -1 && tri.j == -1 && tri.k == -1)
            return;

        sp.near_tris[sp.first_near_tris[tri.i] + atomicAdd(&sp.near_offset[tri.i], 1)] = tri_index;
        sp.near_tris[sp.first_near_tris[tri.j] + atomicAdd(&sp.near_offset[tri.j], 1)] = tri_index;
        sp.near_tris[sp.first_near_tris[tri.k] + atomicAdd(&sp.near_offset[tri.k], 1)] = tri_index;
    }

    __global__ void count_edge_kernel(CUSimp sp)
    {
        int tri_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (tri_index >= sp.n_tris)
            return;

        Triangle<int> tri = sp.triangles[tri_index];
        if (tri.i == -1 && tri.j == -1 && tri.k == -1)
            return;
        sp.first_edge[tri_index] += (tri.i > tri.j) + (tri.j > tri.k) + (tri.k > tri.i);
    }

    __global__ void create_edge_kernel(CUSimp sp)
    {
        int tri_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (tri_index >= sp.n_tris)
            return;

        Triangle<int> tri = sp.triangles[tri_index];
        if (tri.i == -1 && tri.j == -1 && tri.k == -1)
            return;

        int first = sp.first_edge[tri_index];
        if (tri.i > tri.j)
            sp.edges[first++] = {int(tri.j), int(tri.i)};
        if (tri.j > tri.k)
            sp.edges[first++] = {int(tri.k), int(tri.j)};
        if (tri.k > tri.i)
            sp.edges[first++] = {int(tri.i), int(tri.k)};
    }

    __global__ void create_boundary_mask_kernel(CUSimp sp)
    {
        int edge_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (edge_index >= sp.n_edges)
            return;

        // Edge<int> edge = sp.edges[edge_index];
        
    }

    __device__ Vec4<float> tri2plane(Vertex<float> const *points, Triangle<int> tri)
    {
        Vertex<float> v0 = points[tri.i];
        Vertex<float> v1 = points[tri.j];
        Vertex<float> v2 = points[tri.k];
        // Vertex<float> v0 = {0, 0, 2};
        // Vertex<float> v1 = {1, 0, 0};
        // Vertex<float> v2 = {0, 3, 0};
        Vertex<float> normal = (v1 - v0).cross(v2 - v0);
        normal /= normal.norm();
        float offset = -normal.dot(v0);
        return {normal.x, normal.y, normal.z, offset};
    }

    __device__ int other_boundary_vertex(int v, int exclude, const CUSimp& sp)
    {
        int n0 = sp.boundary_next[v];
        int n1 = sp.boundary_prev[v];

        // Helper to test a candidate neighbour
        auto valid = [&] __device__ (int x) {
            return (x >= 0)                       // slot was filled
                && (x != exclude)                 // not the same as the other endpoint
                && sp.boundary_vert_mask[x];      // still marked as boundary
        };

        if (valid(n0)) return n0;
        if (valid(n1)) return n1;
        return -1;  // no valid neighbour found
    }


    __device__ float compute_vert_angle(int v,  const CUSimp& sp)
    {   
        // printf("compute_vert_angle %d\n", v);
        int w_u = other_boundary_vertex(v, -1, sp);
        int w_v = other_boundary_vertex(v, w_u, sp);

        if (w_u == -1 || w_v == -1) return 0;

        auto safe_angle = [] __device__ (Vertex<float> a,
                                        Vertex<float> b) -> float
        {
            float len_ab = a.norm() * b.norm() + 1e-20f;
            float cos_th = clamp(a.dot(b) / len_ab, -1.0f, 1.0f);
            return acosf(cos_th);                    // [0, π]
        };
        const float PI = 3.14159265358979323846f;

        float bend_cost ;
        if (w_u >= 0 && w_v >= 0)
        {
            Vertex<float> duv = sp.points[w_u] - sp.points[v];
            Vertex<float> duw = sp.points[w_v] - sp.points[v];
            bend_cost = fabsf(PI - safe_angle(duv, duw));   // 0 when straight
        }else{
            bend_cost = PI;
        }
        return bend_cost / PI;
        
        
    }

    __global__ void compute_vert_Q_kernel(CUSimp sp)
    {
        int pt_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (pt_index >= sp.n_pts)
            return;
        // printf("pt_index %d sp.n_pts %d\n", pt_index, sp.n_pts);

        int first = sp.first_near_tris[pt_index];
        int last = sp.first_near_tris[pt_index + 1];
        Mat4x4<float> Kp{0};
        for (int i = first; i < last; ++i)
        {
            Vec4<float> p = tri2plane(sp.points, sp.triangles[sp.near_tris[i]]);
            // if (pt_index == 2644)
            //     printf("p %f %f %f %f\n", p.x, p.y, p.z, p.w);
            // Mat4x4<float> temp = p.dot_T(p);
            Kp += p.dot_T(p);
            // printf("p %f %f %f %f\ntemp %f %f %f %f\n %f %f %f %f\n %f %f %f %f\n %f %f %f %f\n\n", p.x, p.y, p.z, p.w, temp.m00, temp.m01, temp.m02, temp.m03, temp.m10, temp.m11, temp.m12, temp.m13, temp.m20, temp.m21, temp.m22, temp.m23, temp.m30, temp.m31, temp.m32, temp.m33);
        }
        sp.vert_Q[pt_index] = Kp;
    }

    __device__ float triangle_area(Vertex<float> p0, Vertex<float> p1, Vertex<float> p2)
    {
        float a = (p0 - p1).norm();
        float b = (p1 - p2).norm();
        float c = (p2 - p0).norm();
        float s = (a + b + c) / 2;
        return sqrt(s * (s - a) * (s - b) * (s - c));
    }

    __device__ float edge_length(Vertex<float> p0, Vertex<float> p1)
    {
        return (p0 - p1).norm();
    }

    __device__ int find_adjacent_boundary_edge(int v,
                                            int current_eidx,
                                            const CUSimp& sp)
    {
        for (int ei = sp.first_edge[v]; ei < sp.first_edge[v+1]; ++ei)
        {
            if (ei == current_eidx) continue;
            Edge<int> e = sp.edges[ei];
            // boundary edge ⇔ both endpoints are boundary vertices
            if (sp.boundary_vert_mask[e.u] && sp.boundary_vert_mask[e.v])
                return ei;
        }
        return -1;              // isolated end‑vertex on the boundary (hole)
    }


    __global__ void compute_edge_cost_kernel(CUSimp sp)
    {
        int edge_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (edge_index >= sp.n_edges)
            return;
        // printf("edge_index %d sp.n_edges %d\n", edge_index, sp.n_edges);


        Edge<int> edge = sp.edges[edge_index];
        Vertex<float> v0 = sp.points[edge.u];
        Vertex<float> v1 = sp.points[edge.v];

        // if (sp.boundary_vert_mask[edge.u] || sp.boundary_vert_mask[edge.v])
        // {
        //     // printf("edge %d %d is a boundary edge\n", edge.u, edge.v);  
        //     sp.edge_cost[edge_index] = std::numeric_limits<uint32_t>::max();
        //     return;
        // }

        // printf("edge.u %d edge.v %d\n", edge.u, edge.v);
        // const float MAX_VERT_ANGLE_COST = 1./4;
        // // Skip edges that touch a user‑fixed vertex
        // if (sp.fixed_vert_mask[edge.u] && compute_vert_angle(edge.u, sp) > MAX_VERT_ANGLE_COST)
        // {
        //     sp.edge_cost[edge_index] = std::numeric_limits<uint32_t>::max();
        //     return;
        // }

        // if (sp.fixed_vert_mask[edge.v] && compute_vert_angle(edge.v, sp) > MAX_VERT_ANGLE_COST)
        // {
        //     sp.edge_cost[edge_index] = std::numeric_limits<uint32_t>::max();
        //     return;
        // }

        if(sp.fixed_vert_mask[edge.u] || sp.fixed_vert_mask[edge.v]){
            sp.edge_cost[edge_index] = std::numeric_limits<uint32_t>::max();
            return;
        }


        //-----------------------------------------------
        //  Boundary‑angle penalty   (runs *only* when the
        //  current edge itself is on the boundary)
        

        float bend_cost = 0.0f;
        // if (sp.boundary_vert_mask[edge.u] || sp.boundary_vert_mask[edge.v]){
        //     if (sp.boundary_vert_mask[edge.u] && sp.boundary_vert_mask[edge.v])
        //     {
        //         int w_u = other_boundary_vertex(edge.u, edge.v, sp);
        //         int w_v = other_boundary_vertex(edge.v, edge.u, sp);

        //         auto safe_angle = [] __device__ (Vertex<float> a,
        //                                         Vertex<float> b) -> float
        //         {
        //             float len_ab = a.norm() * b.norm() + 1e-20f;
        //             float cos_th = clamp(a.dot(b) / len_ab, -1.0f, 1.0f);
        //             return acosf(cos_th);                    // [0, π]
        //         };

        //         const float PI = 3.14159265358979323846f;

        //         if (w_u >= 0)
        //         {
        //             Vertex<float> duv = sp.points[edge.v] - sp.points[edge.u];
        //             Vertex<float> duw = sp.points[w_u   ] - sp.points[edge.u];
        //             bend_cost += fabsf(PI - safe_angle(duv, duw));   // 0 when straight
        //         }
        //         if (w_v >= 0)
        //         {
        //             Vertex<float> dvu = sp.points[edge.u] - sp.points[edge.v];
        //             Vertex<float> dvw = sp.points[w_v   ] - sp.points[edge.v];
        //             bend_cost += fabsf(PI - safe_angle(dvu, dvw));
        //         }
        //         if (bend_cost == 0)
        //         {
        //             sp.edge_cost[edge_index] = std::numeric_limits<uint32_t>::max();
        //             return;
        //         }
        //         // Normalise and weight   (max possible sum is 2·π)
        //         const float max_cost = 1./8 ; // angle_left + angle_right = 1/2 * PI (max is 2 pi )
        //         // const float w_bend = sp.tres / max_cost;
        //         bend_cost = bend_cost / PI;       // range 0 … 2·w_bend
                
        //         // printf("w_bend %f, collapse_t %f, max_cost %f\n", w_bend, sp.tres, max_cost);

        //         // printf("bend_cost %f for edge %d %d\n", bend_cost, edge.u, edge.v);
        //         if (bend_cost > max_cost)
        //         {
        //             sp.edge_cost[edge_index] = std::numeric_limits<uint32_t>::max();
        //             return;
        //         }
        //     }else{
        //         sp.edge_cost[edge_index] = std::numeric_limits<uint32_t>::max();
        //         return;
        //     }
        // }

        // check if the collapse is valid
        int dup_num = 0;
        for (int i = sp.first_near_tris[edge.u]; i < sp.first_near_tris[edge.u + 1]; ++i)
        {
            int idx_u;
            // edge i-j; j-k; k-i
            if (sp.triangles[sp.near_tris[i]].i == edge.u)
                idx_u = sp.triangles[sp.near_tris[i]].j;
            else if (sp.triangles[sp.near_tris[i]].j == edge.u)
                idx_u = sp.triangles[sp.near_tris[i]].k;
            else if (sp.triangles[sp.near_tris[i]].k == edge.u)
                idx_u = sp.triangles[sp.near_tris[i]].i;
            else
                printf("error1\n");
            for (int j = sp.first_near_tris[edge.v]; j < sp.first_near_tris[edge.v + 1]; ++j)
            {
                int idx_v;
                // edge i-j; j-k; k-i
                if (sp.triangles[sp.near_tris[j]].i == edge.v)
                    idx_v = sp.triangles[sp.near_tris[j]].j;
                else if (sp.triangles[sp.near_tris[j]].j == edge.v)
                    idx_v = sp.triangles[sp.near_tris[j]].k;
                else if (sp.triangles[sp.near_tris[j]].k == edge.v)
                    idx_v = sp.triangles[sp.near_tris[j]].i;
                else
                    printf("error2\n");
                if (idx_u == idx_v)
                    dup_num++;
                if (dup_num > 2)
                {
                    sp.edge_cost[edge_index] = std::numeric_limits<uint32_t>::max();
                    return;
                }
            }
        }
        if (dup_num != 2)
        {
            return;
            printf("dup_num %d\n", dup_num);
        }

        // compute near edge length
        float edge_len = 0;
        int num_edge = 0;
        for (int i = sp.first_near_tris[edge.u]; i < sp.first_near_tris[edge.u + 1]; ++i)
        {
            if (sp.triangles[sp.near_tris[i]].i > sp.triangles[sp.near_tris[i]].j)
            {
                edge_len += (sp.points[sp.triangles[sp.near_tris[i]].i] - sp.points[sp.triangles[sp.near_tris[i]].j]).norm();
                num_edge++;
            }
            if (sp.triangles[sp.near_tris[i]].j > sp.triangles[sp.near_tris[i]].k)
            {
                edge_len += (sp.points[sp.triangles[sp.near_tris[i]].j] - sp.points[sp.triangles[sp.near_tris[i]].k]).norm();
                num_edge++;
            }
            if (sp.triangles[sp.near_tris[i]].k > sp.triangles[sp.near_tris[i]].i)
            {
                edge_len += (sp.points[sp.triangles[sp.near_tris[i]].k] - sp.points[sp.triangles[sp.near_tris[i]].i]).norm();
                num_edge++;
            }
        }
        for (int i = sp.first_near_tris[edge.v]; i < sp.first_near_tris[edge.v + 1]; ++i)
        {
            if (sp.triangles[sp.near_tris[i]].i > sp.triangles[sp.near_tris[i]].j)
            {
                edge_len += (sp.points[sp.triangles[sp.near_tris[i]].i] - sp.points[sp.triangles[sp.near_tris[i]].j]).norm();
                num_edge++;
            }
            if (sp.triangles[sp.near_tris[i]].j > sp.triangles[sp.near_tris[i]].k)
            {
                edge_len += (sp.points[sp.triangles[sp.near_tris[i]].j] - sp.points[sp.triangles[sp.near_tris[i]].k]).norm();
                num_edge++;
            }
            if (sp.triangles[sp.near_tris[i]].k > sp.triangles[sp.near_tris[i]].i)
            {
                edge_len += (sp.points[sp.triangles[sp.near_tris[i]].k] - sp.points[sp.triangles[sp.near_tris[i]].i]).norm();
                num_edge++;
            }
        }
        edge_len = edge_len / num_edge / sp.edge_s * sp.tres;
        // printf("edge_len %f\n", edge_len);

        // compute edge length
        edge_len += edge_len + (v0 - v1).norm() / sp.edge_s * sp.tres;
        // edge_len = 0;

        Vertex<float> v = (v0 + v1) / 2;

        // compute skinny triangle cost Q_a
        // test if the triangle normal is flipped
        float Q_a = 0;
        int num_tri = 0;
        for (int i = sp.first_near_tris[edge.u]; i < sp.first_near_tris[edge.u + 1]; ++i)
        {
            Triangle<int> old_tri = sp.triangles[sp.near_tris[i]];
            // if the triangle is shared by edge.u and edge.v, skip
            if (old_tri.i == edge.v || old_tri.j == edge.v || old_tri.k == edge.v)
                continue;

            Vertex<float> old_v0 = sp.points[old_tri.i];
            Vertex<float> old_v1 = sp.points[old_tri.j];
            Vertex<float> old_v2 = sp.points[old_tri.k];

            Vertex<float> new_v0 = sp.points[old_tri.i];
            Vertex<float> new_v1 = sp.points[old_tri.j];
            Vertex<float> new_v2 = sp.points[old_tri.k];

            // replace edge.u with v
            if (old_tri.i == edge.u)
                new_v0 = v;
            if (old_tri.j == edge.u)
                new_v1 = v;
            if (old_tri.k == edge.u)
                new_v2 = v;

            Vertex<float> old_normal = (old_v1 - old_v0).cross(old_v2 - old_v0);
            old_normal /= old_normal.norm();
            Vertex<float> new_normal = (new_v1 - new_v0).cross(new_v2 - new_v0);
            new_normal /= new_normal.norm();

            // if the normal is flipped, invalid collapse
            if (old_normal.dot(new_normal) < 0)
            {
                sp.edge_cost[edge_index] = std::numeric_limits<uint32_t>::max();
                return;
            }

            // compute the area of the new triangle
            Q_a += 1.0f - clamp(float(4.0f * sqrt(3) * triangle_area(new_v0, new_v1, new_v2) / pow(edge_length(new_v0, new_v1), 2) + pow(edge_length(new_v1, new_v2), 2) + pow(edge_length(new_v2, new_v0), 2) + 0.0000001f), 0.0f, 1.0f);
            num_tri++;
        }

        for (int i = sp.first_near_tris[edge.v]; i < sp.first_near_tris[edge.v + 1]; ++i)
        {
            Triangle<int> old_tri = sp.triangles[sp.near_tris[i]];
            // if the triangle is shared by edge.u and edge.v, skip
            if (old_tri.i == edge.u || old_tri.j == edge.u || old_tri.k == edge.u)
                continue;

            Vertex<float> old_v0 = sp.points[old_tri.i];
            Vertex<float> old_v1 = sp.points[old_tri.j];
            Vertex<float> old_v2 = sp.points[old_tri.k];

            Vertex<float> new_v0 = sp.points[old_tri.i];
            Vertex<float> new_v1 = sp.points[old_tri.j];
            Vertex<float> new_v2 = sp.points[old_tri.k];

            // replace edge.v with v
            if (old_tri.i == edge.v)
                new_v0 = v;
            if (old_tri.j == edge.v)
                new_v1 = v;
            if (old_tri.k == edge.v)
                new_v2 = v;

            Vertex<float> old_normal = (old_v1 - old_v0).cross(old_v2 - old_v0);
            old_normal /= old_normal.norm();
            Vertex<float> new_normal = (new_v1 - new_v0).cross(new_v2 - new_v0);
            new_normal /= new_normal.norm();

            // if the normal is flipped, invalid collapse
            if (old_normal.dot(new_normal) < 0)
            {
                sp.edge_cost[edge_index] = std::numeric_limits<uint32_t>::max();
                return;
            }

            // compute the area of the new triangle
            Q_a += 1.0f - clamp(float(4.0f * sqrt(3) * triangle_area(new_v0, new_v1, new_v2) / pow(edge_length(new_v0, new_v1), 2) + pow(edge_length(new_v1, new_v2), 2) + pow(edge_length(new_v2, new_v0), 2) + 0.0000001f), 0.0f, 1.0f);
            num_tri++;
        }

        Mat4x4<float> Q = sp.vert_Q[edge.u] + sp.vert_Q[edge.v];
        Vec4<float> v4 = {v.x, v.y, v.z, 1};
        float cost = Q.vTMv(v4) / (sp.edge_s * sp.edge_s);
        // printf("cost %f, edge_len %f, Q_a %f, bend_cost %f\n", cost, edge_len, Q_a / num_tri * sp.tres, bend_cost);
        sp.edge_cost[edge_index] = uint32_t(clamp(cost + edge_len + Q_a / num_tri * sp.tres + bend_cost, 0.0f, COST_RANGE) / COST_RANGE * std::numeric_limits<uint32_t>::max());
    }

    __global__ void propagate_edge_cost_kernel(CUSimp sp)
    {
        uint32_t edge_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (edge_index >= sp.n_edges)
            return;
        // printf("edge_index %d sp.n_edges %d\n", edge_index, sp.n_edges);
        // printf("sp.n_edges %d\n", sp.n_edges);

        Edge<int> edge = sp.edges[edge_index];
        uint64_cu cost = (((uint64_cu)sp.edge_cost[edge_index]) << 32) | edge_index;
        // printf("cost %llu edge_index %d, ((uint64_cu)edge_index) << 32 %llu, sp.edge_cost[edge_index] %u \n", cost, edge_index, ((uint64_cu)edge_index) << 32, sp.edge_cost[edge_index]);

        int first = sp.first_near_tris[edge.u];
        int last = sp.first_near_tris[edge.u + 1];
        for (int i = first; i < last; ++i)
        {
            atomicMin(&sp.tri_min_cost[sp.near_tris[i]], cost);
        }

        first = sp.first_near_tris[edge.v];
        last = sp.first_near_tris[edge.v + 1];
        for (int i = first; i < last; ++i)
        {
            atomicMin(&sp.tri_min_cost[sp.near_tris[i]], cost);
        }
    }

    __global__ void collapse_edge_kernel(CUSimp sp)
    {
        int edge_index = blockIdx.x * blockDim.x + threadIdx.x;
        if (edge_index >= sp.n_edges)
            return;
        // printf("edge_index %d\n", edge_index);
        // atomicAdd(&sp.debug[0], 1);

        Edge<int> edge = sp.edges[edge_index];
        uint64_cu cost = (((uint64_cu)sp.edge_cost[edge_index]) << 32) | edge_index;
        // printf("sp.edge_cost[edge_index] %u, sp.collapse_t %u\n", sp.edge_cost[edge_index], sp.collapse_t);
        if (sp.edge_cost[edge_index] > sp.collapse_t)
        {
            // printf("enter\n");
            return;
        }

        Vertex<float> v0 = sp.points[edge.u];
        Vertex<float> v1 = sp.points[edge.v];

        int first = sp.first_near_tris[edge.u];
        int last = sp.first_near_tris[edge.u + 1];
        for (int i = first; i < last; ++i)
        {
            // printf("sp.near_tri[i] %d, sp.tri_min_cost[sp.near_tri[i]] %llu, cost %llu\n", sp.near_tris[i], sp.tri_min_cost[sp.near_tris[i]], cost);
            if (sp.tri_min_cost[sp.near_tris[i]] != cost)
            {
                // printf("%d edge %d - %d, sp.tri_min_cost[sp.near_tris[i]] %llu, cost %llu\n", edge_index, edge.u, edge.v, sp.tri_min_cost[sp.near_tris[i]], cost);
                return;
            }
        }

        first = sp.first_near_tris[edge.v];
        last = sp.first_near_tris[edge.v + 1];
        for (int i = first; i < last; ++i)
        {
            // printf("sp.tri_min_cost[i] %llu, cost %llu\n", sp.tri_min_cost[i], cost);
            if (sp.tri_min_cost[sp.near_tris[i]] != cost)
            {
                // printf("%d edge %d - %d, sp.tri_min_cost[sp.near_tris[i]] %llu, cost %llu\n", edge_index, edge.u, edge.v, sp.tri_min_cost[sp.near_tris[i]], cost);
                return;
            }
        }

        // printf("collapse %d - %d\n", edge.u, edge.v);
        Vertex<float> v = (v0 + v1) / 2;
        sp.points[edge.u] = v;
        sp.points[edge.v] = {0, 0, 0};
        sp.pts_occ[edge.v] = 0;
        first = sp.first_near_tris[edge.u];
        last = sp.first_near_tris[edge.u + 1];
        for (int i = first; i < last; ++i)
        {
            if (sp.triangles[sp.near_tris[i]].i == edge.v || sp.triangles[sp.near_tris[i]].j == edge.v || sp.triangles[sp.near_tris[i]].k == edge.v)
            {
                sp.triangles[sp.near_tris[i]].i = sp.triangles[sp.near_tris[i]].j = sp.triangles[sp.near_tris[i]].k = -1;
            }
        }

        first = sp.first_near_tris[edge.v];
        last = sp.first_near_tris[edge.v + 1];
        for (int i = first; i < last; ++i)
        {
            if (sp.triangles[sp.near_tris[i]].i == edge.u || sp.triangles[sp.near_tris[i]].j == edge.u || sp.triangles[sp.near_tris[i]].k == edge.u)
            {
                sp.triangles[sp.near_tris[i]].i = sp.triangles[sp.near_tris[i]].j = sp.triangles[sp.near_tris[i]].k = -1;
            }
            else if (sp.triangles[sp.near_tris[i]].i == edge.v)
                sp.triangles[sp.near_tris[i]].i = edge.u;
            else if (sp.triangles[sp.near_tris[i]].j == edge.v)
                sp.triangles[sp.near_tris[i]].j = edge.u;
            else if (sp.triangles[sp.near_tris[i]].k == edge.v)
                sp.triangles[sp.near_tris[i]].k = edge.u;
        }
    }

__host__ void CUSimp::forward(Vertex<float> *pts,
                              Triangle<int> *tris,
                              Edge<int> *b_edges,
                              int *fixed_vs,             //  ← new
                              int  nPts,
                              int  nTris,
                              int  nBoundaryEdges,
                              int  nFixedVertices,       //  ← new
                              float scale,
                              float threshold,
                              bool  init)
    {
        tres = threshold;
        edge_s = scale;
        collapse_t = uint32_t(clamp(threshold, float(0.0), COST_RANGE) / COST_RANGE * std::numeric_limits<uint32_t>::max());

        resize(nPts, nTris, nBoundaryEdges);

        ensure_pts_storage_size(n_pts);
        CHECK_CUDA(cudaMemcpy(points, pts, n_pts * sizeof(Vertex<float>),
                              cudaMemcpyHostToDevice));
        std::vector<int> tmp(n_pts, 1);
        CHECK_CUDA(cudaMemcpy(pts_occ, tmp.data(), n_pts * sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(pts_occ+n_pts, 0, sizeof(int)));
        ensure_tris_storage_size(n_tris);
        CHECK_CUDA(cudaMemcpy(triangles, tris, n_tris * sizeof(Triangle<int>),
                              cudaMemcpyHostToDevice));

        // ensure_boundary_edges_storage_size(nBoundaryEdges);
        // CHECK_CUDA(cudaMemcpy(boundary_edges, b_edges, nBoundaryEdges * sizeof(Edge<int>),
        //                       cudaMemcpyHostToDevice));

        // CHECK_CUDA(cudaMemcpy(edges, boundary_edges, n_edges * sizeof(Edge<int>),
        //                       cudaMemcpyHostToDevice));
        // ---------------- fixed vertices -----------------------------------
        n_fixed_vertices = nFixedVertices;

        ensure_fixed_vertices_storage_size(n_fixed_vertices);
        CHECK_CUDA(cudaMemcpy(fixed_vertices, fixed_vs,
                            n_fixed_vertices * sizeof(int),
                            cudaMemcpyHostToDevice));

        ensure_fixed_vertex_mask_storage_size(n_pts);
        CHECK_CUDA(cudaMemset(fixed_vert_mask, 0, n_pts * sizeof(bool)));

        create_fixed_vertex_mask_kernel<<<(n_fixed_vertices + BLOCK_SIZE - 1) /
                                        BLOCK_SIZE, BLOCK_SIZE>>>(*this);

        if (init){
            thrust::device_ptr<Triangle<int>> thrust_triangles(triangles);
            thrust::default_random_engine rng;
            thrust::shuffle(thrust_triangles, thrust_triangles + n_tris, rng);
        }

        size_t temp_storage_bytes = 0;

        ensure_near_count_storage_size(n_pts);
        CHECK_CUDA(cudaMemset(first_near_tris, 0, (n_pts + 1) * sizeof(int)));
        count_near_tris_kernel<<<(n_tris + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(*this);


        cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes, first_near_tris, first_near_tris, n_pts + 1);
        ensure_temp_storage_size(temp_storage_bytes);
        cub::DeviceScan::ExclusiveSum(temp_storage, allocated_temp_storage_size, first_near_tris, first_near_tris, n_pts + 1);

        CHECK_CUDA(cudaMemcpy(&n_near_tris, first_near_tris + n_pts, sizeof(int),
                                cudaMemcpyDeviceToHost));
        // fprintf(stderr, "near tris %d\n", n_near_tris);


        ensure_near_tris_storage_size(n_near_tris);
        ensure_near_offset_storage_size(n_pts);
        CHECK_CUDA(cudaMemset(near_offset, 0, (n_pts + 1) * sizeof(int)));
        create_near_tris_kernel<<<(n_tris + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(*this);

        // // Test near_tris
        // int *near_tris_host;
        // cudaMallocAsyncHost((void **)&near_tris_host, n_near_tris * sizeof(int));
        // cudaMemcpy(near_tris_host, near_tris, n_near_tris * sizeof(int), cudaMemcpyDeviceToHost);
        // for (int i=0;i<n_near_tris;i++)
        //     printf("%d - near_tris_host: %d\n", i, near_tris_host[i]);

        ensure_edge_count_storage_size(n_tris);
        CHECK_CUDA(cudaMemset(first_edge, 0, (n_tris + 1) * sizeof(int)));
        count_edge_kernel<<<(n_tris + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(*this);

        cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes, first_edge, first_edge, n_tris + 1);
        ensure_temp_storage_size(temp_storage_bytes);
        cub::DeviceScan::ExclusiveSum(temp_storage, allocated_temp_storage_size, first_edge, first_edge, n_tris + 1);

        CHECK_CUDA(cudaMemcpy(&n_edges, first_edge + n_tris, sizeof(int),
                                cudaMemcpyDeviceToHost));

        // printf("n_edges: %d\n", n_edges);

        ensure_edge_storage_size(n_edges);
        create_edge_kernel<<<(n_tris + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(*this);
        
        
        // printf("n_edges=%d  n_boundary_edges=%d  n_fixed_vertices=%d\n", n_edges, n_boundary_edges, n_fixed_vertices);
        // ensure_boundary_vertex_mask_storage_size(n_pts);
        // create_boundary_vertex_mask_kernel<<<(n_edges + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(*this);


        // ensure_boundary_links_storage_size(n_pts);
        // CHECK_CUDA(cudaMemset(boundary_next, -1, n_pts * sizeof(int)));
        // CHECK_CUDA(cudaMemset(boundary_prev, -1, n_pts * sizeof(int)));
        // create_boundary_links_kernel<<<(n_boundary_edges + BLOCK_SIZE - 1) / BLOCK_SIZE,
        //                             BLOCK_SIZE>>>(*this);
        
        // // print boundary_vert_mask
        // bool *boundary_vert_mask_host;
        // cudaMallocAsyncHost((void **)&boundary_vert_mask_host, n_pts * sizeof(bool));
        // cudaMemcpy(boundary_vert_mask_host, this->boundary_vert_mask, n_pts * sizeof(bool), cudaMemcpyDeviceToHost);
        // for (int i=0;i<n_pts;i++)
        //     if (boundary_vert_mask_host[i])
        //         printf("%d - boundary_vert_mask_host: %d\n", i, boundary_vert_mask_host[i]);
        // cudaFreeAsyncHost(boundary_vert_mask_host);

        // // print fixed_vert_mask
        // bool *fixed_vert_mask_host;
        // cudaMallocAsyncHost((void **)&fixed_vert_mask_host, n_pts * sizeof(bool));
        // cudaMemcpy(fixed_vert_mask_host, this->fixed_vert_mask, n_pts * sizeof(bool), cudaMemcpyDeviceToHost);
        // for (int i=0;i<n_pts;i++)
        //     if (fixed_vert_mask_host[i])
        //         printf("%d - fixed_vert_mask_host: %d\n", i, fixed_vert_mask_host[i]);
        // cudaFreeAsyncHost(fixed_vert_mask_host);

        // // Test edges
        // Edge<int> *edges_host;
        // cudaMallocAsyncHost((void **)&edges_host, n_edges * sizeof(Edge<int>));
        // cudaMemcpy(edges_host, edges, n_edges * sizeof(Edge<int>), cudaMemcpyDeviceToHost);
        // for (int i=0;i<n_edges;i++)
        //     printf("%d - edges_host: %d - %d\n", i, edges_host[i].u, edges_host[i].v);

        ensure_vert_Q_storage_size(n_pts);
        compute_vert_Q_kernel<<<(n_pts + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(*this);

        // // Test vert_cost
        // float *vert_cost_host;
        // cudaMallocAsyncHost((void **)&vert_cost_host, n_pts * sizeof(float));
        // cudaMemcpy(vert_cost_host, vert_cost, n_pts * sizeof(float), cudaMemcpyDeviceToHost);
        // for (int i=0;i<n_pts;i++)
        //     printf("%d - vert_cost_host: %f\n", i, vert_cost_host[i]);

        ensure_edge_cost_storage_size(n_edges);
        compute_edge_cost_kernel<<<(n_edges + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(*this);

        // // Test edge_cost
        // uint32_t *edge_cost_host;
        // cudaMallocAsyncHost((void **)&edge_cost_host, n_edges * sizeof(uint32_t));
        // cudaMemcpy(edge_cost_host, edge_cost, n_edges * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        // for (int i = 0; i < n_edges; i++)
        //   printf("%d - edge_cost_host: %u\n", i, edge_cost_host[i]);

        ensure_tri_min_cost_storage_size(n_tris);
        std::vector<uint64_cu> temp(n_tris, std::numeric_limits<uint64_cu>::max());
        CHECK_CUDA(cudaMemcpy(tri_min_cost, temp.data(), n_tris * sizeof(uint64_cu), cudaMemcpyHostToDevice));
        propagate_edge_cost_kernel<<<(n_edges + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(*this);

        collapse_edge_kernel<<<(n_edges + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(*this);

        CHECK_CUDA(cudaMemcpy(pts_map, pts_occ, (n_pts+1) * sizeof(int), cudaMemcpyDeviceToDevice));
        cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes, pts_map, pts_map, n_pts + 1);
        ensure_temp_storage_size(temp_storage_bytes);
        cub::DeviceScan::ExclusiveSum(temp_storage, allocated_temp_storage_size, pts_map, pts_map, n_pts + 1);

        // // Test pts_occ_cost
        // int *pts_occ_host;
        // cudaMallocAsyncHost((void **)&pts_occ_host, (n_pts+1) * sizeof(int));
        // cudaMemcpy(pts_occ_host, pts_occ, (n_pts+1) * sizeof(int), cudaMemcpyDeviceToHost);
        // for (int i = 0; i < (n_pts+1); i++)
        //   printf("%d - pts_occ_host: %u\n", i, pts_occ_host[i]);

        // // Test pts_map_cost
        // int *pts_map_host;
        // cudaMallocAsyncHost((void **)&pts_map_host, (n_pts+1) * sizeof(int));
        // cudaMemcpy(pts_map_host, pts_map, (n_pts+1) * sizeof(int), cudaMemcpyDeviceToHost);
        // for (int i = 0; i < (n_pts+1); i++)
        //   printf("%d - pts_map_host: %u\n", i, pts_map_host[i]);
    }

__host__ void CUSimp::forward(Vertex<float> *pts,
                              Triangle<int> *tris,
                              Edge<int> *b_edges,
                              int *fixed_vs,             //  ← new
                              int  nPts,
                              int  nTris,
                              int  nBoundaryEdges,
                              int  nFixedVertices,       //  ← new
                              float scale,
                              float threshold,
                              bool  init,
                              cudaStream_t input_stream)
    {
        tres = threshold;
        edge_s = scale;
        collapse_t = uint32_t(clamp(threshold, float(0.0), COST_RANGE) / COST_RANGE * std::numeric_limits<uint32_t>::max());

        resize(nPts, nTris, nBoundaryEdges);
        EASY_BLOCK("cusimp::allocate memory", profiler::colors::Cyan);
        stream = input_stream;

        ensure_pts_storage_size(n_pts);
        CHECK_CUDA(cudaMemcpyAsync(points, pts, n_pts * sizeof(Vertex<float>),
                              cudaMemcpyHostToDevice, stream));
        std::vector<int> tmp(n_pts, 1);
        CHECK_CUDA(cudaMemcpyAsync(pts_occ, tmp.data(), n_pts * sizeof(int), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemsetAsync(pts_occ+n_pts, 0, sizeof(int), stream));
        ensure_tris_storage_size(n_tris);
        CHECK_CUDA(cudaMemcpyAsync(triangles, tris, n_tris * sizeof(Triangle<int>),
                              cudaMemcpyHostToDevice, stream));

        // ensure_boundary_edges_storage_size(nBoundaryEdges);
        // CHECK_CUDA(cudaMemcpyAsync(boundary_edges, b_edges, nBoundaryEdges * sizeof(Edge<int>),
        //                       cudaMemcpyHostToDevice, stream));

        // CHECK_CUDA(cudaMemcpy(edges, boundary_edges, n_edges * sizeof(Edge<int>),
        //                       cudaMemcpyHostToDevice));
        // ---------------- fixed vertices -----------------------------------
        n_fixed_vertices = nFixedVertices;

        ensure_fixed_vertices_storage_size(n_fixed_vertices);
        CHECK_CUDA(cudaMemcpyAsync(fixed_vertices, fixed_vs,
                            n_fixed_vertices * sizeof(int),
                            cudaMemcpyHostToDevice, stream));

        ensure_fixed_vertex_mask_storage_size(n_pts);
        CHECK_CUDA(cudaMemsetAsync(fixed_vert_mask, 0, n_pts * sizeof(bool), stream));

        create_fixed_vertex_mask_kernel<<<(n_fixed_vertices + BLOCK_SIZE - 1) /
                                        BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(*this);
        EASY_END_BLOCK;
        EASY_BLOCK("cusimp::shuffle triangles", profiler::colors::Cyan100);
        if (init){
            thrust::device_ptr<Triangle<int>> thrust_triangles(triangles);
            thrust::default_random_engine rng;
            auto policy = thrust::cuda::par.on(stream);
            thrust::shuffle(policy, thrust_triangles, thrust_triangles + n_tris, rng);
        }
        EASY_END_BLOCK;
        EASY_BLOCK("cusimp::acclocation2", profiler::colors::Cyan200);
        size_t temp_storage_bytes = 0;

        ensure_near_count_storage_size(n_pts);
        CHECK_CUDA(cudaMemsetAsync(first_near_tris, 0, (n_pts + 1) * sizeof(int), stream));
        count_near_tris_kernel<<<(n_tris + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(*this);


        cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes, first_near_tris, first_near_tris, n_pts + 1, stream);
        ensure_temp_storage_size(temp_storage_bytes);
        cub::DeviceScan::ExclusiveSum(temp_storage, allocated_temp_storage_size, first_near_tris, first_near_tris, n_pts + 1, stream);

        CHECK_CUDA(cudaMemcpyAsync(&n_near_tris, first_near_tris + n_pts, sizeof(int),
                                cudaMemcpyDeviceToHost, stream));
        // fprintf(stderr, "near tris %d\n", n_near_tris);


        ensure_near_tris_storage_size(n_near_tris);
        ensure_near_offset_storage_size(n_pts);
        CHECK_CUDA(cudaMemsetAsync(near_offset, 0, (n_pts + 1) * sizeof(int), stream));
        create_near_tris_kernel<<<(n_tris + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(*this);

        // // Test near_tris
        // int *near_tris_host;
        // cudaMallocAsyncHost((void **)&near_tris_host, n_near_tris * sizeof(int));
        // cudaMemcpy(near_tris_host, near_tris, n_near_tris * sizeof(int), cudaMemcpyDeviceToHost);
        // for (int i=0;i<n_near_tris;i++)
        //     printf("%d - near_tris_host: %d\n", i, near_tris_host[i]);

        ensure_edge_count_storage_size(n_tris);
        CHECK_CUDA(cudaMemsetAsync(first_edge, 0, (n_tris + 1) * sizeof(int), stream));
        count_edge_kernel<<<(n_tris + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(*this);

        cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes, first_edge, first_edge, n_tris + 1, stream);
        ensure_temp_storage_size(temp_storage_bytes);
        cub::DeviceScan::ExclusiveSum(temp_storage, allocated_temp_storage_size, first_edge, first_edge, n_tris + 1, stream);

        CHECK_CUDA(cudaMemcpyAsync(&n_edges, first_edge + n_tris, sizeof(int),
                                cudaMemcpyDeviceToHost, stream));
        EASY_END_BLOCK;
        // printf("n_edges: %d\n", n_edges);

        ensure_edge_storage_size(n_edges);
        create_edge_kernel<<<(n_tris + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(*this);
        
        
        // printf("n_edges=%d  n_boundary_edges=%d  n_fixed_vertices=%d\n", n_edges, n_boundary_edges, n_fixed_vertices);
        // ensure_boundary_vertex_mask_storage_size(n_pts);
        // create_boundary_vertex_mask_kernel<<<(n_edges + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(*this);


        // ensure_boundary_links_storage_size(n_pts);
        // CHECK_CUDA(cudaMemsetAsync(boundary_next, -1, n_pts * sizeof(int), stream));
        // CHECK_CUDA(cudaMemsetAsync(boundary_prev, -1, n_pts * sizeof(int), stream));
        // create_boundary_links_kernel<<<(n_boundary_edges + BLOCK_SIZE - 1) / BLOCK_SIZE,
        //                             BLOCK_SIZE, 0, stream>>>(*this);
        
        // // print boundary_vert_mask
        // bool *boundary_vert_mask_host;
        // cudaMallocAsyncHost((void **)&boundary_vert_mask_host, n_pts * sizeof(bool));
        // cudaMemcpy(boundary_vert_mask_host, this->boundary_vert_mask, n_pts * sizeof(bool), cudaMemcpyDeviceToHost);
        // for (int i=0;i<n_pts;i++)
        //     if (boundary_vert_mask_host[i])
        //         printf("%d - boundary_vert_mask_host: %d\n", i, boundary_vert_mask_host[i]);
        // cudaFreeAsyncHost(boundary_vert_mask_host);

        // // print fixed_vert_mask
        // bool *fixed_vert_mask_host;
        // cudaMallocAsyncHost((void **)&fixed_vert_mask_host, n_pts * sizeof(bool));
        // cudaMemcpy(fixed_vert_mask_host, this->fixed_vert_mask, n_pts * sizeof(bool), cudaMemcpyDeviceToHost);
        // for (int i=0;i<n_pts;i++)
        //     if (fixed_vert_mask_host[i])
        //         printf("%d - fixed_vert_mask_host: %d\n", i, fixed_vert_mask_host[i]);
        // cudaFreeAsyncHost(fixed_vert_mask_host);

        // // Test edges
        // Edge<int> *edges_host;
        // cudaMallocAsyncHost((void **)&edges_host, n_edges * sizeof(Edge<int>));
        // cudaMemcpy(edges_host, edges, n_edges * sizeof(Edge<int>), cudaMemcpyDeviceToHost);
        // for (int i=0;i<n_edges;i++)
        //     printf("%d - edges_host: %d - %d\n", i, edges_host[i].u, edges_host[i].v);

        ensure_vert_Q_storage_size(n_pts);
        compute_vert_Q_kernel<<<(n_pts + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(*this);

        // // Test vert_cost
        // float *vert_cost_host;
        // cudaMallocAsyncHost((void **)&vert_cost_host, n_pts * sizeof(float));
        // cudaMemcpy(vert_cost_host, vert_cost, n_pts * sizeof(float), cudaMemcpyDeviceToHost);
        // for (int i=0;i<n_pts;i++)
        //     printf("%d - vert_cost_host: %f\n", i, vert_cost_host[i]);

        ensure_edge_cost_storage_size(n_edges);
        compute_edge_cost_kernel<<<(n_edges + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(*this);

        // // Test edge_cost
        // uint32_t *edge_cost_host;
        // cudaMallocAsyncHost((void **)&edge_cost_host, n_edges * sizeof(uint32_t));
        // cudaMemcpy(edge_cost_host, edge_cost, n_edges * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        // for (int i = 0; i < n_edges; i++)
        //   printf("%d - edge_cost_host: %u\n", i, edge_cost_host[i]);

        ensure_tri_min_cost_storage_size(n_tris);
        std::vector<uint64_cu> temp(n_tris, std::numeric_limits<uint64_cu>::max());
        CHECK_CUDA(cudaMemcpyAsync(tri_min_cost, temp.data(), n_tris * sizeof(uint64_cu), cudaMemcpyHostToDevice, stream));
        propagate_edge_cost_kernel<<<(n_edges + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(*this);

        collapse_edge_kernel<<<(n_edges + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream  >>>(*this);

        CHECK_CUDA(cudaMemcpyAsync(pts_map, pts_occ, (n_pts+1) * sizeof(int), cudaMemcpyDeviceToDevice, stream));
        cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes, pts_map, pts_map, n_pts + 1, stream);
        ensure_temp_storage_size(temp_storage_bytes);
        cub::DeviceScan::ExclusiveSum(temp_storage, allocated_temp_storage_size, pts_map, pts_map, n_pts + 1, stream);

        // // Test pts_occ_cost
        // int *pts_occ_host;
        // cudaMallocAsyncHost((void **)&pts_occ_host, (n_pts+1) * sizeof(int));
        // cudaMemcpy(pts_occ_host, pts_occ, (n_pts+1) * sizeof(int), cudaMemcpyDeviceToHost);
        // for (int i = 0; i < (n_pts+1); i++)
        //   printf("%d - pts_occ_host: %u\n", i, pts_occ_host[i]);

        // // Test pts_map_cost
        // int *pts_map_host;
        // cudaMallocAsyncHost((void **)&pts_map_host, (n_pts+1) * sizeof(int));
        // cudaMemcpy(pts_map_host, pts_map, (n_pts+1) * sizeof(int), cudaMemcpyDeviceToHost);
        // for (int i = 0; i < (n_pts+1); i++)
        //   printf("%d - pts_map_host: %u\n", i, pts_map_host[i]);
    }

}