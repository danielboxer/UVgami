#include "CuMeshSimplifier.h"
#include "cusimp.h"
#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#include <cuda_runtime.h>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <unordered_set>


#include <easy/profiler.h>


using cusimp::Vertex;
using cusimp::Triangle;
using cusimp::Edge;

// ‑‑‑‑ helper: largest bbox edge ------------------------------------------------
static float bbox_max_extent(const Eigen::MatrixXf &V)
{
    auto minv = V.colwise().minCoeff();
    auto maxv = V.colwise().maxCoeff();
    return (maxv - minv).maxCoeff();
}

// ‑‑‑‑ helper: boundary edges ---------------------------------------------------
static std::vector<Edge<int>>
boundary_edges_from_faces(const std::vector<Triangle<int>> &F)
{
    struct H { size_t operator()(const std::pair<int,int>& p) const noexcept
               { return (size_t(p.first) << 32) ^ size_t(p.second); } };

    std::unordered_map<std::pair<int,int>,int,H> cnt;  cnt.reserve(F.size()*3);

    auto add=[&](int a,int b){ if(a>b) std::swap(a,b); ++cnt[{a,b}]; };
    for(auto &t:F){ add(t.i,t.j); add(t.j,t.k); add(t.k,t.i); }

    std::vector<Edge<int>> E;  E.reserve(cnt.size());
    for(auto &kv:cnt) if(kv.second==1) E.push_back({kv.first.first,kv.first.second});
    return E;
}

// ‑‑‑‑ helper: boundary vertices -------------------------------------------------
static std::vector<int>
get_boundary_vertices(const std::vector<Edge<int>> &E)
{
    std::unordered_set<int> boundary_vertices;
    for(auto &e:E)
    {
        boundary_vertices.insert(e.u);
        boundary_vertices.insert(e.v);
    }
    return std::vector<int>(boundary_vertices.begin(),boundary_vertices.end());
}

// ‑‑‑‑ helper: copy device→host --------------------------------------------------
template<class T>
static std::vector<T> d2h(const T* d,size_t n, cudaStream_t stream)
{
    std::vector<T> h(n);
    cudaMemcpyAsync(h.data(),d,n*sizeof(T),cudaMemcpyDeviceToHost, stream);
    return h;
}

// ‑‑‑‑ public interface ---------------------------------------------------------
std::pair<Eigen::MatrixXf,Eigen::MatrixXi>
CuMeshSimplifier::simplify(const Eigen::MatrixXf& V_in,
                           const Eigen::MatrixXi& F_in,
                           float threshold,
                           int   max_iter,
                           cudaStream_t stream)
{
    // copy – we promise "not in‑place"
    Eigen::MatrixXf V = V_in;
    Eigen::MatrixXi F = F_in;

    // centre the mesh like PaSP.preprocess_mesh
    Eigen::Vector3f mean = V.colwise().mean();
    V.rowwise() -= mean.transpose();
    float scale = bbox_max_extent(V);

    // convert to cusimp host vectors
    std::vector<Vertex<float>> verts(V.rows());
    for(int i=0;i<V.rows();++i) verts[i]={V(i,0),V(i,1),V(i,2)};

    std::vector<Triangle<int>> faces(F.rows());
    for(int i=0;i<F.rows();++i) faces[i]={F(i,0),F(i,1),F(i,2)};

    cusimp::CUSimp sp;
    bool init=true;
    EASY_BLOCK("Simplify",profiler::colors::Grey);

    auto boundary = boundary_edges_from_faces(faces);
    auto fixed_vertices = get_boundary_vertices(boundary);
    EASY_BLOCK("mallocHost",profiler::colors::White);

    int* d_fixed_vertices;
    Vertex<float>*  d_v; Triangle<int>* d_f; Edge<int>* d_e;
    cudaMallocHost(&d_v, verts.size()*sizeof(Vertex<float>));
    cudaMallocHost(&d_f, faces.size()*sizeof(Triangle<int>));
    // cudaMallocHost(&d_e, boundary.size()*sizeof(Edge<int>));
    cudaMallocHost(&d_fixed_vertices, fixed_vertices.size()*sizeof(int));
    EASY_END_BLOCK;
    for(int it=0; it<max_iter; ++it)
    {
        boundary = boundary_edges_from_faces(faces);
        fixed_vertices = get_boundary_vertices(boundary);


        // empty boundary
        // boundary.clear();

        // managed copies of current mesh


        EASY_BLOCK("malloc",profiler::colors::Brown);

        std::memcpy(d_v,verts.data(),verts.size()*sizeof(Vertex<float>));
        std::memcpy(d_f,faces.data(),faces.size()*sizeof(Triangle<int>));
        // std::memcpy(d_e,boundary.data(),boundary.size()*sizeof(Edge<int>));
        std::memcpy(d_fixed_vertices, fixed_vertices.data(), fixed_vertices.size()*sizeof(int));
        EASY_END_BLOCK;

        


        EASY_BLOCK("forward",profiler::colors::Grey);

        if (stream == nullptr) {
            sp.forward(d_v,d_f,d_e,d_fixed_vertices,
                       int(verts.size()), int(faces.size()), int(boundary.size()),
                       int(fixed_vertices.size()),
                       scale,threshold,init);
        } else {
            sp.forward(d_v,d_f,d_e,d_fixed_vertices,
                       int(verts.size()), int(faces.size()), int(boundary.size()),
                       int(fixed_vertices.size()),
                       scale,threshold,init, stream);
        }
        EASY_END_BLOCK;
        // cudaDeviceSynchronize();
        cudaStreamSynchronize(stream);
        EASY_BLOCK("d2h",profiler::colors::Blue);
        auto occ = d2h(sp.pts_occ, sp.n_pts, stream);
        auto map = d2h(sp.pts_map, sp.n_pts, stream);
        auto v   = d2h(sp.points , sp.n_pts, stream);
        auto f   = d2h(sp.triangles, sp.n_tris, stream);
        EASY_END_BLOCK;

        std::vector<Triangle<int>> faces_new;
        faces_new.reserve(f.size());
        for(auto &t:f) if(t.i>=0)
            faces_new.push_back({map[t.i],map[t.j],map[t.k]});

        std::vector<Vertex<float>> verts_new;
        verts_new.reserve(verts.size());
        for(size_t i=0;i<v.size();++i) if(occ[i]) verts_new.push_back(v[i]);

        if(faces_new.size()==faces.size())
        {
            // cudaFree(d_v); cudaFree(d_f); cudaFree(d_e);
            break;      // converged
        }
        verts.swap(verts_new);
        faces.swap(faces_new);
        init=false;
        // std::cout << "iter: " << it << " faces: " << faces.size() << std::endl;
        // cudaFree(d_v); cudaFree(d_f); cudaFree(d_e);
    }
    EASY_END_BLOCK;
    // convert back to Eigen & un‑centre
    Eigen::MatrixXf V_out(verts.size(),3);
    for(int i=0;i<V_out.rows();++i)
    {
        V_out.row(i) << verts[i].x,verts[i].y,verts[i].z;
    }
    V_out.rowwise() += mean.transpose();

    Eigen::MatrixXi F_out(faces.size(),3);
    for(int i=0;i<F_out.rows();++i)
    {
        F_out(i,0)=faces[i].i; F_out(i,1)=faces[i].j; F_out(i,2)=faces[i].k;
    }

    // cudaStreamDestroy(stream); // Cleanup
    return {V_out,F_out};
}
