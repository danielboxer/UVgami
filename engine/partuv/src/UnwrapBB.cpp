#include <igl/read_triangle_mesh.h>
#include <igl/write_triangle_mesh.h>
#include <igl/per_face_normals.h>
#include <igl/adjacency_list.h>
#include <igl/lscm.h>

// #define STB_IMAGE_WRITE_IMPLEMENTATION
// #include "stb_image_write.h"

// LSCM
#include "Mesh.h"
#include "FormTrait.h"
#include "LSCM.h"

// Triangle intersection
#include "triangleHelper.hpp"

// Distortion
#include "Distortion.h"

#include <Eigen/Dense>
#include <Eigen/Core>
#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <string>
#include <unordered_map>
#include <set>

#include <execution>
#include <mutex>
#include <exception>
#include <stdexcept>
#include <sstream>
#include <fstream>
#include <filesystem>


#include <omp.h>
#include <atomic>


#include "UnwrapBB.h"
#include "UnwrapMerge.h"

#include "Component.h"

#include "merge.h"
#include "IO.h"






#include <easy/profiler.h>


using namespace MeshLib;

// A small helper to compute PCA (Eigen decomposition) on a set of rows in a MatrixXd
Eigen::Matrix3d computePCAEigenvectors(const Eigen::MatrixXd &data)
{
    // data: F x 3 (face normals)
    // 1. Compute centroid
    Eigen::RowVector3d mean = data.colwise().mean();

    // 2) Center the normals by subtracting the mean
    Eigen::MatrixXd FN_centered = data.rowwise() - mean;

    // 3) Compute the covariance matrix (3x3)
    //    Note: we use (N - 1) in the denominator for an unbiased estimate
    double denom = static_cast<double>(data.rows() - 1);
    Eigen::Matrix3d cov = (FN_centered.transpose() * FN_centered) / denom;

    // 3. Eigen decomposition
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
    if(solver.info() != Eigen::Success)
    {
        std::cerr << "PCA eigen decomposition failed!" << std::endl;
        return Eigen::Matrix3d::Identity();
    }
    // Columns of `eigenvectors()` are eigenvectors
    // By default, they are sorted in ascending order of eigenvalues
    Eigen::Matrix3d evecs = solver.eigenvectors();
    // Return the matrix of eigenvectors


    // Sort the eigenvalues and eigenvectors in descending order
    Eigen::Vector3d eigen_values = solver.eigenvalues();
    Eigen::Matrix3d eigen_vectors = solver.eigenvectors();

    // Create a vector of indices and sort them based on eigenvalues
    std::vector<int> idx(3);
    idx[0] = 0; idx[1] = 1; idx[2] = 2;
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        return eigen_values[a] > eigen_values[b];
    });

    // Reorder the eigenvectors based on sorted indices
    Eigen::Matrix3d sorted_evecs;
    for(int i = 0; i < 3; ++i){
        sorted_evecs.col(i) = eigen_vectors.col(idx[i]);
    }

    // Ensure the rotation matrix is right-handed and orthogonal
    if(sorted_evecs.determinant() < 0){
        sorted_evecs.col(2) *= -1;
    }

    sorted_evecs.col(0) *= -1;
    sorted_evecs.col(1) *= -1;

    return sorted_evecs;
}


// A small helper to get face adjacency (faces connected by a common edge)
std::vector<std::vector<int>> computeFaceAdjacency(
    const Eigen::MatrixXi &F, 
    const Eigen::MatrixXd &V,
    std::vector<std::vector<double>> &edge_lengths)
{
    // Basic approach to build face adjacency:
    //   1) For each face, add its edges into a map edge -> face_index
    //   2) If an edge is shared by multiple faces, those faces are adjacent
    // This is a simplified approach; libigl does not directly provide "face adjacency" 
    // but provides vertex adjacency and other utilities, so we do a custom approach here.

    struct EdgeKey
    {
        int v1, v2;
        EdgeKey(int a, int b)
        {
            // Store sorted to avoid directional mismatch
            v1 = std::min(a, b);
            v2 = std::max(a, b);
        }
        bool operator<(const EdgeKey &other) const 
        {
            return (v1 < other.v1) || ((v1 == other.v1) && v2 < other.v2);
        }
        bool operator==(const EdgeKey &other) const
        {
            return (v1 == other.v1 && v2 == other.v2);
        }
    };

    std::map<EdgeKey, std::vector<int>> edge2faces;
    std::map<EdgeKey, double> edge2length;

    // Collect edges in a map
    for(int fi = 0; fi < F.rows(); ++fi)
    {
        // Each face has 3 edges
        for(int e = 0; e < 3; e++)
        {
            int v0 = F(fi, e);
            int v1 = F(fi, (e + 1) % 3);
            EdgeKey key(v0, v1);
            edge2faces[key].push_back(fi);
            if(edge2length.find(key) == edge2length.end())
            {
                Eigen::RowVector3d p0 = V.row(v0);
                Eigen::RowVector3d p1 = V.row(v1);
                double length = (p1 - p0).norm();
                edge2length[key] = length;
            }
        }
    }

    // Now build adjacency from the map
    std::vector<std::vector<int>> faceAdj(F.rows());
    edge_lengths.resize(F.rows());
    for(const auto &kv : edge2faces)
    {
        const auto &faces = kv.second;
        double length = edge2length[kv.first];

        // If an edge is shared by multiple faces, link them
        for(size_t i = 0; i < faces.size(); ++i)
        {
            for(size_t j = i + 1; j < faces.size(); ++j)
            {
                int f1 = faces[i];
                int f2 = faces[j];
                faceAdj[f1].push_back(f2);
                edge_lengths[f1].push_back(length);
                faceAdj[f2].push_back(f1);
                edge_lengths[f2].push_back(length);
            }
        }
    }
    return faceAdj;
}



int findMostMatchedNeighbor(
    int compIndex,
    const std::vector<std::vector<std::pair<int,int>>> &compAdjList,
    const std::string &method)
{
    // compAdjList[compIndex] = list of (neighborComp, weight)
    // We pick the neighbor with the largest weight
    int bestComp = -1;
    int bestWeight = -1;
    for(const auto &edge : compAdjList[compIndex])
    {
        int neighbor = edge.first;
        int weight   = edge.second;
        if(weight > bestWeight)
        {
            bestWeight = weight;
            bestComp   = neighbor;
        }
    }
    return bestComp;
}




Eigen::MatrixXd computeDotProducts(
    const MatrixX3R &faceNormals,
    const Eigen::Matrix<double, 6, 3, Eigen::RowMajor> &obbDirections) 
{
    Eigen::MatrixXd result(faceNormals.rows(), 6);
    result.noalias() = faceNormals * obbDirections.transpose();
    return result;
}



   // Wrap the preparation code into a standalone function
void prepareOBBData(
    Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    MatrixX3R &FN,
    std::vector<int> &faceAssignment,
    std::vector<std::vector<int>> &faceAdj,
    std::vector<std::vector<double>> &edge_lengths)
{
    // 1) Compute face normals
    igl::per_face_normals(V, F, FN);

    // 2) Align main axes via PCA on face normals
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> evecs = computePCAEigenvectors(FN);
    Eigen::MatrixXd evecs_inv = evecs.inverse();
    V = (V * evecs).eval();
    igl::per_face_normals(V, F, FN);

    // 3) Prepare 6 OBB directions
    Eigen::MatrixXd obb_face_normals(6, 3);
    obb_face_normals << 
            1,  0,  0,
            -1,  0,  0,
            0,  1,  0,
            0, -1,  0,
            0,  0,  1,
            0,  0, -1;

    // 4) Compute dot products => face assignment
    auto start = std::chrono::high_resolution_clock::now();
    Eigen::MatrixXd dotProducts = computeDotProducts(FN, obb_face_normals);
    #ifdef ENABLE_PROFILING
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "computeDotProducts took "
                    << std::chrono::duration<double>(end - start).count()
                    << " seconds." << std::endl;
    #endif

    faceAssignment.resize(F.rows(), 0);
    start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < F.rows(); ++i)
    {
        double maxVal = -1e9;
        int maxIdx = -1;
        for(int d = 0; d < 6; ++d)
        {
            if(dotProducts(i, d) > maxVal)
            {
                maxVal = dotProducts(i, d);
                maxIdx = d;
            }
        }
        faceAssignment[i] = maxIdx;
    }
    #ifdef ENABLE_PROFILING
    end = std::chrono::high_resolution_clock::now();

    start = std::chrono::high_resolution_clock::now();
    #endif
    
    // 5) Build face adjacency
    faceAdj = computeFaceAdjacency(F, V, edge_lengths);
    #ifdef ENABLE_PROFILING
        end = std::chrono::high_resolution_clock::now();
        std::cout << "computeFaceAdjacency took "
                    << std::chrono::duration<double>(end - start).count()
                    << " seconds." << std::endl;
    #endif
}

// Returns a grouping of faces in `finalUVComponents`, where each entry is a list of face indices
// std::vector<std::vector<int>> unwrap_aligning_BB( Eigen::MatrixXd &V,  Eigen::MatrixXi &F)
std::vector<Component> unwrap_aligning_BB( const Eigen::MatrixXd &V, const Eigen::MatrixXi &F,double threshold, bool check_overlap, int chart_limit)
{

    MatrixX3R FN;
    std::vector<int> faceAssignment;
    std::vector<std::vector<int>> faceAdj;

    // Prepare data for OBB alignment
    Eigen::MatrixXd V_rotated = V;
    std::vector<std::vector<double>> edge_lengths;
    prepareOBBData(V_rotated, F, FN, faceAssignment, faceAdj, edge_lengths);

    // 6) For each of the 6 directions, gather connected components of faces
    //    We'll store them in finalUVComponents (largest one) + standbyUVComponents (remaining)
    std::vector<Component> finalUVComponents;    // each entry is a chart
    std::vector<Component> standbyUVComponents;  // leftover sub-charts
 
    std::vector<Component> componentsMap;
    std::map<int, int> faceToComponent;

    for(int dir = 0; dir < 6; ++dir)
    {
        // Mark faces assigned to `dir`
        std::vector<bool> visitedMask(F.rows(), false);
        for(int i = 0; i < F.rows(); ++i)
        {
            if(faceAssignment[i] == dir)
                visitedMask[i] = false; // not visited but valid
            else
                visitedMask[i] = true;  // exclude
        }

        // Find all connected components among these faces
        std::vector<std::vector<int>> components;
        std::vector<bool> globalVisited(F.rows(), false);
        for(int i = 0; i < F.rows(); i++)
        {
            if(!visitedMask[i] && !globalVisited[i])
            {
                auto comp = findConnectedComponent(i, faceAdj, visitedMask, globalVisited);
                if(!comp.empty())
                    components.push_back(comp);
            }
        }

        // Largest as base
        std::vector<int> largestCC = largestComponent(components);
        if(!largestCC.empty())
        {
            Eigen::MatrixXd curr_FN(largestCC.size(), FN.cols());
            for (int i = 0; i < largestCC.size(); ++i) {
                curr_FN.row(i) = FN.row(largestCC[i]);
            }
        
            Component curr_comp = Component(componentsMap.size(), largestCC, -1, curr_FN);

            finalUVComponents.push_back(curr_comp);
            componentsMap.push_back(curr_comp);
        }

        // The leftover components go to standby
        for(const auto &comp : components)
        {
            if(comp.size() == largestCC.size() && comp == largestCC)
                continue; // skip the largest
            Eigen::MatrixXd curr_FN(comp.size(), FN.cols());
            for (int i = 0; i < comp.size(); ++i) {
                curr_FN.row(i) = FN.row(comp[i]);
            }
            Component curr_comp = Component(componentsMap.size(), comp, -1, curr_FN);
            if(!comp.empty())
                standbyUVComponents.push_back(curr_comp);
            componentsMap.push_back(curr_comp);
        }
    }

    // std::vector<int> faceToComponent(F.rows(), -1);
    // int numComponents = faceAssignment.max();
    int numComponents = componentsMap.size();


    for (const auto &component : componentsMap)
    {
        for (int faceIdx : component.faces)
        {
            faceToComponent[faceIdx] = component.index;
        }
    }


    // int total_faces = 0;
    // for (const auto &comp : componentsMap)
    // {
    //     total_faces += comp.faces.size();
    //     if(comp.faces.size() == 22){
    //         std::cout << "component index: " << comp.index << std::endl;
    //     }
    // }
    // std::cout << "total faces in finalUVComponents: " << total_faces << std::endl;




    // Now build adjacency between these components
    std::vector<std::vector<std::pair<int,int>>>  compAdjList = buildComponentAdjacency(faceAdj, faceToComponent, componentsMap, numComponents);

    #ifdef VERBOSE
    int row_num = 0;
    for ( const auto &row : compAdjList )
    {
        std::cout << "Row: " << row_num++ << ": ";
        for ( const auto &s : row ){  std::cout << "(" << s.first << "," << s.second << ") ";}
        std::cout << std::endl;  
    }
    std::cout << "total components at starting point: " << numComponents << std::endl;
    #endif
    // 7) Merge remaining components to the "most matched" chart
    std::vector<Eigen::RowVector3d> chartNormals(finalUVComponents.size(), Eigen::RowVector3d::Zero());
    // Now merge leftover


    auto new_standbyUVComponents = standbyUVComponents;

    while(!new_standbyUVComponents.empty()){

        compAdjList = buildComponentAdjacency(faceAdj, faceToComponent, componentsMap, numComponents);

        standbyUVComponents = new_standbyUVComponents;
        new_standbyUVComponents.clear();

        for(auto &standbyComp : standbyUVComponents)
        {
            // Compute average normal of standbyComp
            Eigen::RowVector3d avg(0,0,0);

            Eigen::RowVector3d normal = standbyComp.face_normals.colwise().mean();
            avg += normal;
            avg.normalize();

            // Find best matched chart
            double bestDot = -1e9;
            int bestChart = -1;
            int bestChartIndex = -1;
            int compIndex = -1;


            
            // std::vector<int> neighbors;
            // neighbors.reserve(compAdjList[standbyComp.index].size()); // optional but can help performance
            // for (const auto& edge : compAdjList[standbyComp.index]) neighbors.push_back(edge.first);

            // std::sort(
            //     neighbors.begin(), 
            //     neighbors.end(), 
            //     [&](int a, int b) {return avg.dot(componentsMap[a].avg_normal) > avg.dot(componentsMap[b].avg_normal);}
            //     // [&](int a, int b) {return compAdjList[standbyComp.index][a].second > compAdjList[standbyComp.index][b].second;}
            // );
            // int success_to_index = -1;
            // for (int neighbor : neighbors)



            for(Component final_comp : finalUVComponents)
            {
                compIndex++;

                // #ifdef VERBOSE  std::cout << "checking for the component: " << standbyComp.index << " with the component: " << final_comp.index << std::endl; #endif
                bool found = false;
                for(const auto &edge : compAdjList[standbyComp.index])
                {
                    if(edge.first == final_comp.index)
                    {
                        found = true;
                        break;
                    }
                }
                if(!found) { continue; }
                #ifdef VERBOSE  
                std::cout << "found the component: " << standbyComp.index << " with the component: " << final_comp.index << std::endl; 
                #endif

                double d = avg.dot(final_comp.avg_normal);
                if(d > bestDot)
                {
                    bestDot = d;
                    bestChart = final_comp.index;
                    bestChartIndex = compIndex;
                }

            }

            if(bestChartIndex!= -1){
                standbyComp.cube_face_idx = bestChart;
                finalUVComponents[bestChartIndex].faces.insert(finalUVComponents[bestChartIndex].faces.end(), standbyComp.faces.begin(), standbyComp.faces.end());

                // update faceToComponent
                for (int faceIdx : standbyComp.faces)
                {
                    faceToComponent[faceIdx] = bestChart;
                }
                // erase from componentsMap
                // componentsMap.erase(componentsMap.begin() + standbyComp.index);
                standbyComp.faces.clear();

            }
            else{
                // std::cout << "Component " << standbyComp.index << " has no best chart" << std::endl;
                // std::cout << "standbyComp.faces.size(): " << standbyComp.faces.size() << std::endl;
                new_standbyUVComponents.push_back(standbyComp);
            }

        }

        if (new_standbyUVComponents.size() ==  standbyUVComponents.size()){
            std::cout << "WARNING: Found dangling components" << std::endl;
            for (const auto &standbyComp : new_standbyUVComponents)
            {
                std::cout << "standbyComp.index: " << standbyComp.index << std::endl;
            }
            break;

        }
        #ifdef VERBOSE  
        std::cout << "bestChartIndex: " << bestChartIndex << "for the component: " << standbyComp.index << std::endl; 
        std::cout << "standbyComp.faces.size(): " << standbyComp.faces.size() << std::endl; 
        #endif

    }

    // finalUVComponents now holds the "segmented" faces
    // In a real UV unwrapping pipeline, you'd:
    //    - Parameterize each chart individually (e.g., planar mapping)
    //    - Store them in a UV matrix or per-face attribute
    // For this demonstration, we just return the grouping as sets of face indices

    std::vector<std::vector<int>> tempFaces;
    tempFaces.reserve(finalUVComponents.size());
    for (auto &comp : finalUVComponents) {
        tempFaces.push_back(std::move(comp.faces));
    }
    
    // Step 2: Call the function (which will modify tempFaces)
    smoothComponentEdge(tempFaces, faceAdj);
    
    // Step 3: Move the modified vectors back into the original components
    for (size_t i = finalUVComponents.size(); i-- > 0; ) {
        if (tempFaces[i].empty()) {
            finalUVComponents.erase(finalUVComponents.begin() + i);
        } else {
            finalUVComponents[i].faces = std::move(tempFaces[i]);
        }
    }
#ifndef USE_MP
    for(Component& comp : finalUVComponents)
    {
        Eigen::MatrixXd UVc;
        Eigen::MatrixXd Vc;Eigen::MatrixXi Fc;

        ExtractSubmesh(comp.faces, F, V, Fc, Vc);
        EASY_BLOCK("Unwrapin BB", profiler::colors::Brown);
        auto start = std::chrono::high_resolution_clock::now();
        int lscm_success = unwrap(Vc, Fc, UVc);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        EASY_END_BLOCK; 

        // Write to log file
        if(CONFIG_saveStuff){
            std::ofstream logfile("abf_timings.txt", std::ios::app);
            if (logfile.is_open()) {
                logfile << Vc.rows() << ", " << Fc.rows() << ", " << "Six part" <<  " : " << elapsed.count() << "\n";
            }
        }

        // std::cout << "number of vertices in the submesh: " << Vc.rows() << std::endl;
        // std::cout << "number of faces in the submesh: " << Fc.rows() << std::endl;
        if(lscm_success != 0){
            if(CONFIG_verbose)
                std::cout << "LSCM projection failed in UnwrapBB for component: " << comp.index << std::endl;
            return {};
        }else if (check_overlap){
            std::vector<std::pair<int, int>> overlap_triangles;
            auto num_overlaps = computeOverlapingTrianglesFast(UVc, Fc, overlap_triangles);
            if (num_overlaps > 0){

                if(CONFIG_verbose) std::cout << overlap_triangles.size() <<  " Overlapping triangles found in UnwrapBB for component: " << comp.index << std::endl;
                // int folder_file_count = std::distance(std::filesystem::directory_iterator("/ariesdv0/zhaoning/workspace/IUV/lscm/libigl-example-project/meshes/overlapped_meshes_2"), std::filesystem::directory_iterator{});
                // igl::write_triangle_mesh("/ariesdv0/zhaoning/workspace/IUV/lscm/libigl-example-project/meshes/overlapped_meshes_2/mesh_" + std::to_string(folder_file_count) + ".obj", V, F);
                
                return {};
            }
        }


        comp.V = Vc;
        comp.F = Fc;
        comp.UV = UVc;
        comp.distortion = calculate_distortion_area(Vc, Fc, UVc);

    }

    #else
        // An atomic flag to signal an error has occurred.
        std::atomic<bool> hasError(false);

        // Parallel loop over the components.
        // Use a dynamic schedule if iterations vary in workload.
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < static_cast<int>(finalUVComponents.size()); ++i)
        {
            // If another thread has already flagged an error, you might choose to skip processing.
            if (hasError.load(std::memory_order_acquire))
                continue;
    
            Component& comp = finalUVComponents[i];
            Eigen::MatrixXd UVc;
            Eigen::MatrixXd Vc;
            Eigen::MatrixXi Fc;
    
            // Extract the submesh for this component.
            ExtractSubmesh(comp.faces, F, V, Fc, Vc);
    
            // Perform the LSCM projection.
            // // writetotext("Unwrap BB: part " + std::to_string(i) + " With " + std::to_string(Vc.rows()) + " vertices and " + std::to_string(Fc.rows()) + " faces");
            int lscm_success = unwrap(Vc, Fc, UVc);
            if(lscm_success != 0)
            {
                // Use a critical section for output and updating the error flag.
                #pragma omp critical
                {
                    if(CONFIG_verbose)
                    std::cerr << "LSCM projection failed in UnwrapBB for component: " << comp.index << std::endl;
                    hasError.store(true, std::memory_order_release);
                }
                continue; // Optionally, skip further processing in this iteration.
            }
            else if (check_overlap)
            {
                std::vector<std::pair<int, int>> overlap_triangles;
                auto num_overlaps = computeOverlapingTrianglesFast(UVc, Fc, overlap_triangles);
                if (num_overlaps > 0)
                {
                    #pragma omp critical
                    {
                        if(CONFIG_verbose)
                            std::cerr << overlap_triangles.size() << " Overlapping triangles found in UnwrapBB for component: " 
                                    << comp.index << std::endl;
                        hasError.store(true, std::memory_order_release);
                    }
                    continue;
                }
            }
    
            // Only update the component if no error occurred.
            comp.V = Vc;
            comp.F = Fc;
            comp.UV = UVc;
            comp.distortion = calculate_distortion_area(Vc, Fc, UVc);
        }
    
        // Check if any error occurred during the parallel processing.
        if (hasError.load(std::memory_order_acquire))
        {
            // Handle the error: here we return an empty vector (or you could throw an exception).
            return {};
        }
    #endif

    return finalUVComponents;
}





