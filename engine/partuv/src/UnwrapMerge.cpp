#include <igl/read_triangle_mesh.h>
#include <igl/write_triangle_mesh.h>
#include <igl/per_face_normals.h>
#include <igl/adjacency_list.h>
#include <igl/lscm.h>


// LSCM
#include "Mesh.h"
#include "FormTrait.h"
#include "LSCM.h"
#include "Component.h"
#include "UnwrapMerge.h"
#include "UnwrapBB.h"

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


#include <omp.h>
#include <functional>
#include <optional> 

#include "UnwrapBB.h"
#include "stream_pool.h"

#include "merge.h"
#include "IO.h"
#include "UnwrapBB.h"
#include <filesystem>
namespace fs = std::filesystem;



#include <easy/profiler.h>







using namespace MeshLib;
/*
 * @brief Merges Component B into Component A with distortion and overlap checks.
 * @param A Reference to Component A which will receive Component B.
 * @param B Reference to Component B to be merged into Component A.
 * @param distortion_threshold  Defaults to 1.0.(NO threshold)
 * @return int Returns 0 if the merge is successful, -1 otherwise.
 */

int merge_B_to_A(Component &A, const Component &B, double distortion_threshold)
{
    Eigen::MatrixXd VA = A.V; 
    Eigen::MatrixXd VB = B.V; 
    Eigen::MatrixXi FA = A.F; 
    Eigen::MatrixXi FB = B.F;


    Component retComp = A;
    Eigen::MatrixXd UVc;
    auto start_merge = std::chrono::high_resolution_clock::now();
    int numNewVertices = merge_mesh_B_to_A(A, B, retComp);
    #ifdef ENABLE_PROFILING
    auto end_merge = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_merge = end_merge - start_merge;
    std::cout << "merge_mesh_B_to_A took " << duration_merge.count() << " seconds." << std::endl;

    auto start_lscm = std::chrono::high_resolution_clock::now();
    #endif
    std::string profile_log = "merging " + std::to_string(B.index) + " -> " + std::to_string(A.index) ;
    EASY_BLOCK(profile_log, profiler::colors::Magenta);
    auto start = std::chrono::high_resolution_clock::now();

    Eigen::MatrixXd V_simp = retComp.V;
    Eigen::MatrixXi F_simp = retComp.F;

    int lscm_success = unwrap_pamo(V_simp, F_simp, UVc);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    // Write to log file
    if(CONFIG_saveStuff){
        std::ofstream logfile("abf_timings.txt", std::ios::app);
        if (logfile.is_open()) {
            logfile << retComp.V.rows() << ", " << retComp.F.rows() << ", " << "Merge" <<  " : " << elapsed.count() << "\n";
        }
    }
    EASY_END_BLOCK; 
    #ifdef ENABLE_PROFILING
    auto end_lscm = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_lscm = end_lscm - start_lscm;
    std::cout << "unwrap took " << duration_lscm.count() << " seconds." << std::endl;
    auto start_overlap = std::chrono::high_resolution_clock::now();
    #endif
    double num_overlaps;
    if(lscm_success == 0)
    {   std::vector<std::pair<int, int>> overlap_triangles;
        num_overlaps = computeOverlapingTrianglesFast(UVc, F_simp, overlap_triangles);
    }
    #ifdef ENABLE_PROFILING
    auto end_overlap = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_overlap = end_overlap - start_overlap;
    std::cout << "computeOverlapingTrianglesFast took " << duration_overlap.count() << " seconds." << std::endl;
    #endif

    double distortion;
    bool not_enough_vertices = B.V.rows() - numNewVertices< 2;

    #ifdef VERBOSE
        if (not_enough_vertices){
            std::cout << "not enough common vertices between " << A.index << " and " << B.index << std::endl;
        }
    #endif
    // if overlap, or no common edge between original A and B (meaning B is sharing with appendix of A, and A is updated in adjList from appendix operation, 
    // in this case we always merge all)
    if(lscm_success!=0 || num_overlaps > 0 || not_enough_vertices ){
        #ifdef VERBOSE
            std::cout << "merging " << B.index << " to " << A.index << " with all A vertices " << std::endl;
        #endif

        retComp = A;
        merge_mesh_B_to_A(A, B, retComp, true);
        Eigen::MatrixXd UVc;
        std::string profile_log = " 2. merging " + std::to_string(B.index) + " -> " + std::to_string(A.index) ;
        EASY_BLOCK(profile_log, profiler::colors::Magenta);
        auto start = std::chrono::high_resolution_clock::now();
        Eigen::MatrixXd V_simp = retComp.V;
        Eigen::MatrixXi F_simp = retComp.F;

        int lscm_success = unwrap_pamo(V_simp, F_simp, UVc);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
    
        // Write to log file
        if(CONFIG_saveStuff){
            std::ofstream logfile("abf_timings.txt", std::ios::app);
            if (logfile.is_open()) {
                logfile << retComp.V.rows() << ", " << retComp.F.rows() << ", " << "Merge" <<  " : " << elapsed.count() << "\n";
            }
        }
        EASY_END_BLOCK; 
        if(lscm_success != 0){
            if(CONFIG_verbose)
                std::cout << "lscm failed after all A" << std::endl;
            return -1;
        }
        std::vector<std::pair<int, int>> overlap_triangles;
        num_overlaps = computeOverlapingTrianglesFast(UVc, F_simp, overlap_triangles);

        distortion = calculate_distortion_area(V_simp, F_simp, UVc);

        if(num_overlaps > 0 || distortion > distortion_threshold) {
            #ifdef VERBOSE
                        std::cout << "num_overlaps: " << num_overlaps << std::endl;
                        std::cout << "distortion: " << distortion << std::endl;
            #endif

            return -1;
        }
    }
    else{
        distortion = calculate_distortion_area(V_simp, F_simp, UVc);
        #ifdef VERBOSE
            std::cout << "merging " << B.index << " to " << A.index << " with partial A vertices " << std::endl;
        #endif

        if(distortion > distortion_threshold){
            #ifdef VERBOSE
            std::cout << "distortion occurs after all B" << "distortion: " << distortion << std::endl;
            #endif
            return -1;
        }
    }

    // Merge successful
    if(UVc.rows() == 0){
        std::cout << "UVc is empty" << std::endl;
        return -1;
    }
    retComp.UV = UVc;
    retComp.distortion = distortion;

    A = retComp;

    return A.index;
}
 

void clean_up_neighbors(std::vector<std::vector<std::pair<int,double>>> &compAdjList, Component &standbyComp, int success_to_index)
{
        // Replace all references to standbyComp.index with success_to_index


        for (size_t comp_index = 0; comp_index < compAdjList.size(); ++comp_index) {
            bool has_success = false;
            for (const auto &neighbor : compAdjList[comp_index]) {
                if (neighbor.first == success_to_index) {
                    has_success = true;
                    break;
                }
            }
            // if(! has_success) continue;
            for (auto it = compAdjList[comp_index].begin(); it != compAdjList[comp_index].end(); ) {
                if (it->first == standbyComp.index) {
                    if (comp_index == success_to_index || has_success) {
                        #ifdef VERBOSE
                        std::cout << "erased " << it->first << " from Row " << comp_index << std::endl;
                        #endif
                        it = compAdjList[comp_index].erase(it);
                    } else {
                        it->first = success_to_index;
                        break;
                    }
                } else {
                    ++it;
                }
            }
        }
}





int reassignFace(int f, const std::vector<std::vector<int>> &faceAdj,
                  std::vector<int> &faceToComp,  std::vector<std::vector<int>> &components, Tree* tree)
{
    int oldComp = faceToComp[f];

    // Count how many neighbors belong to each different component.
    // Since there are at most 3 neighbors, we can simply store them in an array.
    // Key: we only care if any *other* component ( != oldComp ) appears >= 2 times.
    // We'll track:
    //   - the ID of the first "other" component we see
    //   - how many times it appears
    //   - whether we see a second "other" component

    int compA = -1; // first other component encountered
    int compB = -1; // second other component encountered
    int countA = 0;
    bool multipleOthers = false; // did we encounter more than one other component?

    int face_to_reassign = -1;

    // used for recusive updating when reassigning under : 1 common and 2 different neighbors
    int neighbor_same = -1;

    for (int nbr : faceAdj[f])
    {
        int nbrComp = faceToComp[nbr];
        if (nbrComp == oldComp){
            neighbor_same = nbr;
            continue; 
        }


        // If we haven't seen any other component yet
        if (compA < 0)
        {
            compA = nbrComp;
            countA = 1;
        }
        else if (nbrComp == compA)
        {
            countA++;
            face_to_reassign = nbr;
        }
        else if (compB < 0)
        {
            compB = nbrComp;
        }
        else if (nbrComp == compB)
        {
            compA = nbrComp;
            countA = 2;
            face_to_reassign = nbr;
        }
        else
        {
            // We found a different "other" component than compA and B
            multipleOthers = true;
            break;
        }
    }

    // Decide whether to reassign:
    // Conditions to reassign:
    //   1) We found at least one other component (compA != -1),
    //   2) That same component appears at least 2 times (countA >= 2),
    //   3) We did *not* find a second different other component (multipleOthers == false).
    if (compA != -1 && countA >= 2 && !multipleOthers)
    {
        components[oldComp].erase(std::remove(components[oldComp].begin(), components[oldComp].end(), f), components[oldComp].end());
        components[compA].push_back(f);
        faceToComp[f] = compA;
        if(tree != nullptr)
            reassign_face_in_tree(*tree, f, face_to_reassign);
        return neighbor_same;
    }
    else
    {
        return -1;
    }
}
/// Reassign faces that have at least two neighbors in exactly one other component.
/// - `components[c]` is a list of face indices belonging to component `c`.
/// - `faceAdj[f]` is a list (size <= 3) of faces adjacent to face `f`.
void smoothComponentEdge(
    std::vector<std::vector<int>>& components,
    const std::vector<std::vector<int>>& faceAdj,
    Tree *tree)
{
    // --- 1) Build a face -> component map for quick lookup ---
    int nFaces = static_cast<int>(faceAdj.size());
    std::vector<int> faceToComp(nFaces, -1);

    // Record how many components we currently have
    int numComponents = static_cast<int>(components.size());

    for (int c = 0; c < numComponents; ++c)
    {
        for (int face : components[c])
        {
            if(face >= nFaces){
                std::cout << "face: " << face << std::endl;
            }
            faceToComp[face] = c;
        }
    }


    // Concatenate all components into a single list of faces
    // so that the processing is per-component
    std::vector<int> allFaces;
    allFaces.reserve(nFaces);
    for (const auto &compVec : components)
    {
        allFaces.insert(allFaces.end(), compVec.begin(), compVec.end());
    }

    for (int f : allFaces)
    {
        int neighbor_same = f;
        do{
            f = neighbor_same;
            neighbor_same = reassignFace(f, faceAdj, faceToComp, components, tree);
        }while (neighbor_same != -1);

        
    }

}


std::vector<Component> unwrap_aligning_merge(const Eigen::MatrixXd &V,   const  Eigen::MatrixXi &F, double threshold, bool check_overlap, int chart_limit){

    MatrixX3R FN;
    std::vector<int> faceAssignment;
    std::vector<std::vector<int>> faceAdj;

    // Prepare data for OBB alignment
    Eigen::MatrixXd V_rotated = V;
    std::vector<std::vector<double>> edge_lengths;
    prepareOBBData(V_rotated, F, FN, faceAssignment, faceAdj, edge_lengths);

    // 6) For each of the 6 directions, gather connected components of faces
    //    We'll store them in finalUVComponents (largest one) + standbyUVComponents (remaining)
    // each entry is a chart

    double total_elapsed_here = total_time_spent_lscm;
    std::vector<std::vector<int>> components;
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
        std::vector<bool> globalVisited(F.rows(), false);
        for(int i = 0; i < F.rows(); i++)
        {
            if(!visitedMask[i] && !globalVisited[i])
            {
                std::vector<int> comp = findConnectedComponent(i, faceAdj, visitedMask, globalVisited);
                if(!comp.empty())
                    components.push_back(comp);
            }
        }

    }

    smoothComponentEdge(components, faceAdj);
    
    return merge_components(components, faceAdj,  V, F, FN, chart_limit,check_overlap, threshold,edge_lengths);
}

std::vector<Component> merge_components(
    const std::vector<std::vector<int>> &components,
    const std::vector<std::vector<int>> faceAdj,
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const MatrixX3R &FN,
    const int chart_limit,
    const bool check_overlap,
    const double threshold,
    const std::vector<std::vector<double>> &edge_lengths)
    { 
    std::vector<Component> allComponents;  
    std::map<int, int> faceToComponent;
    std::vector<Component*> finalUVComponents;    // each entry is a chart


    
    for(const auto &comp : components)
    {
        Eigen::MatrixXd curr_FN(comp.size(), FN.cols());
        for (int i = 0; i < comp.size(); ++i) {
            curr_FN.row(i) = FN.row(comp[i]);
        }
        
        Eigen::MatrixXd Vc;Eigen::MatrixXi Fc;
        ExtractSubmesh(comp, F, V, Fc, Vc);

        Component curr_comp = Component(allComponents.size(), comp, -1, curr_FN, Fc,Vc);
        if(!comp.empty()) 
            allComponents.push_back(curr_comp);

    }

    int numComponents = allComponents.size();


    for (const auto &component : allComponents)
    {
        for (int faceIdx : component.faces)
        {
            faceToComponent[faceIdx] = component.index;
        }
    }

    // Now build adjacency between these components
    auto compAdjList = buildComponentAdjacencyEdgeLength(faceAdj, faceToComponent,allComponents, numComponents, edge_lengths);
    // std::cout << "total components at starting point: " << numComponents << std::endl;

    #ifdef VERBOSE
    int row_num = 0;
    for ( const auto &row : compAdjList )
    {
        std::cout << "Row: " << row_num++ << ": ";
       for ( const auto &s : row ){  std::cout << "(" << s.first << "," << s.second << ") ";}
       std::cout << std::endl;  
    }
    #endif


    std::vector<int> sortedIndices(allComponents.size());
    std::iota(sortedIndices.begin(), sortedIndices.end(), 0);  // Fill with 0,1,2,...
    
    std::sort(sortedIndices.begin(), 
              sortedIndices.end(),
              [&allComponents](int a, int b)
              {
                  // Smallest first
                  return allComponents[a].faces.size() < allComponents[b].faces.size();
              });
#ifdef VERBOSE
    std::cout << "allComponents.size(): " << allComponents.size() << std::endl;
    for (int idx : sortedIndices)
    {
        Component &comp = allComponents[idx];
        std::cout << "comp.index: " << comp.index << std::endl;
        std::cout << "comp.faces.size(): " << comp.faces.size() << std::endl;
        std::cout << "comp.vertices.size(): " << comp.V.rows() << std::endl;
    }

    for(Component comp : allComponents)
    {
        std::cout << "component index: " << comp.index << std::endl;
        std::cout << "with faces: " << comp.faces.size() << std::endl;
        std::cout << "with vertices: " << comp.V.rows() << std::endl;
    }
#endif
    static double total_elapsed = 0.0;
    auto start = std::chrono::high_resolution_clock::now();



    for (int idx : sortedIndices)
    {

        Component &standbyComp = allComponents[idx];

        Eigen::RowVector3d avg = standbyComp.face_normals.colwise().mean();
        avg.normalize();
        

        
        std::vector<int> neighbors;
        neighbors.reserve(compAdjList[standbyComp.index].size()); // optional but can help performance
        for (int i = 0; i < compAdjList[standbyComp.index].size(); i++) {
            neighbors.push_back(i);
        }

        std::sort(neighbors.begin(), neighbors.end(), [&](int a, int b) {
            if (compAdjList[standbyComp.index][a].second == compAdjList[standbyComp.index][b].second){
                return allComponents[compAdjList[standbyComp.index][a].first].faces.size() < allComponents[compAdjList[standbyComp.index][b].first].faces.size();
            }
            return  compAdjList[standbyComp.index][a].second > compAdjList[standbyComp.index][b].second; // sort by edge length
        });


        #ifdef VERBOSE
        std::cout << "Neighbors of component " << standbyComp.index << ": ";
        for (int neighbor : neighbors) {
            std::cout << '(' << compAdjList[standbyComp.index][neighbor].first << ", " << compAdjList[standbyComp.index][neighbor].second << ") ";
        }
        std::cout << std::endl;


        #endif

        int success_to_index = -1;
        for (int neighbor : neighbors)
        {

            auto start_time = std::chrono::high_resolution_clock::now();
            success_to_index = merge_B_to_A(allComponents[compAdjList[standbyComp.index][neighbor].first], standbyComp, threshold);
            auto end_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end_time - start_time;
            total_elapsed += elapsed.count();

            if (success_to_index != -1) break;

        }

        if(success_to_index == -1){
            #ifdef VERBOSE
                        std::cout << "Push component to final:" << standbyComp.index << " to Part "<< finalUVComponents.size() << std::endl;
            #endif
            finalUVComponents.push_back(&standbyComp);
            if (chart_limit !=   std::numeric_limits<int>::min() && finalUVComponents.size() > chart_limit){
                return {};
            }
        }else{

            clean_up_neighbors(compAdjList, standbyComp, success_to_index);
        }


    }
#ifdef ENABLE_PROFILING
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Merging components took " << elapsed.count() << " seconds." << std::endl;
    std::cout << "Total elapsed time: " << total_elapsed << " seconds." << std::endl;
#endif

bool hasError = false;
#ifndef USE_MP
    std::vector<Component> retFinalUVComponents(finalUVComponents.size()); 

    // for(Component* comp : finalUVComponents){
    // #pragma omp parallel for shared(finalUVComponents, retFinalUVComponents) 
    for (int i = 0; i < static_cast<int>(finalUVComponents.size()); ++i)
    {
        if (hasError) continue; // Skip if an error has occurred
        Component* comp = finalUVComponents[i];

        Eigen::MatrixXd UVc;

        EASY_BLOCK("final unwrap in merge", profiler::colors::Magenta);
        int lscm_success = unwrap(comp->V, comp->F,  UVc);
        EASY_END_BLOCK;

        if(lscm_success != 0){
            if(CONFIG_verbose)
                std::cout << "LSCM projection failed in UnwrapBB for component: " << comp->index << std::endl;
            hasError = true;
            continue; // Optionally, skip further processing in this iteration.

        }else if (check_overlap){
            std::vector<std::pair<int, int>> overlap_triangles;
            auto num_overlaps = computeOverlapingTrianglesFast(UVc, comp->F, overlap_triangles);
            if (num_overlaps > 0){
                if(CONFIG_verbose) std::cout << overlap_triangles.size() <<  " Overlapping triangles found in Unwrap Merge for component: " << comp->index << std::endl;
                hasError = true;
                continue;
            }
        }

        comp->UV = UVc;
        comp->distortion = calculate_distortion_area(comp->V, comp->F, UVc);

        retFinalUVComponents[i] = *comp;  // Copy the object.

    }
#else
    std::atomic<bool> hasError(false);

    // Parallel loop over the components.
    // Use a dynamic schedule if iterations vary in workload.
    // retFinalUVComponents = finalUVComponents;
    for (Component* compPtr : finalUVComponents) {
        if (compPtr != nullptr) {  // Always good to check for nullptr
            retFinalUVComponents.push_back(*compPtr);  // Copy the object.
        }
    }

    // Parallel loop over the components.
    // Use a dynamic schedule if iterations vary in workload.
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < static_cast<int>(finalUVComponents.size()); ++i)
    {
        // If another thread has already flagged an error, you might choose to skip processing.
        if (hasError.load(std::memory_order_acquire))
            continue;

        Component& comp = retFinalUVComponents[i];
        Eigen::MatrixXd UVc;
        Eigen::MatrixXd Vc;
        Eigen::MatrixXi Fc;

        // Extract the submesh for this component.
        ExtractSubmesh(comp.faces, F, V, Fc, Vc);

        // Perform the LSCM projection.
        int lscm_success = unwrap(Vc, Fc, UVc);
        if(lscm_success != 0)
        {
            // Use a critical section for output and updating the error flag.
            #pragma omp critical
            {
                if(CONFIG_verbose)
                    std::cout << "LSCM projection failed in UnwrapBB for component: " << comp.index << std::endl;
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

    return hasError ? std::vector<Component>() : retFinalUVComponents;

}





// still developping
std::vector<Component> merge_components_parallel(
    const std::vector<std::vector<int>> &components,
    const std::vector<std::vector<int>> faceAdj,
    const Eigen::MatrixXd &V,
    const Eigen::MatrixXi &F,
    const MatrixX3R &FN,
    const int chart_limit,
    const bool check_overlap,
    const double threshold,
    const std::vector<std::vector<double>> &edge_lengths)
    { 
    std::vector<Component> allComponents;  
    std::map<int, int> faceToComponent;
    std::vector<Component*> finalUVComponents;    // each entry is a chart


    
    for(const auto &comp : components)
    {
        Eigen::MatrixXd curr_FN(comp.size(), FN.cols());
        for (int i = 0; i < comp.size(); ++i) {
            curr_FN.row(i) = FN.row(comp[i]);
        }
        
        Eigen::MatrixXd Vc;Eigen::MatrixXi Fc;
        ExtractSubmesh(comp, F, V, Fc, Vc);

        Component curr_comp = Component(allComponents.size(), comp, -1, curr_FN, Fc,Vc);
        if(!comp.empty()) 
            allComponents.push_back(curr_comp);

    }

    int numComponents = allComponents.size();


    for (const auto &component : allComponents)
    {
        for (int faceIdx : component.faces)
        {
            faceToComponent[faceIdx] = component.index;
        }
    }

    // Now build adjacency between these components
    auto compAdjList = buildComponentAdjacencyEdgeLength(faceAdj, faceToComponent,allComponents, numComponents, edge_lengths);
    // std::cout << "total components at starting point: " << numComponents << std::endl;

    #ifdef VERBOSE
    int row_num = 0;
    for ( const auto &row : compAdjList )
    {
        std::cout << "Row: " << row_num++ << ": ";
       for ( const auto &s : row ){  std::cout << "(" << s.first << "," << s.second << ") ";}
       std::cout << std::endl;  
    }
    #endif


    std::vector<int> sortedIndices(allComponents.size());
    std::iota(sortedIndices.begin(), sortedIndices.end(), 0);  // Fill with 0,1,2,...
    
    std::sort(sortedIndices.begin(), 
              sortedIndices.end(),
              [&allComponents](int a, int b)
              {
                  // Smallest first
                  return allComponents[a].faces.size() < allComponents[b].faces.size();
              });



    static double total_elapsed = 0.0;
    auto start = std::chrono::high_resolution_clock::now();




    int num_chart_handled = 0;
    while(num_chart_handled < allComponents.size()){

        std::unordered_set<int>  neighbour_taken;
        std::vector<std::pair<int,int>> mergePairs;           // (src , dst)
        for (int idx : sortedIndices)
        {
            Component &standbyComp = allComponents[idx];

            if(neighbour_taken.find(standbyComp.index) != neighbour_taken.end()) continue;
            neighbour_taken.insert(standbyComp.index);
            // Compute average normal of standbyComp
            // Eigen::RowVector3d avg = standbyComp.face_normals.colwise().mean();
            // avg.normalize();
            

            // Instead of sorting all neighbors, find the best one using std::max_element
            
            int best_neighbor = -1;
            if (!compAdjList[standbyComp.index].empty()) {

               best_neighbor = compAdjList[standbyComp.index].begin()->first;
                for (const auto &edge : compAdjList[standbyComp.index]){
                    if (edge.second > compAdjList[standbyComp.index][best_neighbor].second){
                        best_neighbor = edge.first;
                    }
                    if (edge.second == compAdjList[standbyComp.index][best_neighbor].second
                    && allComponents[edge.first].faces.size() > allComponents[best_neighbor].faces.size()){
                        best_neighbor = edge.first;
                    }
                }
                if (neighbour_taken.find(best_neighbor) == neighbour_taken.end()){
                    mergePairs.emplace_back(standbyComp.index, best_neighbor);
                    std::cout << "pushed merge pairs: " << standbyComp.index << " -> " << best_neighbor << std::endl;
                    for (int i = 0; i < compAdjList[standbyComp.index].size(); i++) {
                        neighbour_taken.insert(compAdjList[standbyComp.index][i].first);
                    }
                }else{
                    continue;
                }

            }else{

                #ifdef VERBOSE
                std::cout << "No neighbors found for component " << standbyComp.index << std::endl;
                std::cout << "Push component to final:" << standbyComp.index << " to Part "<< finalUVComponents.size() << std::endl;
                #endif
                sortedIndices.erase(std::remove(sortedIndices.begin(), sortedIndices.end(), standbyComp.index), sortedIndices.end());
                #ifdef VERBOSE
                std::cout << "Current sortedIndices: ";
                for (int idx : sortedIndices) {
                    std::cout << idx << " ";
                }
                std::cout << std::endl;
                #endif
                finalUVComponents.push_back(&standbyComp);
                num_chart_handled++;
                if (chart_limit !=   std::numeric_limits<int>::min() && finalUVComponents.size() > chart_limit){
                    return {};
                }
                continue;
            }

        }


        // print mergePairs
        std::cout << "mergePairs: " << std::endl;
        for (const auto &pair : mergePairs) {
            std::cout << "(" << pair.first << ", " << pair.second << ")" << std::endl;
        }
        #pragma omp parallel for 
        for (std::size_t k = 0; k < mergePairs.size(); ++k)
        {
            auto [src, dst] = mergePairs[k];          // src → dst
            int succ = merge_B_to_A(allComponents[dst],allComponents[src],  threshold);            
            #pragma omp critical
            {
                if (succ != -1)
                {
                    sortedIndices.erase(std::remove(sortedIndices.begin(), sortedIndices.end(), src), sortedIndices.end());
                    /*  Critical zone: mark src dead & patch adjacency tables   */
                    clean_up_neighbors(compAdjList,
                                        allComponents[src],
                                        dst);
                    num_chart_handled++;
                }else{
                    // remove neighbor from compAdjList
                    compAdjList[src].erase(std::remove_if(compAdjList[src].begin(), compAdjList[src].end(), 
                                                        [&](const std::pair<int, double>& p) {
                                                            return p.first == dst;
                                                        }), compAdjList[src].end());
                }
            }
        }
    }

    std::vector<Component> retFinalUVComponents(finalUVComponents.size()); 


    std::atomic<bool> hasError(false);

    #pragma omp parallel for shared(finalUVComponents, retFinalUVComponents) 
    for (int i = 0; i < static_cast<int>(finalUVComponents.size()); ++i)
    {
        if (hasError) continue; // Skip if an error has occurred
        Component* comp = finalUVComponents[i];

        Eigen::MatrixXd UVc;
        EASY_BLOCK("final unwrap in merge", profiler::colors::Magenta);
        int lscm_success = unwrap(comp->V_original, comp->F_original,  UVc);
        EASY_END_BLOCK;
        if(lscm_success != 0){
            if(CONFIG_verbose)
                std::cout << "LSCM projection failed in UnwrapBB for component: " << comp->index << std::endl;
            hasError = true;
            continue; // Optionally, skip further processing in this iteration.

        }else if (check_overlap){
            std::vector<std::pair<int, int>> overlap_triangles;
            auto num_overlaps = computeOverlapingTrianglesFast(UVc, comp->F, overlap_triangles);
            if (num_overlaps > 0){
                if(CONFIG_verbose) std::cout << overlap_triangles.size() <<  " Overlapping triangles found in Unwrap Merge for component: " << comp->index << std::endl;
                hasError = true;
                continue;
            }
        }
        comp->UV = UVc;
        comp->distortion = calculate_distortion_area(comp->V, comp->F, UVc);

        retFinalUVComponents[i] = *comp;  // Copy the object.

    }
    return hasError ? std::vector<Component>() : retFinalUVComponents;
}

