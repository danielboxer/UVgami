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
#include "UnwrapPlane.h"
#include "UnwrapAgg.h"

#include "pipeline.h"

#include "Component.h"

#include "merge.h"
#include "IO.h"
#include "Config.h"


#include "AgglomerativeClustering.h"

#include <omp.h>
// omp_set_nested(1);  

#include <easy/profiler.h>


using namespace MeshLib;



// Returns a grouping of faces in `finalUVComponents`, where each entry is a list of face indices
// std::vector<std::vector<int>> unwrap_aligning_BB( Eigen::MatrixXd &V,  Eigen::MatrixXi &F)
std::vector<Component> unwrap_aligning_Agg_helper( const Eigen::MatrixXd &V, const Eigen::MatrixXi &F,double threshold, bool check_overlap, int chart_limit, int num_cluster)
{
    EASY_BLOCK("unwrap_aligning_Agg_helper", profiler::colors::Purple);
    std::filesystem::path meshFilename(CONFIG_meshPath);
    meshFilename.replace_filename(std::string("tempAGG_") + std::to_string(num_cluster) + ".obj");
    std::string mesh_filename = meshFilename.string();
    if (!igl::write_triangle_mesh(mesh_filename, V, F)) {
         std::cerr << "Failed to write mesh file: " << mesh_filename << std::endl;
         return std::vector<Component>();  // Alternatively, throw an exception.
    }
        // Step 2: Call the Python script.
    // The command is constructed to pass the mesh file path to the Python script.
    std::string command = "python /ariesdv0/zhaoning/workspace/IUV/lscm/libigl-example-project/AgglomerativeClustering.py  --no_save_mesh --mesh_path " + mesh_filename;
    int ret = std::system(command.c_str());
    if (ret != 0) {
         std::cerr << "Python script failed with return code " << ret << std::endl;
         return std::vector<Component>();  // Alternatively, throw an exception.
    }
    std::string bin_filename = mesh_filename.substr(0, mesh_filename.find_last_of('.')) + ".bin";
    
    std::cout << "Loading tree from: " << bin_filename << std::endl;
    Tree agglomerative_tree(bin_filename);
    
    validate_tree_leaves(agglomerative_tree, F.rows());

    int max_cluster = 20;
    auto labels = hierarchical_clustering_labels(agglomerative_tree, max_cluster);


    std::vector<std::vector<int>>  faces = group_samples_by_label(labels[max_cluster-num_cluster]);
    std::vector<std::vector<double>> edge_lengths;
    std::vector<std::vector<int>> faceAdj = computeFaceAdjacency(F, V, edge_lengths);

    smoothComponentEdge(faces, faceAdj);

    // std::cout << "Number of components: " << faces.size() << std::endl;
    std::vector<Component> ret_comp(faces.size());
    
     
    for (size_t i = 0; i < faces.size(); ++i) {
        if (process_submesh(faces[i], V, F, i, ret_comp[i], check_overlap) != 0) {
            // std::cerr << "Failed to process submesh for component: " << i << std::endl;
            return std::vector<Component>(); // Alternatively, throw an exception.
        }
    }
    
    // remove empty component
    ret_comp.erase(std::remove_if(ret_comp.begin(), ret_comp.end(), [](const Component& comp) {
        return comp.faces.empty();
    }), ret_comp.end());

    if(CONFIG_saveStuff){
        Component comp;
        for (int i = 0; i < ret_comp.size(); i++){
            comp = comp + ret_comp[i];
        }
        double distort =  calculate_distortion_area(comp.V , comp.F, comp.UV, & agglomerative_tree);
        agglomerative_tree.print_tree(3);
        std::cout << "distortion: " << distort << std::endl;
    }
    EASY_END_BLOCK;
    return ret_comp; // Placeholder for the actual return value
}

std::vector<Component> unwrap_aligning_Agglomerative_old(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, double threshold, bool check_overlap, int chart_limit)
{

     // We know the loop runs from i = 2 to i = 9, inclusive.
     const int start = 2;
     const int end   = 10; // Exclusive bound
 
     // Make a temporary array to hold results for each i.
     // This way, each index is written by only one thread.
     std::vector<UVParts> all_components(end - start);
 
     // Use OpenMP to parallelize the loop.
     #pragma omp parallel for
     for (int i = start; i < end; i++) {
         // Each iteration builds its own std::vector<Component>.
         std::vector<Component> components = unwrap_aligning_Agg_helper(V, F, threshold, check_overlap, chart_limit, i);
         // Store the result in the temp array at index (i - start).
         all_components[i - start] = std::move(UVParts(components));
     }
     
    UVParts best_part =  get_best_part(all_components, threshold, check_overlap);


    return best_part.components;


}

std::vector<Component> unwrap_aligning_Agglomerative(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, double threshold, bool check_overlap, int chart_limit){
    EASY_BLOCK("unwrap_aligning_Agg", profiler::colors::Purple);
    
    if((int)F.rows() <= 2){
        std::cerr << "Not enough faces to unwrap. Returning empty vector." << std::endl;
        return std::vector<Component>();
    }
    if(chart_limit == NO_CHART_LIMIT){
        chart_limit = CONFIG_unwrapAggParts;
    }else{
        chart_limit = std::min(chart_limit, CONFIG_unwrapAggParts);
    }
    int n_clusters = std::min((int)F.rows()-1, chart_limit);



    AgglomerativeClustering agglomerative_cluster(n_clusters);

    
    Eigen::MatrixXd normals;
    igl::per_face_normals(V, F, normals); 
    std::vector<std::vector<double>> edge_lengths;
    std::vector<std::vector<int>> adj = computeFaceAdjacency(F, V, edge_lengths);

    std::vector<std::vector<std::vector<int>>> h_labels =  agglomerative_cluster.fit(normals, adj);
    std::vector<UVParts> all_components(n_clusters - 1 );
    
    #pragma omp parallel for
    for(int i = 0; i < n_clusters - 1; i++){

        std::vector<std::vector<int>> faces = h_labels[i];

        smoothComponentEdge(faces, adj);
        std::vector<Component> ret_comp(faces.size());
        bool success = true;
        for (size_t j = 0; j < faces.size(); ++j) {
            if (process_submesh(faces[j], V, F, j, ret_comp[j], check_overlap) != 0) {
                // std::cerr << "Failed to process submesh for component: " << i << std::endl;
                // all_components[i] = UVParts();
                success = false;
                break;
            }
        }
        if(!success){
            continue;
        }
        ret_comp.erase(std::remove_if(ret_comp.begin(), ret_comp.end(), [](const Component& comp) {
            return comp.faces.empty();
        }), ret_comp.end());

        // std::cout << "Number of components: " << ret_comp.size() << std::endl;
        all_components[i] = UVParts(ret_comp);
    }
    UVParts best_part =  get_best_part(all_components, threshold, check_overlap);
    EASY_END_BLOCK;
    return best_part.components;

}



/**
 * @brief Unwraps a mesh using agglomerative clustering and returns the best component
 * 
 * This function performs agglomerative clustering on the input mesh to create multiple
 * partitions, unwraps each partition, and returns the components of the best partition
 * based on distortion threshold.
 * 
 * @param V Vertex positions of the input mesh
 * @param F Face indices of the input mesh
 * @param threshold Maximum allowed distortion
 * @param check_overlap Whether to check for overlaps in the parameterization
 * @param chart_limit Maximum number of charts to generate (NO_CHART_LIMIT for default)
 * @param check_break When this is toggled, we check threshold and overlap here and break early
 * 
 * @return A vector of Component objects representing the unwrapped mesh parts
 */

std::vector<UVParts> unwrap_aligning_Agglomerative_all(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, double threshold, bool check_overlap, int chart_limit, bool check_break){
    EASY_BLOCK("unwrap_aligning_Agg", profiler::colors::Purple);
    
    if((int)F.rows() < 1){
        std::cerr << "Not enough faces to unwrap. Returning empty vector." << std::endl;
        return std::vector<UVParts>();
    }
    if(chart_limit == NO_CHART_LIMIT){
        chart_limit = CONFIG_unwrapAggParts;
    }else{
        chart_limit = std::min(chart_limit, CONFIG_unwrapAggParts);
    }
    int n_clusters = std::min((int)F.rows()-1, chart_limit);



    AgglomerativeClustering agglomerative_cluster(n_clusters);

    
    Eigen::MatrixXd normals;
    igl::per_face_normals(V, F, normals); 
    std::vector<std::vector<double>> edge_lengths;
    std::vector<std::vector<int>> adj = computeFaceAdjacency(F, V, edge_lengths);

    std::vector<std::vector<std::vector<int>>> h_labels =  agglomerative_cluster.fit(normals, adj);
    std::vector<UVParts> all_components(h_labels.size()  );
    

    if(check_break){
        std::sort(h_labels.begin(), h_labels.end(), [](const std::vector<std::vector<int>>& a, const std::vector<std::vector<int>>& b) {
            return a.size() < b.size();
        });
        check_overlap = true;
    }
    // #pragma omp parallel for shared(h_labels,all_components) 
    for(int i = 0; i < h_labels.size() ; i++){

        std::vector<std::vector<int>> faces = h_labels[i];
        cudaStream_t stream;
        if (CONFIG_unwrapPamo){
            // stream = StreamPool::getStream();
            try{
                stream = StreamPool::getStream( i);
            }catch(const std::exception& e){
                std::cerr << "Can't get stream for component, are you trying to run pamo with a CPU node?" << std::endl;
                // stream = nullptr;
                throw std::runtime_error("Can't get stream for component, are you trying to run pamo with a CPU node?");
            }
        }else{
            stream = nullptr;
        }
        smoothComponentEdge(faces, adj);
        std::vector<Component> ret_comp(faces.size());
        bool success = true;
        for (size_t j = 0; j < faces.size(); ++j) {
            if (process_submesh(faces[j], V, F, j, ret_comp[j], check_overlap, stream) != 0) {
                // std::cerr << "Failed to process submesh for component: " << i << std::endl;
                // all_components[i] = UVParts();
                success = false;
                break;
            }
            
            if(check_break && ret_comp[j].distortion > threshold){
                success = false;
                break;
            }
        }
        if(!success){
            continue;
        }
        ret_comp.erase(std::remove_if(ret_comp.begin(), ret_comp.end(), [](const Component& comp) {
            return comp.faces.empty();
        }), ret_comp.end());


        if(check_break){
            return {UVParts(ret_comp)};
        }
        all_components[i] = UVParts(ret_comp);
    }
    EASY_END_BLOCK;
    return all_components;

}



std::vector<Component> unwrap_aligning_Agglomerative_merge(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, double threshold, bool check_overlap, int chart_limit){
    EASY_BLOCK("unwrap_aligning_Agg_merge", profiler::colors::Purple);
    
    if((int)F.rows() <= 2){
        std::cerr << "Not enough faces to unwrap. Returning empty vector." << std::endl;
        return std::vector<Component>();
    }
    // if(chart_limit == NO_CHART_LIMIT){
    //     chart_limit = 20;
    // }else{
    //     chart_limit = std::min(chart_limit, 20);
    // }
    int n_clusters = std::min((int)F.rows()-1, chart_limit);
    
    std::cout << "[INFO] Number of clusters: " << n_clusters << std::endl;
    


    AgglomerativeClustering agglomerative_cluster(n_clusters);

    
    Eigen::MatrixXd normals;
    igl::per_face_normals(V, F, normals); 
    std::vector<std::vector<double>> edge_lengths;
    std::vector<std::vector<int>> adj = computeFaceAdjacency(F, V, edge_lengths);
    std::vector<std::vector<std::vector<int>>> h_labels =  agglomerative_cluster.fit(normals, adj, n_clusters-1);
    std::vector<UVParts> all_components(n_clusters - 1 );
    
    // std::cout << "Number of components: " << h_labels[0].size() << std::endl;
    return merge_components(h_labels[0], adj, V, F, normals, chart_limit, check_overlap, threshold);


}




