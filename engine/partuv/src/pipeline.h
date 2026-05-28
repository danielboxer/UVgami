#ifndef UV_PIPELINE_H
#define UV_PIPELINE_H

#include <Eigen/Core>
#include <vector>
#include <string>
#include <unordered_map>
#include <limits>

#include "Component.h"

#include "IO.h"



#include <functional>

// constexpr double NEG_INF = -std::numeric_limits<double>::infinity();
static constexpr double NO_CHART_LIMIT = (double) std::numeric_limits<int>::min();
constexpr double POS_INF = std::numeric_limits<double>::infinity();

constexpr int MAX_CHARTS = std::numeric_limits<int>::max()/2;


/**
 * @brief A structure to a Segment, containing several charts/components.
 */
 struct UVParts {

    std::vector<Component> components;
    Hierarchy hierarchy;

    double distortion;

    int num_components;

    // Default constructor
    UVParts( const std::vector<Component> &components): components(components) {
        distortion = -1;
        for (const auto &comp : components) {
            distortion = std::max( distortion, comp.distortion );
        }

        if (std::isnan(distortion)) {
            distortion = std::numeric_limits<double>::max();
        }

        num_components = components.size(); 
    }

    // Constructor of dummy parts with num_components
    UVParts( int num_components): components({}), num_components(num_components) {
        distortion = static_cast<double>(std::numeric_limits<int>::max());
    }

    UVParts() : components{}, distortion(static_cast<double>(std::numeric_limits<int>::max())), num_components(0) {}



    UVParts operator+(const UVParts &other) const {
        // Create a new UVParts object to store the result
        UVParts result(components);

        // Concat the components
        // result.components = components; // Copy the first vector
        result.components.insert(result.components.end(), other.components.begin(), other.components.end());

        // Combine the distortion values using std::max
        result.distortion = std::max(distortion, other.distortion);

        // Update the number of parts
        result.num_components = num_components + other.num_components;

        
        return result;
    }

    Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor> getUV() const {
        // First, calculate the total number of rows needed.
        int totalRows = 0;
        for (const auto &comp : components) {
            totalRows += comp.UV.rows();
        }

        // Create the concatenated UV matrix with the total rows.
        Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor> allUV(totalRows, 2);
        
        // Fill in the matrix by copying each component's UV.
        int currentRow = 0;
        for (const auto &comp : components) {
            int numRows = comp.UV.rows();
            if (numRows > 0) {
                allUV.block(currentRow, 0, numRows, 2) = comp.UV;
                currentRow += numRows;
            }
        }
        return allUV;
    }


    int getNumFaces() const {
        int totalFaces = 0;
        for (const auto &comp : components) {
            totalFaces += comp.F.rows();
        }
        return totalFaces;
    }

    Component to_components() const
    {
        if (components.empty()) {
            // If there are no components, return a default (empty) Component
            return {};
        }

        // Start from the first component
        Component result = components[0];

        // Concatenate all subsequent components
        for (size_t i = 1; i < components.size(); ++i) {
            result = result + components[i];
        }
        return result;
    }

    bool operator==(const UVParts &other) const {
        if (num_components != other.num_components || distortion != other.distortion) {
            return false;
        }
        for (size_t i = 0; i < components.size(); ++i) {
            if (components[i].F.rows() != other.components[i].F.rows() || components[i].V.rows() != other.components[i].V.rows() || components[i].UV.rows() != other.components[i].UV.rows()) {
                return false;
            }
        }
        return true;
    }
    
    bool operator!=(const UVParts &other) const {
        return !(*this == other);
    }

};

#include "UnwrapBB.h"
#include "UnwrapMerge.h"
#include "UnwrapPlane.h"
#include "UnwrapOne.h"
#include "UnwrapAgg.h"


using UnwrapFunction = std::function<std::vector<Component>(const Eigen::MatrixXd &V, const Eigen::MatrixXi &F, double threshold, bool check_overlap, int chart_limit)>;
// Wrap each function with its corresponding name.
static const std::map<std::string, UnwrapFunction> available_unwrap_methods = {
    {"unwrap_aligning_one", unwrap_aligning_one},
    {"unwrap_aligning_plane", unwrap_aligning_plane},
    {"unwrap_aligning_BB", unwrap_aligning_BB},
    {"unwrap_aligning_merge", unwrap_aligning_merge},
    // {"unwrap_aligning_Agg", unwrap_aligning_Agglomerative},
};



UVParts get_best_part( std::vector<UVParts> all_candidates, double threshold, bool check_overlap, UVParts *best_part_in_list = nullptr, int *debug_index = nullptr);


UVParts get_best_part( std::vector<std::vector<UVParts>> all_candidates, double threshold, bool check_overlap, int* debug_index = nullptr, bool use_dummy_best_part = false);



/**
 * @brief Collects UV candidates for a submesh using different methods, filters them by distortion threshold,
 *        and returns the best candidate.
 * @param submesh   The mesh or submesh.
 * @param threshold The distortion threshold.
 * @return          UVParts for the chosen candidate.
 */



std::vector<std::vector<UVParts>>  get_uv_wrapper( const Eigen::MatrixXi &F,const Eigen::MatrixXd &V, double threshold,bool check_overlap, bool use_full, int chart_limit);



// UVParts get_uv( const Eigen::MatrixXi &F,const Eigen::MatrixXd &V, double threshold,bool check_overlap, bool use_full = false);
std::vector<UVParts> get_uv( const Eigen::MatrixXi &F,const Eigen::MatrixXd &V, double threshold,bool check_overlap, bool use_full, int chart_limit);

void mock_pipeline(const std::string &mesh_filename, double threshold);

void get_individual_parts(std::vector<UVParts> &individual_parts);

UVParts pipeline_helper(std::vector<int> leaves, Tree tree, int root, double chart_limit, int stack_level = 0);

/**
 * @brief The main pipeline function, which recursively processes a tree of submeshes
 *        and returns their combined UVParts.
 * @param root        The current node index.
 * @param tree        The entire tree structure, mapping node -> {"left", "right"} children.
 * @param threshold   Distortion threshold to decide how to handle partitioning.
 * @param chart_limit Optional integer limiting how many charts we allow in total. Default is `nullptr`.
 * @return            The final UVParts of the processed (sub)tree.
 */
UVParts pipeline(const std::string &tree_filename,const std::string &mesh_filename,
                 double threshold,
                 std::vector<UVParts> & individual_parts);

/**
 * @brief UVParts pipeline function that takes mesh data directly as parameters.
 * @param V              Vertex matrix
 * @param F              Face matrix  
 * @param tree_filename  Path to the tree binary file
 * @param configPath     Path to the config file
 * @param threshold      Distortion threshold
 * @param individual_parts Output vector to store individual parts
 */
UVParts pipeline(const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const std::vector<NodeRecord>& tree_nodes,
    const std::string& configPath,
    double threshold,
    bool pack_final_mesh,
    std::vector<UVParts>& individual_parts);
/**
 * @brief Sets the global mesh data and threshold for the pipeline.
 * @param V         Vertex matrix
 * @param F         Face matrix  
 * @param threshold Distortion threshold
 */
void set_global_mesh(const Eigen::MatrixXd& V,
                     const Eigen::MatrixXi& F,
                     double threshold);

#endif // UV_PIPELINE_H
