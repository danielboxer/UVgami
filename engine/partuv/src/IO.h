#ifndef IO_H
#define IO_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include <string>
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <fstream>
#include <iostream>

#include <variant>
#include <string>
#include <nlohmann/json.hpp>   // single-header JSON library

// #include "Distortion.h"


/**
 * @brief Draw a line on the given R, G, B matrices.
 * 
 * @param x0 Starting x-coordinate
 * @param y0 Starting y-coordinate
 * @param x1 Ending x-coordinate
 * @param y1 Ending y-coordinate
 * @param R  Red channel matrix
 * @param G  Green channel matrix
 * @param B  Blue channel matrix
 */
void draw_line(
    int x0, 
    int y0, 
    int x1, 
    int y1, 
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> &R,
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> &G,
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> &B);

/**
 * @brief Save UV layout to a PNG image.
 * 
 * @param V_uv    The UV coordinates (Eigen::MatrixXd)
 * @param F       Face indices (Eigen::MatrixXi)
 * @param uv_png  Filename/path to save PNG
 */
void save_uv_layout(const Eigen::MatrixXd &V_uv, const Eigen::MatrixXi &F, const std::string &uv_png,    const std::vector<std::pair<int,int>> &red_face_pairs = {});

/**
 * @brief Save a single Eigen::MatrixXd to a file.
 * 
 * @param matrix   The matrix to save
 * @param filename File path to save
 * @return true    If saving was successful
 * @return false   If an error occurred
 */
bool saveMatrix(const Eigen::MatrixXd &matrix, const std::string &filename);

/**
 * @brief Save two Eigen::MatrixXd objects to a single file.
 * 
 * @param matrix1  The first matrix to save
 * @param matrix2  The second matrix to save
 * @param filename File path to save
 * @return true    If saving was successful
 * @return false   If an error occurred
 */
bool saveMatrices(const Eigen::MatrixXd &matrix1, const Eigen::MatrixXd &matrix2, const std::string &filename);





// Pack the structure so that there is no padding (ensure the layout matches the Python-written binary)
#pragma pack(push, 1)
struct NodeRecord {
    int id;    // Node ID
    int left;  // Left child ID (or some sentinel value if none)
    int right; // Right child ID (or some sentinel value if none)
    
    double distortion;
    int num_faces;
    std::vector<int> children() const {
        return {left, right};
    }
};

struct BaseNodeRecord {
    int id;
    int left;
    int right;
};
#pragma pack(pop)


struct FaceAreaData {
    double area_2D;
    double area_3D;
    double ratio;
};


struct SubtreeInfo {
    double sum_2D;
    double sum_3D;
    double sum_ratio;
    int count;
};
 /**
  * @brief A Tree class that loads nodes from a binary file and provides O(1) access by node id.
  *
  * The binary file format:
  *   - 4 bytes (int): number of nodes.
  *   - For each node: 3 ints (id, left, right).
  */
 class Tree {
 public:
     /**
      * @brief Constructs the Tree by loading it from the specified binary file.
      *
      * @param filename The path to the binary file.
      */
     Tree(const std::string &filename);

     Tree(std::vector<NodeRecord> loaded_nodes);
 
     Tree() = default; // a default constructor
     /**
      * @brief Provides O(1) access to a node by its id.
      *
      * @param node_id The id of the node.
      * @return const NodeRecord& Reference to the node record.
      * @throws std::out_of_range if the node id is not found.
      */
     const NodeRecord& operator[](int node_id) const;
    /**
     * @brief Checks whether a node with the given id exists in the tree.
     *
     * @param node_id The node id to check.
     * @return true if the node exists (non-leaf), false otherwise.
     */
    bool contains(int node_id) const;

    int size() const { return nodes_.size(); }

    int root() const { 
        return root_; 
    }

    using const_iterator = std::unordered_map<int, NodeRecord>::const_iterator;

    // Provide iterator access to the underlying nodes
    const_iterator begin() const { return nodes_.cbegin(); }
    const_iterator end() const { return nodes_.cend(); }


    int set_node(int node_id, NodeRecord node) {
        nodes_[node_id] = node;
        return node_id;
    }

    void delete_node(int node_id) {
        nodes_.erase(node_id);
    }

    void update_distortion(const std::unordered_map<int, double> &leaf_distortion, int root);
 
    void update_distortion(const std::unordered_map<int, double> &leaf_distortion) {
        update_distortion(leaf_distortion, root_);
    }

    // void update_distortion_norm(const std::unordered_map<int, double> &leaf_distortion, int root);
    void update_distortion_norm(const std::vector<FaceAreaData>& face_data, const int root);

    void print_tree_recursive(int node_id,
        const std::string &prefix,
        bool isLast,
        int current_depth,
        int max_depth) const;


    void print_tree(int max_depth, int root = -1) const;
        
 private:
     /**
      * @brief Helper function that loads the tree nodes from the binary file.
      *
      * @param filename The path to the binary file.
      * @return std::vector<NodeRecord> Vector containing all the node records.
      */
     void update_num_faces();

     std::vector<NodeRecord> load_tree(const std::string &filename);
 
     // Internal hash map for O(1) lookup: node id -> NodeRecord.
     std::unordered_map<int, NodeRecord> nodes_;

     const int root_; // Initialized in the initializer list and remains constant.
 
 };
/**
 * @brief Returns all leaves (i.e., node IDs that are not keys in the tree) under the given node.
 *
 * This function performs a depth-first search starting from node_key. If a node exists
 * in the tree, it is assumed to be non-leaf and its children are pushed onto the stack.
 * Otherwise, the node is considered a leaf.
 *
 * @param tree The Tree object.
 * @param node_key The starting node id.
 * @return std::vector<int> Vector of leaf node ids.
 */
std::vector<int> get_tree_leaves(const Tree &tree, int node_key);

// int find_parent(const Tree &tree, int node_id) {
//     for (const auto &entry : tree) {
//         int parent_id = entry.first;
//         const NodeRecord &node = entry.second;
//         if (node.left == node_id || node.right == node_id) {
//             return parent_id;
//         }
//     }
//     return -1; // Parent not found (node is root or not in tree)
// };
int find_parent_with_child(const Tree &tree, int target_child) ;

void validate_tree_leaves(Tree &tree, int num_valid_faces);



/**
 * @brief Reassigns a face to another face's segment.
 *
 * This function does two things:
 *  1. Removes (detaches) `face_to_move` from its current segment by bypassing its parent.
 *  2. Inserts `face_to_move` into the segment of `target_face` by creating a new parent
 *     that groups `target_face` and `face_to_move`, and updating the tree accordingly.
 *
 * Note: This implementation assumes the existence of a helper function:
 *       int find_parent_with_child(const Tree &tree, int child_id)
 *       which returns the id of the parent node of the given child (or -1 if not found).
 *
 * @param tree The segmentation tree.
 * @param face_to_move The face index to reassign.
 * @param target_face The face index whose segment will absorb the face.
 */
void reassign_face_in_tree(Tree &tree, int face_to_move, int target_face);



std::vector<std::vector<int>> hierarchical_clustering_labels(const Tree& tree, int max_clusters);

std::vector<std::vector<int>> group_samples_by_label(const std::vector<int> &labels);

int find_lowest_common_ancestor(const Tree &tree, const std::vector<int> &nodes);

/*********************************SAVE HIERARCHY LABELS *********************************/


class Hierarchy
{
public:
    struct Node
    {
        // Leaf  : data = int(part_id)
        // Inner : data = std::pair<int,int>(left_id,right_id)
        std::variant<int, std::pair<int,int>> data;
        int faces = 0;                         // number of faces in this node
    };

    /* ------------ mutation -------------------- */
    void addLeaf (int id, int part_id, int faces);
    void addInner(int id, int left_id, int right_id, int faces);   // faces auto-computed
    void removeInner(int id);

    void updateLeaf(int id, int part_id);

    /* ------------ queries --------------------- */
    bool contains(int id) const;
    int  findRoot() const;
    const std::unordered_map<int,Node>& nodes() const;

    /* ------------ serialization --------------- */
    nlohmann::json to_json() const;
    void            save(const std::string& path) const;

private:
    std::unordered_map<int,Node> nodes_;
    int  computeFaces(int id) const;   // helper for addInner
};



#endif  // IO_H
