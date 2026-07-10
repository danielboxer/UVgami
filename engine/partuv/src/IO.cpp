#include "IO.h"
#include "Config.h"

// Standard library headers
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include <stdexcept>
#include <stack>
#include <cassert>

#include <Eigen/Core>
#include <set>
#include <utility>  // for std::pair
#include <algorithm> // for std::minmax, std::sort
#include <string>

#include <iomanip>
#include <stdexcept>

// STB library for PNG writing
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

bool saveMatrix(const Eigen::MatrixXd &matrix, const std::string &filename)
{
    std::ofstream outFile(filename);
    if (!outFile) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return false;
    }

    // Write dimensions
    outFile << matrix.rows() << " " << matrix.cols() << "\n";
    // Write actual matrix data
    outFile << matrix << "\n";

    outFile.close();
    return true;
}

bool saveMatrices(const Eigen::MatrixXd &matrix1, const Eigen::MatrixXd &matrix2, const std::string &filename)
{
    std::ofstream outFile(filename);
    if (!outFile) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return false;
    }

    // Write the first matrix
    outFile << matrix1.rows() << " " << matrix1.cols() << "\n";
    outFile << matrix1 << "\n";

    // Write the second matrix
    outFile << matrix2.rows() << " " << matrix2.cols() << "\n";
    outFile << matrix2 << "\n";

    outFile.close();
    return true;
}
void draw_line(
    int x0, 
    int y0, 
    int x1, 
    int y1, 
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> &R,
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> &G,
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> &B,
    unsigned char r_color = 0,    // default black
    unsigned char g_color = 0,    // default black
    unsigned char b_color = 0     // default black
)
{
    // Use Bresenham's line algorithm
    bool steep = (std::abs(y1 - y0) > std::abs(x1 - x0));
    if (steep)
    {
        std::swap(x0, y0);
        std::swap(x1, y1);
    }
    if (x0 > x1)
    {
        std::swap(x0, x1);
        std::swap(y0, y1);
    }

    int dx = x1 - x0;
    int dy = std::abs(y1 - y0);
    int error = dx / 2;
    int ystep = (y0 < y1) ? 1 : -1;
    int y = y0;

    for (int x = x0; x <= x1; x++)
    {
        if (steep)
        {
            // row = y, col = x
            if (y >= 0 && y < R.rows() && x >= 0 && x < R.cols())
            {
                R(y, x) = r_color;
                G(y, x) = g_color;
                B(y, x) = b_color;
            }
        }
        else
        {
            // row = x, col = y
            if (x >= 0 && x < R.rows() && y >= 0 && y < R.cols())
            {
                R(x, y) = r_color;
                G(x, y) = g_color;
                B(x, y) = b_color;
            }
        }

        error -= dy;
        if (error < 0)
        {
            y += ystep;
            error += dx;
        }
    }
}


// Helper to store edges (v0, v1) as an ordered pair (smallest first)
inline std::pair<int,int> make_edge_key(int v0, int v1)
{
    if (v0 > v1) std::swap(v0,v1);
    return std::make_pair(v0,v1);
}


void save_uv_layout(
    const Eigen::MatrixXd &V_uv,
    const Eigen::MatrixXi &F,
    const std::string &uv_png,
    // Optional list of face-pairs to highlight the shared edge in red
    const std::vector<std::pair<int,int>> &red_face_pairs
)
{
    using namespace Eigen;
    using namespace std;

    // ------------------------------------------------------------
    // 1) Normalize UVs to [0, 1]
    // ------------------------------------------------------------
    VectorXd min_uv = V_uv.colwise().minCoeff();
    VectorXd max_uv = V_uv.colwise().maxCoeff();
    MatrixXd uv_01 = V_uv;
    uv_01.rowwise() -= min_uv.transpose();
    for (int i = 0; i < 2; i++)
    {
        double range = max_uv(i) - min_uv(i);
        if (range > 1e-9)
        {
            uv_01.col(i) /= range;
        }
    }

    // ------------------------------------------------------------
    // 2) Build set of edges that should be drawn in red
    // ------------------------------------------------------------
    // For each pair (f0, f1), find the shared edge (if any).
    // That edge is stored in a std::set so we can quickly check membership.
    set<pair<int,int>> red_edges;  // store (min(v0,v1), max(v0,v1))

    for (const auto &fp : red_face_pairs)
    {
        int f0 = fp.first;
        int f1 = fp.second;
        if (f0 < 0 || f0 >= F.rows() || f1 < 0 || f1 >= F.rows()) continue;

        // Add all edges of face f0
        for (int i = 0; i < 3; i++) {
            int v0 = F(f0, i);
            int v1 = F(f0, (i+1) % 3);
            red_edges.insert(make_edge_key(v0, v1));
        }

        // Add all edges of face f1
        for (int i = 0; i < 3; i++) {
            int v0 = F(f1, i);
            int v1 = F(f1, (i+1) % 3);
            red_edges.insert(make_edge_key(v0, v1));
        }
    }

    // ------------------------------------------------------------
    // 3) Create an image buffer
    // ------------------------------------------------------------
    const int width  = 512;
    const int height = 512;
    Matrix<unsigned char, Dynamic, Dynamic> R(height, width),
                                           G(height, width),
                                           B(height, width),
                                           A(height, width);

    // White background
    R.setConstant(255);
    G.setConstant(255);
    B.setConstant(255);
    A.setConstant(255);

    // ------------------------------------------------------------
    // 4) Draw each edge
    // ------------------------------------------------------------
    for (int f_i = 0; f_i < F.rows(); f_i++)
    {
        for (int e = 0; e < 3; e++)
        {
            int v0 = F(f_i, e);
            int v1 = F(f_i, (e + 1) % 3);

            int x0 = static_cast<int>(uv_01(v0, 0) * (width  - 1));
            int y0 = static_cast<int>(uv_01(v0, 1) * (height - 1));
            int x1 = static_cast<int>(uv_01(v1, 0) * (width  - 1));
            int y1 = static_cast<int>(uv_01(v1, 1) * (height - 1));

            // Check if this edge is in the red set
            auto edge_key = make_edge_key(v0, v1);
            bool is_red = (red_edges.find(edge_key) != red_edges.end());

            // Choose color
            unsigned char r_col = is_red ? 255 : 0;  // red if in set, else black
            unsigned char g_col = 0;
            unsigned char b_col = 0;

            // draw_line expects (row, col) => (y, x)
             draw_line(y0, x0, y1, x1, R, G, B, r_col, g_col, b_col);
        }
    }

    // ------------------------------------------------------------
    // 5) Convert to a single RGBA array for stb_image_write
    // ------------------------------------------------------------
    vector<unsigned char> RGBA_data(width * height * 4, 255);
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int idx = (y * width + x) * 4;
            RGBA_data[idx + 0] = R(y, x);  // R
            RGBA_data[idx + 1] = G(y, x);  // G
            RGBA_data[idx + 2] = B(y, x);  // B
            RGBA_data[idx + 3] = A(y, x);  // A
        }
    }

    // ------------------------------------------------------------
    // 6) Save as PNG (using stb_image_write or similar)
    // ------------------------------------------------------------
    if (stbi_write_png(uv_png.c_str(), width, height, 4, RGBA_data.data(), width * 4))
    {
        cout << "UV layout saved as " << uv_png << endl;
    }
    else
    {
        cerr << "Failed to write PNG file: " << uv_png << endl;
    }
}


Tree::Tree(const std::string &filename)
// The lambda is executed immediately during initialization:
// - It loads the nodes from the file.
// - It fills the nodes_ hash map.
// - It returns the computed root value.
: root_([&]() -> int {
    std::vector<NodeRecord> loaded_nodes = load_tree(filename);
    for (const auto &node : loaded_nodes) {
        nodes_[node.id] = node;
    }
    return static_cast<int>(loaded_nodes.size() * 2);
}())
{
    update_num_faces();
// Nothing else to do in the constructor body.
}


Tree::Tree(std::vector<NodeRecord> loaded_nodes)
: root_([&]() -> int {
    for (const auto &node : loaded_nodes) {
        nodes_[node.id] = node;
    }
    return static_cast<int>(loaded_nodes.size() * 2);
}())
{
    update_num_faces();
}


std::vector<NodeRecord> Tree::load_tree(const std::string &filename) {
    std::vector<NodeRecord> nodes;
    
    std::ifstream fin(filename, std::ios::binary);
    if (!fin) {
        std::cerr << "Error opening file: " << filename << "\n";
        return nodes;
    }
    
    // Read the number of nodes (4 bytes).
    int num_nodes = 0;
    fin.read(reinterpret_cast<char*>(&num_nodes), sizeof(num_nodes));
    if (!fin) {
        std::cerr << "Error reading number of nodes from file: " << filename << "\n";
        return nodes;
    }
    
    // Resize the vector to hold all nodes.
    nodes.resize(num_nodes);
    
    // Read the node records.
    std::vector<BaseNodeRecord> old_nodes(num_nodes);
    fin.read(reinterpret_cast<char*>(old_nodes.data()), num_nodes * sizeof(BaseNodeRecord));
    if (!fin) {
        std::cerr << "Error reading node records from file: " << filename << "\n";
        return nodes;
    }

    nodes.resize(num_nodes);
    for (int i = 0; i < num_nodes; i++) {
        nodes[i].id         = old_nodes[i].id;
        nodes[i].left       = old_nodes[i].left;
        nodes[i].right      = old_nodes[i].right;
    }



    return nodes;
}

const NodeRecord& Tree::operator[](int node_id) const {
    auto it = nodes_.find(node_id);
    if (it == nodes_.end()) {
        std::cerr << "Node id not found in tree: " << node_id << std::endl;
        std::cerr << "Tree size: " << nodes_.size() << std::endl;
        std::cerr << "node parent: " << find_parent_with_child(*this, node_id) << std::endl;
        throw std::out_of_range("Node id not found in tree");
    }
    return it->second;
}



bool Tree::contains(int node_id) const {
    return nodes_.find(node_id) != nodes_.end();
}

void Tree::update_num_faces() {
    // Define a local lambda for the DFS/post-order traversal:
    std::function<int(int)> dfs = [&](int node_id) -> int {
        // If node doesn't exist or is sentinel, return 0
        if (!contains(node_id)) {
            return 1;
        }

        // Get a reference to the current node (note: non-const because we update num_faces)
        NodeRecord &node = nodes_[node_id];

        // Recurse on children:
        int left_count = dfs(node.left);
        int right_count = dfs(node.right);

        node.num_faces = left_count + right_count;
        return node.num_faces;
    };

    // Initiate recursion from the root
    dfs(root_);
}

void Tree::update_distortion(const std::unordered_map<int, double> &leaf_distortion, const int root)
{
    // We'll define a helper function returning (sum_of_leaf_distortions, number_of_leaves).
    // Using a std::pair<double, int> to carry both values up the recursion.
    std::function<std::pair<double, int>(int)> dfs = [&](int node_id) -> std::pair<double, int>
    {
        // If this node doesn't exist/sentinel => no leaves, no sum.
        if (!contains(node_id)) {
            try{
                return {leaf_distortion.at(node_id), 1}; // leaf distortion, count 1
            } catch (const std::out_of_range& e) {
                std::cerr << "Node ID " << node_id << " not found in leaf_distortion map." << std::endl;
                return {0.0, 0}; // no distortion, count 0
            }
        }

        // Get reference to node (non-const since we'll update node.distortion)
        NodeRecord &node = nodes_[node_id];

        // Recursively process children
        auto [left_sum, left_count] = dfs(node.left);
        auto [right_sum, right_count] = dfs(node.right);

        double total_sum = left_sum + right_sum;
        int total_count = left_count + right_count;

        if(total_count!=node.num_faces){
            std::cerr << "Node ID " << node_id << " has a mismatch in num_faces: " << node.num_faces << " vs. total_count: " << total_count << std::endl;
        }

        double avgDist = (total_count > 0) ? (total_sum / total_count) : 0.0;
        node.distortion = avgDist;
        return {total_sum, total_count};

    };

    // Initiate the recursion from our root
    dfs(root);
}




void Tree::update_distortion_norm(const std::vector<FaceAreaData>& face_data, const int root) {
    std::function<SubtreeInfo(int)> dfs = [&](int node_id) -> SubtreeInfo {
        if (!contains(node_id)) {  // Leaf node (face)
            if (node_id < 0 || node_id >= face_data.size()) {
                return {0.0, 0.0, 0.0, 0};
            }
            const auto& data = face_data[node_id];
            double sum_2D = data.area_2D;
            double sum_3D = data.area_3D;
            double total_ratio = (sum_3D == 0.0) ? 0.0 : (sum_2D / sum_3D);
            double face_ratio = data.ratio;
            double distortion = face_ratio / total_ratio;

            if (CONFIG_pipelineThreshold > 1) {
                distortion = (distortion < 1.0) ? (1.0 / distortion) : distortion;
            } else {
                distortion = (distortion < 1.0) ? (1.0 - distortion) : (1.0 - 1.0 / distortion);
            }
            return {sum_2D, sum_3D, distortion, 1};
        }

        NodeRecord &node = nodes_[node_id];
        SubtreeInfo left = dfs(node.left);
        SubtreeInfo right = dfs(node.right);

        double sum_2D = left.sum_2D + right.sum_2D;
        double sum_3D = left.sum_3D + right.sum_3D;
        double total_ratio = (sum_3D == 0.0) ? 0.0 : (sum_2D / sum_3D);

        // Collect all face indices in the current subtree
        std::vector<int> face_indices;
        std::function<void(int)> collect_faces = [&](int n) {
            if (!contains(n)) {
                face_indices.push_back(n);
                return;
            }
            NodeRecord &current = nodes_[n];
            collect_faces(current.left);
            collect_faces(current.right);
        };
        collect_faces(node_id);

        // Compute distortions for all faces in this subtree
        double sum_distortion = 0.0;
        for (int face_idx : face_indices) {
            const auto& data = face_data[face_idx];
            double face_ratio = data.ratio;
            double distortion = face_ratio / total_ratio;

            if (CONFIG_pipelineThreshold > 1) {
                distortion = (distortion < 1.0) ? (1.0 / distortion) : distortion;
            } else {
                distortion = (distortion < 1.0) ? (1.0 - distortion) : (1.0 - 1.0 / distortion);
            }
            sum_distortion += distortion;
        }

        int count = face_indices.size();
        node.distortion = (count > 0) ? (sum_distortion / count) : 0.0;

        return {sum_2D, sum_3D, sum_distortion, count};
    };

    dfs(root);
}

void Tree::print_tree_recursive(int node_id,
                                const std::string &prefix,
                                bool isLast,
                                int current_depth,
                                int max_depth) const
{
    // Find node in map
    auto it = nodes_.find(node_id);
    if (it == nodes_.end())
    {
        // If it's not in the map, treat it as a "face" node:
        std::cout << prefix
                  << (isLast ? "└── " : "├── ")
                  << "Face id: " << node_id
                  << std::endl;
        return; // After printing, return because there's no further recursion
    }

    // Extract the record
    const NodeRecord &node = it->second;

    // Print the current node, including distortion and num_faces
    std::cout << prefix
              << (isLast ? "└── " : "├── ")
              << "ID=" << node_id
              << " (distortion=" << node.distortion
              << ", num_faces=" << node.num_faces << ")"
              << std::endl;

    // If we've reached our max depth, do not recurse further
    if (current_depth >= max_depth)
    {
        return;
    }

    // Recurse on each child
    const auto &children = node.children();
    for (size_t i = 0; i < children.size(); ++i)
    {
        bool childIsLast = (i == children.size() - 1);

        // The next prefix includes "│   " if not last, or "    " if this is last
        std::string newPrefix = prefix + (isLast ? "    " : "│   ");

        // Recurse, incrementing depth by 1
        print_tree_recursive(children[i], newPrefix, childIsLast, current_depth + 1, max_depth);
    }
}
void Tree::print_tree(int max_depth, int root) const
{   
    if (root == -1)
    {
        root = root_;
    }
    auto it = nodes_.find(root);
    if (it == nodes_.end())
    {
        std::cout << "Tree is empty or root is not in the tree." << std::endl;
        return;
    }

    // Kick off the recursion from the root at depth 0
    print_tree_recursive(root, /*prefix=*/"", /*isLast=*/true, /*current_depth=*/0, max_depth);
}

// Potentially here could cache the tree to avoid repetitive getting leaves
// but currently time may be ok
std::vector<int> get_tree_leaves(const Tree &tree, int node_key) {
    std::vector<int> descendants;
    std::stack<int> stack;
    stack.push(node_key);

    while (!stack.empty()) {
        int current = stack.top();
        stack.pop();

        // If the tree contains the current node, it is non-leaf.
        if (tree.contains(current)) {
            const NodeRecord &node = tree[current];
            // Push both children onto the stack.
            stack.push(node.left);
            stack.push(node.right);
        } else {
            // Otherwise, it's a leaf node.
            descendants.push_back(current);
        }
    }
    return descendants;
}




int find_parent_with_child(const Tree &tree, int target_child) {
    for (const auto &pair : tree) {
        // pair.first is the node id, pair.second is the NodeRecord
        const NodeRecord &node = pair.second;
        if (node.left == target_child || node.right == target_child) {
            return pair.first;
        }
    }
    return -1;
}

void validate_tree_leaves(Tree &tree, int num_valid_faces){
    int tree_size = tree.size();
    for (int i = num_valid_faces; i < tree_size +1; i++) {
        int invalid_parent = find_parent_with_child(tree, i);
        if (invalid_parent == -1) {
            std::cout << "Cannot find parent with invalid child " + std::to_string(i) << std::endl;
            
        }

        bool left_leaf_invalid = tree[invalid_parent].left == i;

        int invalid_grandparent = find_parent_with_child(tree, invalid_parent);
        if (invalid_grandparent == -1) {

            std::cout << "tree size: " << tree.size() << "   "  << std::endl;

            std::cout << "Cannot find grandparent with invalid child " + std::to_string(invalid_parent) << std::endl;
        }

        NodeRecord new_node = tree[invalid_grandparent];
        if (tree[invalid_grandparent].left == invalid_parent) {
            new_node.left = left_leaf_invalid ? tree[invalid_parent].right : tree[invalid_parent].left;
        } else {
            new_node.right = left_leaf_invalid ? tree[invalid_parent].right : tree[invalid_parent].left;
        }

        tree.set_node(invalid_grandparent, new_node);
        tree.delete_node(invalid_parent);
        std::cout << "Deleted " << i << " invalid node " + std::to_string(invalid_parent) << " and updated parent " + std::to_string(invalid_grandparent) << std::endl;
    }

}


void reassign_face_in_tree(Tree &tree, int face_to_move, int target_face) {
    // --- Step 1: Detach face_to_move from its current segment ---
    int current_parent = find_parent_with_child(tree, face_to_move);
    if (current_parent == -1) {
        std::cerr << "Error: Face " << face_to_move << " has no parent in the tree." << std::endl;
        return;
    }
    // Determine if face_to_move is the left child.
    bool face_is_left = (tree[current_parent].left == face_to_move);
    
    // Find the grandparent (parent of the current parent).
    int current_grandparent = find_parent_with_child(tree, current_parent);
    if (current_grandparent == -1) {
        std::cerr << "Error: Parent node " << current_parent << " has no parent. Cannot reassign." << std::endl;
        return;
    }
    // Get the sibling of the node we are about to remove.
    int sibling = face_is_left ? tree[current_parent].right : tree[current_parent].left;
    
    // Update the grandparent to bypass the current parent.
    NodeRecord gp_node = tree[current_grandparent];
    if (gp_node.left == current_parent) {
        gp_node.left = sibling;
    } else {
        gp_node.right = sibling;
    }
    tree.set_node(current_grandparent, gp_node);
    
    // Delete the current parent node from the tree.
    tree.delete_node(current_parent);
    
    // --- Step 2: Insert face_to_move into target_face's segment ---
    // Find the parent of the target face.
    int target_parent = find_parent_with_child(tree, target_face);
    if (target_parent == -1) {
        std::cerr << "Error: Target face " << target_face << " has no parent in the tree." << std::endl;
        return;
    }
    
    // Create a new internal node that will group target_face and face_to_move.
    // Generate a new unique node id. (Here, we simply use tree.size() + 1,

    NodeRecord new_node;
    new_node.id = current_parent;
    
    // Decide how to order the children. For example, if target_face was the left child
    // of its parent, we can keep it that way:
    if (tree[target_parent].left == target_face) {
        new_node.left = target_face;
        new_node.right = face_to_move;
    } else {
        new_node.left = face_to_move;
        new_node.right = target_face;
    }
    
    // Insert the new node into the tree.
    tree.set_node(current_parent, new_node);
    
    // Update the target parent's pointer so that it now points to the new node instead of target_face.
    NodeRecord target_parent_node = tree[target_parent];
    if (target_parent_node.left == target_face) {
        target_parent_node.left = current_parent;
    } else {
        target_parent_node.right = current_parent;
    }
    tree.set_node(target_parent, target_parent_node);
}


class UnionFind {
    private:
        std::vector<int> parent;
        std::vector<int> rank;
    
    public:
        UnionFind(int size) {
            parent.resize(size);
            rank.resize(size, 1);
            for (int i = 0; i < size; ++i) {
                parent[i] = i;
            }
        }
    
        int find(int x) {
            if (parent[x] != x) {
                parent[x] = find(parent[x]);  // Path compression
            }
            return parent[x];
        }
    
        void union_nodes(int x, int y) {
            int rootX = find(x);
            int rootY = find(y);
            if (rootX != rootY) {
                // Union by rank
                if (rank[rootX] > rank[rootY]) {
                    parent[rootY] = rootX;
                } else if (rank[rootX] < rank[rootY]) {
                    parent[rootX] = rootY;
                } else {
                    parent[rootY] = rootX;
                    rank[rootX]++;
                }
            }
        }
    };
    
std::vector<std::vector<int>> hierarchical_clustering_labels(const Tree& tree, int max_clusters) {
    // Get all leaves under the root to determine the number of samples
    std::vector<int> leaves = get_tree_leaves(tree, tree.root());
    int n_samples = leaves.size();
    if (n_samples == 0) {
        return {};  // Edge case: no samples
    }

    // Determine the maximum node ID (root is the highest)
    int root_id = tree.root();
    UnionFind uf(root_id + 1);

    // Collect all internal node IDs (present in the tree's nodes_)
    std::vector<int> internal_nodes;
    for (const auto& pair : tree) {
        internal_nodes.push_back(pair.first);
    }

    // Sort internal nodes by ID in ascending order to process merges correctly
    std::sort(internal_nodes.begin(), internal_nodes.end());

    int current_cluster_count = n_samples;
    std::vector<std::vector<int>> hierarchical_labels;

    for (int node_id : internal_nodes) {
        const NodeRecord& node = tree[node_id];
        // Union the left and right children with the current internal node
        uf.union_nodes(node.left, node_id);
        uf.union_nodes(node.right, node_id);

        current_cluster_count--;

        // Collect labels once the cluster count is <= max_clusters
        if (current_cluster_count <= max_clusters) {
            std::vector<int> labels;
            labels.reserve(n_samples);
            for (int i = 0; i < n_samples; ++i) {
                labels.push_back(uf.find(i));
            }
            hierarchical_labels.push_back(std::move(labels));
        }
    }

    return hierarchical_labels;
}


std::vector<std::vector<int>> group_samples_by_label(const std::vector<int> &labels)
{
    // 1) Map label -> vector of indices (faces)
    std::unordered_map<int, std::vector<int>> label_to_indices;
    label_to_indices.reserve(labels.size());
    
    for (int i = 0; i < static_cast<int>(labels.size()); ++i) {
        int lbl = labels[i];
        label_to_indices[lbl].push_back(i);
    }
    
    // 2) Collect groups into the result
    std::vector<std::vector<int>> result;
    result.reserve(label_to_indices.size());

    for (auto &kv : label_to_indices) {
        result.push_back(std::move(kv.second));
    }
    
    return result;
}


/**
 * @brief Return the path (inclusive) from 'node_id' up to the root.
 * If 'node_id' does not exist or is the root, the path will be size 1 (just node_id).
 */
std::vector<int> get_path_to_root(const Tree &tree, int node_id) {
    std::vector<int> path;
    int current = node_id;

    // Walk upward until we run out of parents (i.e., at the root)
    while (current != -1) {
        path.push_back(current);
        // Use your find_parent_with_child(...) to get the parent of 'current'
        current = find_parent_with_child(tree, current);
    }

    return path;
}


/**
 * @brief Find the Lowest Common Ancestor (LCA) of two node IDs in the tree.
 * Returns -1 if they have no common ancestor in this tree (unlikely in a well-formed tree).
 */
int find_lowest_common_ancestor(const Tree &tree, int node_a, int node_b) {
    // 1) Get the path from each node up to the root
    std::vector<int> path_a = get_path_to_root(tree, node_a);
    std::vector<int> path_b = get_path_to_root(tree, node_b);

    // 2) Put one path in a set for quick lookup
    std::unordered_set<int> ancestors_b(path_b.begin(), path_b.end());

    // 3) Walk up path_a (which is leaf->root).  The first node we encounter
    //    that is also in ancestors_b is the LCA (lowest in the tree).
    for (int node : path_a) {
        if (ancestors_b.count(node) > 0) {
            return node;  // Found the first common node from leaf upward
        }
    }

    // Not found
    return -1;
}

/**
 * @brief Find the Lowest Common Ancestor of a set of node IDs.
 * If the vector is empty, returns -1.
 */
int find_lowest_common_ancestor(const Tree &tree, const std::vector<int> &nodes) {
    if (nodes.empty()) {
        return -1;
    }

    int current_ancestor = nodes[0];
    for (size_t i = 1; i < nodes.size(); ++i) {
        current_ancestor = find_lowest_common_ancestor(tree, current_ancestor, nodes[i]);
        if (current_ancestor == -1) {
            // No need to continue if no common ancestor found
            break;
        }
    }

    return current_ancestor;
}


/*********************************SAVE HIERARCHY LABELS *********************************/



using json = nlohmann::json;

/* ---------------- mutation ---------------- */

void Hierarchy::addLeaf(int id, int part_id, int faces)
{
    nodes_[id].data = part_id;
    nodes_[id].faces = faces;
}

void Hierarchy::addInner(int id, int left_id, int right_id, int faces)
{
    if (left_id == right_id)
        throw std::invalid_argument("left_id and right_id must differ");
    nodes_[id].data = std::pair<int,int>{left_id, right_id};
    nodes_[id].faces = faces;
}

void Hierarchy::removeInner(int id)
{
    nodes_.erase(id);
}

void Hierarchy::updateLeaf(int id, int part_id)
{
    nodes_[id].data = part_id;
}

/* ---------------- queries ----------------- */

bool Hierarchy::contains(int id) const
{
    return nodes_.count(id) != 0;
}

int Hierarchy::findRoot() const
{
    std::unordered_set<int> children;
    for (auto&& [id,node] : nodes_)
        if (auto p = std::get_if<std::pair<int,int>>(&node.data))
            children.insert({p->first, p->second});

    int root = -1;
    for (auto&& [id,_] : nodes_)
        if (!children.count(id))
        {
            if (root != -1) throw std::runtime_error("Hierarchy::findRoot - multiple roots");
            root = id;
        }
    if (root == -1) throw std::runtime_error("Hierarchy::findRoot - no root");
    return root;
}

const std::unordered_map<int,Hierarchy::Node>& Hierarchy::nodes() const
{
    return nodes_;
}

/* ---------------- serialization ---------- */

json Hierarchy::to_json() const
{
    json j;
    for (auto&& [id,node] : nodes_)
    {
        json entry;
        if (std::holds_alternative<int>(node.data))
            entry["part"] = std::get<int>(node.data);
        else
        {
            auto [l,r] = std::get<std::pair<int,int>>(node.data);
            entry["left"]  = l;
            entry["right"] = r;
        }
        entry["faces"] = node.faces;
        j[std::to_string(id)] = std::move(entry);
    }
    return j;
}


void Hierarchy::save(const std::string& path) const
{
    std::ofstream(path) << std::setw(4) << to_json();
}