#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>
#include <stdexcept>
#include <cassert>
#include <fstream> 

#include <igl/adjacency_list.h>
#include <igl/readOBJ.h>
#include <igl/per_face_normals.h>
#include <igl/writeOBJ.h>

#include "AgglomerativeClustering.h"
#include "UnwrapBB.h"
#include "Config.h"

/*
    A minimal C++ Ward-linkage Agglomerative Clustering with connectivity.

    - We assume an (n_samples x n_features) data matrix X.
    - We assume a symmetric adjacency (connectivity) in a SparseMatrix<double>.
    - We do an early stop at n_clusters.

    This sketch follows the spirit of the Python code for ward_tree in scikit-learn
    but does not reproduce every detail (e.g. distance_threshold logic, etc.).
*/

using Eigen::MatrixXd;
using Eigen::SparseMatrix;
using Eigen::VectorXd;
using Eigen::VectorXi;

using namespace std;


// Make the connectivity matrix symmetric: conn <- conn + conn^T
static void symmetrize(SparseMatrix<double> &conn)
{
    // We will build a temporary list of triplets (i, j, val) for the sum.
    // Then re-construct the matrix.
    // Alternatively, one can do conn = conn + conn.transpose().
    SparseMatrix<double> tr = conn.transpose();
    // Now do conn + tr
    // The easiest way: use Eigen's operator+ on sparse:
    conn = conn + tr;
    conn.prune(0.0); // remove strictly-zero entries
}


struct EdgeCompare
{
    bool operator()(const AgglomerativeEdge &e1, const AgglomerativeEdge &e2)
    {
        if (e1.dist == e2.dist)
        {
            // sort by the max(i,j), if the same the min(i,j)
            int e1max = std::max(e1.i, e1.j);
            int e2max = std::max(e2.i, e2.j);
            if (e1max == e2max)
            {
                return std::min(e1.i, e1.j) > std::min(e2.i, e2.j);
            }
            return e1max > e2max;

        }else{
            return e1.dist > e2.dist;
        }
    }
};

// Helper function to compute the Ward distance between two clusters given
// their sums and sizes.
static double wardDistance(const VectorXd &sumI, double sizeI,
                           const VectorXd &sumJ, double sizeJ, double &sq)
{
    // Ward distance ~ scaled squared distance between centroids.
    // diff = centroid_i - centroid_j
    
    
    
    // VectorXd diff = (sumI / sizeI).normalize() - (sumJ / sizeJ).normalize();
    // VectorXd avgI = sumI / sizeI; avgI.normalize();
    // VectorXd avgJ = sumJ / sizeJ; avgJ.normalize();
    // VectorXd diff = avgI - avgJ;
    // cout << "sq: " << sq <<   std::endl;

    // distance = ||diff||^2 * (sizeI * sizeJ)/(sizeI + sizeJ)
    VectorXd diff = sumI / sizeI - sumJ / sizeJ;
    sq = diff.squaredNorm();
    double factor = (sizeI * sizeJ) / (sizeI + sizeJ);
    // double factor = 1;

    return sq * factor;
}



static int findConnectedComponents(const std::vector<std::vector<int>>& adj,
                                   std::vector<int>& labels)
{
    const int n = static_cast<int>(adj.size());
    labels.assign(n, -1);

    int currentLabel = 0;
    for (int start = 0; start < n; ++start)
    {
        // If this node is already labeled, skip it
        if (labels[start] != -1) 
            continue;

        // BFS from 'start'
        std::vector<int> queue;
        queue.push_back(start);
        labels[start] = currentLabel;

        for (size_t idx = 0; idx < queue.size(); ++idx)
        {
            int v = queue[idx];
            // Go through neighbors of v
            for (int nb : adj[v])
            {
                if (labels[nb] == -1)
                {
                    labels[nb] = currentLabel;
                    queue.push_back(nb);
                }
            }
        }
        // Finished traversing one connected component
        ++currentLabel;
    }

    return currentLabel; // number of connected components
}




std::vector<std::vector<std::vector<int>>> AgglomerativeClustering::fit(const MatrixXd &X, std::vector<std::vector<int>>& A, int min_clusters)
{

    min_clusters = std::max(min_clusters, 1);
    std::vector<std::vector<std::vector<int>>> hierarchical_labels(n_clusters_-min_clusters+1);

    const int n_samples = (int)X.rows();
    const int n_features = (int)X.cols();
    if(n_samples <= n_clusters_)
    {
        throw std::runtime_error("n_clusters must be < n_samples.");
    }
    if (A.size() != n_samples)
    {
        throw std::runtime_error("Connectivity matrix must match X.shape[0].");
    }
    // 1) Make connectivity symmetrical
    int n_nodes = 2 * n_samples - 1;
    uf_ = UnionFind(n_nodes);

    // connectivity.resize(n_nodes, n_nodes);
    // symmetrize(connectivity);

    std::vector<int> compLabels;
    int n_connected_components = findConnectedComponents(A, compLabels);
    if(n_connected_components > 1)
    {
        std::cerr << "[Warning] Graph has " << n_connected_components
                    << " connected components; a real fix would connect them.\n";
    }
    // vector<vector<int>> A(n_nodes);

    //  reserve A to n_nodes
    A.resize(n_nodes);


    moments_1_.resize(n_nodes, 0.0);
    moments_2_.resize(n_nodes, VectorXd::Zero(n_features));
    used_node_.assign(n_nodes, true);
    parent_.resize(n_nodes);
    std::iota(parent_.begin(), parent_.end(), 0);

    // Initialize leaf nodes [0..n_samples-1]
    for(int i = 0; i < n_samples; i++)
    {
        moments_1_[i] = 1.0;
        moments_2_[i] = X.row(i);
        // used_node_[i] = true;
    }

    std::priority_queue<AgglomerativeEdge, std::vector<AgglomerativeEdge>, EdgeCompare> heap;
    // ensure the sparse matrix is in compressed form
    // We only want i < j to avoid duplication (or i > j, consistently).
    for(int k = 0; k < n_samples; k++)
    {
        // for(Eigen::SparseMatrix<double>::InnerIterator it(connectivity, k); it; ++it)
        for(int c: A[k])
        {
            if(c <= k) continue; // only keep upper triangular
        
            // double dist = 1; 
            double sq;
            double dist = wardDistance(moments_2_[k], moments_1_[k],
                moments_2_[c], moments_1_[c], sq);
            // push to heap

            AgglomerativeEdge edge(dist, k, c);
            edge.dist_raw = sq;
            heap.push(edge);
        }
    }

    children_.clear();
    children_.reserve(n_samples - 1); // if we built full tree
    int nextNodeID = n_samples; 
    // int mergesToDo = n_samples - n_clusters_;


    // n_nodes - 1 because we don't need the one part 
    int num_cluster = 0;
    for (int k = n_samples; k < n_nodes - (min_clusters - 1); ++k)
    {

        int i = -1, j = -1;
        
        while( !heap.empty())
        {
            AgglomerativeEdge e = heap.top();
            heap.pop();
            // Check if both are still active
            i = e.i;
            j = e.j;
            if(used_node_[i] && used_node_[j]) {
                // std::cout << "distance: " << e.dist << " normal diff " <<  e.dist_raw  << " i: " << i << " j: " << j << std::endl;
                break;
            }
            
        }
        if (i == -1 || j == -1)
        {
            std::cerr << "[Warning] No valid edge found for merge.\n";
            continue;
        }

        used_node_[i] = false;
        used_node_[j] = false;
        parent_[i] = k;
        parent_[j] = k;

        // record children
        children_.push_back(std::make_pair(i, j));

        uf_.unite(i, k);
        uf_.unite(j, k);

        // update moments
        // double sizeK = moments_1_[i] + moments_1_[j];
        moments_1_[k] = moments_1_[i] + moments_1_[j];
        moments_2_[k] = moments_2_[i] + moments_2_[j];

        std::vector<int> neighbors;
        // function to gather from a node's adjacency
        auto gatherNeighbors = [&](int node)
        {
            // for(Eigen::SparseMatrix<double>::InnerIterator it(connectivity, node); it; ++it)
            for (int node : A[node])
            {
                if(parent_[node] != k) {
                    neighbors.push_back(parent_[node]);
                }
            }

        };
        gatherNeighbors(i);
        gatherNeighbors(j);
        // sort & unique to remove duplicates
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());

        A[k] = neighbors;
        for(int nb : neighbors)
        {
            A[nb].push_back(k);
        }


        for(int nb : neighbors)
        {
            if(nb == k) continue;
            double sq;
            double distKnb = wardDistance(moments_2_[k], moments_1_[k],
                                            moments_2_[nb], moments_1_[nb], sq);

            int i_min = (k < nb) ? k : nb;
            int i_max = (k < nb) ? nb : k;
            // push to heap
            AgglomerativeEdge edge(distKnb, i_max, i_min);
            edge.dist_raw = sq;
            heap.push(edge);
        }

        // mergesToDo--;

        if(n_nodes - k <= n_clusters_)
        {
            std::vector<int> roots(n_samples);
            for (int j = 0; j < n_samples; ++j) {
                roots[j] = uf_.find(j);
            }

            // 2) Group samples by root
            std::unordered_map<int, std::vector<int>> groups;
            groups.reserve(n_samples);
            for (int j = 0; j < n_samples; ++j) {
                int r = roots[j];
                groups[r].push_back(j);
            }

            vector<vector<int>> label(n_nodes - k);
            int g = 0;
            for (auto &kv : groups) {
                label[g++] = kv.second;
            }
            // hierarchical_labels[n_nodes - k - 2] = label;
            hierarchical_labels[num_cluster++] = label;

        }
    }

    return hierarchical_labels;

}
 
