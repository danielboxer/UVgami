#ifndef AGGLOMERATIVE_CLUSTERING_H
#define AGGLOMERATIVE_CLUSTERING_H

#include <Eigen/Dense>
#include <vector>
#include <queue>
#include <utility>
#include <stdexcept>


class UnionFind {
    public:
        // Constructor: create a union-find for 'n' elements
        UnionFind(int n) : parent(n), rank(n, 0) {
            for (int i = 0; i < n; i++) {
                parent[i] = i;
            }
        }

        UnionFind()
        {
            // Default constructor
        }
        
        // Find with path compression
        int find(int x) {
            if (parent[x] != x) {
                parent[x] = find(parent[x]);
            }
            return parent[x];
        }
        
        // Union by rank
        void unite(int x, int y) {
            int rx = find(x);
            int ry = find(y);
            if (rx != ry) {
                if (rank[rx] < rank[ry]) {
                    parent[rx] = ry;
                } else if (rank[rx] > rank[ry]) {
                    parent[ry] = rx;
                } else {
                    parent[ry] = rx;
                    rank[rx]++;
                }
            }
        }
    
        private:
        std::vector<int> parent;
        std::vector<int> rank;
    };
    
struct AgglomerativeEdge {
    double dist;
    double dist_raw;
    int i;
    int j;
    AgglomerativeEdge(double d, int a, int b) : dist(d), i(a), j(b) {}
};

struct AgglomerativeEdgeCompare {
    bool operator()(const AgglomerativeEdge &e1, const AgglomerativeEdge &e2);
};

class AgglomerativeClustering {
public:
    AgglomerativeClustering(int n_clusters) : n_clusters_(n_clusters) {}
    
    std::vector<std::vector<std::vector<int>>> fit(const Eigen::MatrixXd &X, std::vector<std::vector<int>>& A, int min_clusters=1);
    std::vector<std::pair<int, int>> children_;

private:


    // static int findConnectedComponents(const std::vector<std::vector<int>>& adj, std::vector<int>& labels);
    // static double wardDistance(const Eigen::VectorXd &sumI, double sizeI, const Eigen::VectorXd &sumJ, double sizeJ);

    int n_clusters_;
    std::vector<double> moments_1_;
    std::vector<Eigen::VectorXd> moments_2_;
    std::vector<bool> used_node_;
    std::vector<int> parent_;
    UnionFind uf_;
};

#endif // AGGLOMERATIVE_CLUSTERING_H