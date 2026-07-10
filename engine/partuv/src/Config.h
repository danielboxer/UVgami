#pragma once

#include <yaml-cpp/yaml.h>
#include <string>
#include <iostream>
#include <vector>

class ConfigManager {
public:
    // Public configuration parameters.
    std::string meshPath;

    bool log_traverse_csv;

    bool pipelineOverlaps;
    double      pipelineThreshold;
    // std::vector<std::string> pipelineSimpleMethods;
    // std::vector<std::string> pipelineFullMethods;

    // New pipeline flags.
    bool checkSelfIntersection;
    bool checkNon2Manifold;

    std::string unwrapMethod;
    int         unwrapAbfIters;
    int         unwrapAggParts;   // NEW: Default set to 10
    bool        unwrapPamo;
    int         unwrapUsePamoFaceThreshold; // use pamo if the face count is greater than this threshold (default: 1000)

    // Global verbose flag.
    bool verbose;

    int componentMaxDepth;
    int parallelDepth;
    // New save_stuff flag.
    bool saveStuff;

    // Threading configuration.
    int num_omp_threads;
    int num_cuda_streams;

    // New output path.
    std::string outputPath;

    // Default constructor with default values.
    ConfigManager()
        : meshPath("default_mesh_path"),
          pipelineOverlaps(true),
          pipelineThreshold(1.25),
          checkSelfIntersection(false),
          checkNon2Manifold(false),
          unwrapMethod("abf"),
          unwrapAbfIters(10),
          unwrapPamo(false),
          unwrapAggParts(10),
          verbose(false),
          componentMaxDepth(10),
          parallelDepth(10),
          saveStuff(false),
          log_traverse_csv(false),
          num_omp_threads(8),
          num_cuda_streams(10),
          unwrapUsePamoFaceThreshold(1000)
    {}

    // Singleton accessor: created on first use.
    static ConfigManager& instance() {
        static ConfigManager instance;
        return instance;
    }

    // Load configuration from a YAML file (call once at startup).
    void loadFromFile(const std::string& filename) {
        try {
            YAML::Node config = YAML::LoadFile(filename);

            std::cout << YAML::Dump(config) << '\n';

            // Extract the mesh path if provided.
            if (config["mesh_path"])
                meshPath = config["mesh_path"].as<std::string>();

            if (config["log_traverse_csv"])
                log_traverse_csv = config["log_traverse_csv"].as<bool>();

            // Extract pipeline configuration.
            if (config["pipeline"]) {
                YAML::Node pipelineNode = config["pipeline"];
                if (pipelineNode["overlaps"])
                    pipelineOverlaps = pipelineNode["overlaps"].as<bool>();
                if (pipelineNode["threshold"])
                    pipelineThreshold = pipelineNode["threshold"].as<double>();

                // if (pipelineNode["simple_methods"]) {
                //     for (const auto &node : pipelineNode["simple_methods"]) {
                //         if (node.as<std::string>() == "unwrap_aligning_Agg"){
                //             std::cout << "Warning: unwrap_aligning_Agg is changed to Use unwrap_aligning_Agglomerative_all instead, skipping this method" << std::endl;
                //             continue;
                //         }
                //         pipelineSimpleMethods.push_back(node.as<std::string>());
                //     }
                // }
                // if (pipelineNode["full_methods"]) {
                //     for (const auto &node : pipelineNode["full_methods"]) {
                //         if (node.as<std::string>() == "unwrap_aligning_Agg"){
                //             std::cout << "Warning: unwrap_aligning_Agg is changed to Use unwrap_aligning_Agglomerative_all instead, skipping this method" << std::endl;
                //             continue;
                //         }
                //         std::string method = node.as<std::string>();
                //         if (std::find(pipelineFullMethods.begin(), pipelineFullMethods.end(), method) == pipelineFullMethods.end()) {
                //             pipelineFullMethods.push_back(method);
                //         }
                //     }
                // }
                if (pipelineNode["parallelDepth"])
                    parallelDepth = pipelineNode["parallelDepth"].as<int>();
                if (pipelineNode["checkSelfIntersection"])
                    checkSelfIntersection = pipelineNode["checkSelfIntersection"].as<bool>();
                if (pipelineNode["checkNon2Manifold"])
                    checkNon2Manifold = pipelineNode["checkNon2Manifold"].as<bool>();
                if (pipelineNode["component_maxDepth"])
                    componentMaxDepth = pipelineNode["component_maxDepth"].as<int>();
                if (pipelineNode["num_omp_threads"])
                    num_omp_threads = pipelineNode["num_omp_threads"].as<int>();
                if (pipelineNode["num_cuda_streams"])
                    num_cuda_streams = pipelineNode["num_cuda_streams"].as<int>();

            }

            // Extract unwrap configuration.
            if (config["unwrap"]) {
                YAML::Node unwrapNode = config["unwrap"];
                if (unwrapNode["method"])   
                    unwrapMethod = unwrapNode["method"].as<std::string>();
                if (unwrapNode["abf_iters"])
                    unwrapAbfIters = unwrapNode["abf_iters"].as<int>();
                if (unwrapNode["agg_parts"])   // NEW: Load unwrapAggParts if provided.
                    unwrapAggParts = unwrapNode["agg_parts"].as<int>();
                if (unwrapNode["pamo"])
                    unwrapPamo = unwrapNode["pamo"].as<bool>();
                if (unwrapNode["usePamoFaceThreshold"])
                    unwrapUsePamoFaceThreshold = unwrapNode["usePamoFaceThreshold"].as<int>();
            }

            // Extract verbose configuration.
            if (config["verbose"])
                verbose = config["verbose"].as<bool>();

            // Extract save_stuff flag.
            if (config["save_stuff"])
                saveStuff = config["save_stuff"].as<bool>();

            // Extract output_path.
            if (config["output_path"])
                outputPath = config["output_path"].as<std::string>();
            else
                outputPath = meshPath.substr(0, meshPath.find_last_of('/')) + "/output/";


        }
        catch (const YAML::Exception& e) {
            std::cerr << "Error loading config file: " << e.what() << std::endl;
            throw;
        }

        std::cout << "Configuration loaded successfully." << std::endl;
    }

    void setMeshPath(const std::string& path) {
        meshPath = path;
        outputPath = path.substr(0, path.find_last_of('/')) + "/output/";
    }
    // Function to print all configuration parameters.
    void printConfigs() const {
        std::cout << "\n[CONFIG] meshPath: " << meshPath << "\n";
        
        if (log_traverse_csv)
            std::cout << "[CONFIG] log_traverse_csv: " << log_traverse_csv << "\n";
        
        std::cout << "[CONFIG] pipeline.overlaps: " << std::boolalpha << pipelineOverlaps << "\n";
        std::cout << "[CONFIG] pipeline.threshold: " << pipelineThreshold << "\n";

        // std::cout << "[CONFIG] pipeline.simple_methods: ";
        // for (const auto& method : pipelineSimpleMethods) {
        //     std::cout << method << " ";
        // }
        // std::cout << "\n";

        // std::cout << "[CONFIG] pipeline.full_methods: ";
        // for (const auto& method : pipelineFullMethods) {
        //     std::cout << method << " ";
        // }
        // std::cout << "\n";

        std::cout << "[CONFIG] pipeline.checkSelfIntersection: " << std::boolalpha << checkSelfIntersection << "\n";
        std::cout << "[CONFIG] pipeline.checkNon2Manifold: " << std::boolalpha << checkNon2Manifold << "\n";

        std::cout << "[CONFIG] unwrap.method: " << unwrapMethod << "\n";
        std::cout << "[CONFIG] unwrap.abf_iters: " << unwrapAbfIters << "\n";
        std::cout << "[CONFIG] unwrap.pamo: " << std::boolalpha << unwrapPamo << "\n";
        std::cout << "[CONFIG] unwrap.usePamoFaceThreshold: " << unwrapUsePamoFaceThreshold << "\n";
        std::cout << "[CONFIG] unwrap.agg_parts: " << unwrapAggParts << "\n";  // NEW: Print unwrapAggParts

        std::cout << "[CONFIG] verbose: " << std::boolalpha << verbose << "\n";
        std::cout << "[CONFIG] save_stuff: " << std::boolalpha << saveStuff << "\n";
        std::cout << "[CONFIG] component_maxDepth: " << componentMaxDepth << "\n"; 
        std::cout << "[CONFIG] parallelDepth: " << parallelDepth << "\n";
        std::cout << "[CONFIG] output_path: " << outputPath << "\n";
        std::cout << "[CONFIG] num_omp_threads: " << num_omp_threads << "\n";
        std::cout << "[CONFIG] num_cuda_streams: " << num_cuda_streams << "\n";
    }

private:
    // Private constructor enforces the singleton pattern.
    // ConfigManager() = default;
};

// Global inline variables for easy access (requires C++17 or later).
inline std::string& CONFIG_meshPath             = ConfigManager::instance().meshPath;
inline bool&        CONFIG_log_traverse_csv     = ConfigManager::instance().log_traverse_csv;

inline bool&        CONFIG_pipelineOverlaps     = ConfigManager::instance().pipelineOverlaps;
inline double&      CONFIG_pipelineThreshold    = ConfigManager::instance().pipelineThreshold;
inline bool&        CONFIG_checkSelfIntersection = ConfigManager::instance().checkSelfIntersection;
inline bool&        CONFIG_checkNon2Manifold     = ConfigManager::instance().checkNon2Manifold;
inline std::string& CONFIG_unwrapMethod         = ConfigManager::instance().unwrapMethod;
inline int&         CONFIG_unwrapAbfIters       = ConfigManager::instance().unwrapAbfIters;
inline bool&        CONFIG_unwrapPamo           = ConfigManager::instance().unwrapPamo;
inline int&         CONFIG_unwrapUsePamoFaceThreshold = ConfigManager::instance().unwrapUsePamoFaceThreshold;
// End of Selection

inline int&         CONFIG_unwrapAggParts       = ConfigManager::instance().unwrapAggParts;  // NEW: Global inline variable
inline bool&        CONFIG_verbose              = ConfigManager::instance().verbose;
inline bool&        CONFIG_saveStuff            = ConfigManager::instance().saveStuff;
// inline std::vector<std::string>& CONFIG_pipelineSimpleMethods = ConfigManager::instance().pipelineSimpleMethods;
// inline std::vector<std::string>& CONFIG_pipelineFullMethods   = ConfigManager::instance().pipelineFullMethods;
inline std::string& CONFIG_outputPath           = ConfigManager::instance().outputPath;
inline int&         CONFIG_componentMaxDepth    = ConfigManager::instance().componentMaxDepth;
inline int&         CONFIG_parallelDepth        = ConfigManager::instance().parallelDepth;
inline int&         CONFIG_num_omp_threads      = ConfigManager::instance().num_omp_threads;
inline int&         CONFIG_num_cuda_streams     = ConfigManager::instance().num_cuda_streams;