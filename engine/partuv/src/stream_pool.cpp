#include "stream_pool.h"
#include <thread>
#include <atomic>
#include <functional> // For std::hash
#include <iostream>

namespace StreamPool {
    std::vector<cudaStream_t> streams;
    std::mutex streams_mutex;
    std::condition_variable streams_cv;
    bool streams_initialized = false;

    void asyncInitializeStreams(int num_streams) {
        std::cout << "Initializing " << num_streams << " streams" << std::endl;
        std::thread([num_streams]() {
            std::vector<cudaStream_t> temp_streams;
            temp_streams.reserve(num_streams);


            for (int i = 0; i < num_streams; ++i) {
                cudaStream_t stream;
                cudaError_t err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
                if (err != cudaSuccess) {
                    // Handle error (e.g., log, throw, or skip)
                    continue;
                }
                // cudaMemPool_t mempool;
                // cudaMemPoolCreate(&mempool, nullptr);

                // cudaStreamSetAttribute(stream, cudaStreamAttributeMemPool, &mempool);

                temp_streams.push_back(stream);
            }

            {
                std::lock_guard<std::mutex> lock(streams_mutex);
                streams = std::move(temp_streams);
                streams_initialized = true;
            }
            streams_cv.notify_all();
        }).detach();
    }

    cudaStream_t getStream() {
        std::unique_lock<std::mutex> lock(streams_mutex);
        streams_cv.wait(lock, []{ return streams_initialized; });

        // Assign a stream based on thread ID hash
        thread_local size_t thread_index = 
            std::hash<std::thread::id>{}(std::this_thread::get_id()) % streams.size();
        
        return streams[thread_index];
    }


    cudaStream_t getStream(int thread_id) {
        std::unique_lock<std::mutex> lock(streams_mutex);
        streams_cv.wait(lock, []{ return streams_initialized; });

        // std::cout << "initialized " << streams.size() << " streams" << std::endl;

        // Assign a stream based on thread ID hash  
        thread_local size_t thread_index = 
            thread_id % streams.size();
        
        return streams[thread_index];
    }

    void destroyStreams() {
        std::lock_guard<std::mutex> lock(streams_mutex);
        for (auto& stream : streams) {
            cudaStreamDestroy(stream);
        }
        streams.clear();
        streams_initialized = false;
    }
}