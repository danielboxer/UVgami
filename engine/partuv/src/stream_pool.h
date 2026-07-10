#pragma once
#include <vector>
#include <mutex>
#include <condition_variable>
#include <cuda_runtime.h>

namespace StreamPool {
    // Initialize the stream pool with 'num_streams' streams (call this once from main)
    void asyncInitializeStreams(int num_streams);

    // Get a stream for the current thread (thread-safe)
    cudaStream_t getStream();

    cudaStream_t getStream(int thread_id);

    // Cleanup all streams (call at program exit)
    void destroyStreams();
}