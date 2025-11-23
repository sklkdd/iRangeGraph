#include "construction.h"
#include <atomic>
#include <thread>
#include <chrono>
#include "fanns_survey_helpers.cpp"
#include "global_thread_counter.h"

std::unordered_map<std::string, std::string> paths;

int M;
int ef_construction;
int threads;

// Global atomic to store peak thread count
std::atomic<int> peak_threads(1);

int main(int argc, char **argv)
{
    for (int i = 0; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--data_path")
            paths["data_vector"] = argv[i + 1];
        if (arg == "--index_file")
            paths["index_save"] = argv[i + 1];
        if (arg == "--M")
            M = std::stoi(argv[i + 1]);
        if (arg == "--ef_construction")
            ef_construction = std::stoi(argv[i + 1]);
        if (arg == "--threads")
            threads = std::stoi(argv[i + 1]);
    }

    if (paths["data_vector"] == "")
        throw Exception("data path is empty");
    if (paths["index_save"] == "")
        throw Exception("index path is empty");
    if (M <= 0)
        throw Exception("M should be a positive integer");
    if (ef_construction <= 0)
        throw Exception("ef_construction should be a positive integer");
    if (threads <= 0)
        throw Exception("threads should be a positive integer");

    // Monitor thread count
    std::atomic<bool> done(false);
    std::thread monitor(monitor_thread_count, std::ref(done));

    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    iRangeGraph::DataLoader storage;
    storage.LoadData(paths["data_vector"]);
    iRangeGraph::iRangeGraph_Build<float> index(&storage, M, ef_construction);
    index.max_threads = threads;
    index.buildandsave(paths["index_save"]);

    // Stop timing
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // Stop monitoring
    done = true;
    monitor.join();

    // Print statistics in the expected format
    std::cout << "Index construction completed." << std::endl;
    std::cout << "Build time (s): " << elapsed.count() << std::endl;
    std::cout << "Peak thread count: " << peak_threads.load() << std::endl;
    
    // Print memory footprint
    peak_memory_footprint();

    return 0;
}
