#include <atomic>
#include <thread>
#include <chrono>
#include <set>
#include "fanns_survey_helpers.cpp"
#include "global_thread_counter.h"
#include "iRG_search.h"

std::unordered_map<std::string, std::string> paths;

const int query_K = 10;
int M;
int ef_search;

// Global atomic to store peak thread count
std::atomic<int> peak_threads(1);

int main(int argc, char **argv)
{
    for (int i = 0; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "--data_path")
            paths["data_vector"] = argv[i + 1];
        if (arg == "--query_path")
            paths["query_vector"] = argv[i + 1];
        if (arg == "--query_ranges_file")
            paths["query_ranges"] = argv[i + 1];
        if (arg == "--groundtruth_file")
            paths["groundtruth"] = argv[i + 1];
        if (arg == "--index_file")
            paths["index"] = argv[i + 1];
        if (arg == "--M")
            M = std::stoi(argv[i + 1]);
        if (arg == "--ef_search")
            ef_search = std::stoi(argv[i + 1]);
    }

    if (paths["data_vector"] == "")
        throw Exception("data path is empty");
    if (paths["query_vector"] == "")
        throw Exception("query path is empty");
    if (paths["query_ranges"] == "")
        throw Exception("query ranges file is empty");
    if (paths["groundtruth"] == "")
        throw Exception("groundtruth file is empty");
    if (paths["index"] == "")
        throw Exception("index path is empty");
    if (M <= 0)
        throw Exception("M should be a positive integer");
    if (ef_search <= 0)
        throw Exception("ef_search should be a positive integer");

    // Monitor thread count
    std::atomic<bool> done(false);
    std::thread monitor(monitor_thread_count, std::ref(done));

    // Load the index and data
    iRangeGraph::DataLoader storage;
    storage.query_K = query_K;
    storage.LoadQuery(paths["query_vector"]);
    
    // Read query ranges from CSV file (format: "low-high" per line)
    std::vector<std::pair<int, int>> query_ranges = read_two_ints_per_line(paths["query_ranges"]);
    
    // Read groundtruth from ivecs file (contains IDs in ORIGINAL unsorted order)
    std::vector<std::vector<int>> groundtruth = read_ivecs(paths["groundtruth"]);
    
    if (query_ranges.size() != storage.query_nb) {
        throw Exception("Number of query ranges does not match number of queries");
    }
    if (groundtruth.size() != storage.query_nb) {
        throw Exception("Number of groundtruth entries does not match number of queries");
    }

    // Load the ID mapping: sorted_index -> original_index
    // This translates from sorted database IDs (used by iRangeGraph) to original IDs (used in groundtruth)
    std::string mapping_file = paths["data_vector"] + ".mapping";
    std::ifstream mapping_in(mapping_file, std::ios::binary);
    if (!mapping_in) {
        throw Exception("Unable to open mapping file: " + mapping_file);
    }
    int num_points;
    mapping_in.read(reinterpret_cast<char*>(&num_points), sizeof(int));
    std::vector<size_t> sorted_to_original(num_points);
    mapping_in.read(reinterpret_cast<char*>(sorted_to_original.data()), num_points * sizeof(size_t));
    mapping_in.close();
    std::cout << "Loaded ID mapping from " << mapping_file << " (" << num_points << " points)" << std::endl;

    // Load the index
    iRangeGraph::iRangeGraph_Search<float> index(paths["data_vector"], paths["index"], &storage, M);

    // Start timing
    auto start_time = std::chrono::high_resolution_clock::now();

    int total_true_positives = 0;
    int total_queries_processed = 0;

    // Execute queries with single ef_search value
    for (int i = 0; i < storage.query_nb; i++)
    {
        auto range_pair = query_ranges[i];
        int ql = range_pair.first;
        int qr = range_pair.second;

        // Perform the search
        std::vector<iRangeGraph::TreeNode*> filterednodes = index.tree->range_filter(index.tree->root, ql, qr);
        std::priority_queue<iRangeGraph::PFI> res = index.TopDown_nodeentries_search(
            filterednodes, 
            storage.query_points[i].data(), 
            ef_search,  // Use the ef_search parameter
            query_K, 
            ql, 
            qr, 
            M  // edge_limit = M
        );

        // Calculate recall for this query
        // Results from iRangeGraph are in sorted database space - need to convert to original space
        std::set<int> result_set_original;
        while (!res.empty())
        {
            int sorted_id = res.top().second;
            int original_id = sorted_to_original[sorted_id];  // Translate to original ID
            result_set_original.insert(original_id);
            res.pop();
        }

        // Count true positives - groundtruth is in original ID space
        for (int gt_id : groundtruth[i])
        {
            if (result_set_original.count(gt_id))
                total_true_positives++;
        }
        total_queries_processed++;
    }

    // Stop timing
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // Stop monitoring
    done = true;
    monitor.join();

    // Calculate metrics
    float recall = (float)total_true_positives / (total_queries_processed * query_K);
    float qps = total_queries_processed / elapsed.count();

    // Print statistics in the expected format
    std::cout << "Query execution completed." << std::endl;
    std::cout << "Query time (s): " << elapsed.count() << std::endl;
    std::cout << "Peak thread count: " << peak_threads.load() << std::endl;
    std::cout << "QPS: " << qps << std::endl;
    std::cout << "Recall: " << recall << std::endl;
    
    // Print memory footprint
    peak_memory_footprint();

    return 0;
}
