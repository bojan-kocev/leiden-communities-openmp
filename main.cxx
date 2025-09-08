#include "inc/main.hxx"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

using namespace std;

#pragma region CONFIGURATION
#ifndef TYPE
/** Type of edge weights. */
#define TYPE float
#endif
#ifndef MAX_THREADS
/** Maximum number of threads to use. */
#define MAX_THREADS 64
#endif
#ifndef REPEAT_METHOD
/** Number of times to repeat each method. */
#define REPEAT_METHOD 5
#endif
#pragma endregion

// HELPERS
// -------

template <class G, class R>
inline double getModularity(const G &x, const R &a, double M) {
  auto fc = [&](auto u) { return a.membership[u]; };
  return modularityByOmp(x, fc, M, 1.0);
}

template <class K, class W>
inline float refinementTime(const LouvainResult<K, W> &a) {
  return 0;
}
template <class K, class W>
inline float refinementTime(const LeidenResult<K, W> &a) {
  return a.refinementTime;
}

template <class K, class W>
void exportLeidenResult(const LeidenResult<K, W> &result,
                        const std::string &filename) {
  std::cout << "Exporting Leiden clustering result to " << filename
            << std::endl;
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cout << "Error opening file: " << filename << std::endl;
    return;
  }

  // Save node_id, community_id
  for (size_t i = 0; i < result.membership.size(); ++i) {
    file << i << "," << result.membership[i] << "\n";
  }

  file.close();
  std::cout << "Leiden clustering result saved to " << filename << std::endl;
}

// PERFORM EXPERIMENT
// ------------------

template <class G>
void runExperiment(const G &x, const std::string &output_file) {
  std::cout << "Running the experiment ..." << std::endl;
  int repeat = REPEAT_METHOD;
  double M = edgeWeightOmp(x) / 2;
  // Follow a specific result logging format, which can be easily parsed later.
  auto flog = [&](const auto &ans, const char *technique) {
    printf("{%09.1fms, %09.1fms mark, %09.1fms init, %09.1fms firstpass, "
           "%09.1fms locmove, %09.1fms refine, %09.1fms aggr, %.3e aff, %04d "
           "iters, %03d passes, %01.9f modularity, %zu/%zu disconnected} %s\n",
           ans.time, ans.markingTime, ans.initializationTime, ans.firstPassTime,
           ans.localMoveTime, refinementTime(ans), ans.aggregationTime,
           double(ans.affectedVertices), ans.iterations, ans.passes,
           getModularity(x, ans, M),
           countValue(communitiesDisconnectedOmp(x, ans.membership), char(1)),
           communities(x, ans.membership).size(), technique);
  };
  // Get community memberships on original graph (static).
  {
    auto a0 = louvainStaticOmp(x, {repeat});
    flog(a0, "louvainStaticOmp");
    auto b0 = leidenStaticOmp(x, {repeat});
    flog(b0, "leidenStaticOmp");
    exportLeidenResult(b0, output_file);
  }
}

int main(int argc, char **argv) {
  using K = int;
  using V = TYPE;
  install_sigsegv();
  char *file = argv[1];
  char *output_file = argv[2];
  bool symmetric = argc > 3 ? stoi(argv[3]) : false;
  bool weighted = argc > 4 ? stoi(argv[4]) : false;
  int processes = omp_get_num_procs();
  omp_set_num_threads(processes);
  std::cout << "omp version = " << _OPENMP << std::endl;
  LOG("OMP_NUM_THREADS=%d\n", processes);
  LOG("Loading graph %s ...\n", file);
  DiGraph<K, None, V> x;
  readMtxOmpW(x, file, weighted);
  LOG("");
  println(x);
  LOG("Loaded graph %s ...\n", file);
  if (!symmetric) {
    x = symmetricizeOmp(x);
    LOG("");
    print(x);
    printf(" (symmetricize)\n");
  }
  runExperiment(x, std::string(output_file));
  printf("\n");
  return 0;
}
