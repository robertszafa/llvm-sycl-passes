
#include "store_queue.hpp"
#include <cstdint>
#include <type_traits>

#include "pipe_utils.hpp"
#include "tuple.hpp"
#include "unrolled_loop.hpp"
#include "constexpr_math.hpp"
using namespace fpga_tools;

template <typename T>
struct st_val_pair { T first;  int second; };

class MainKernel_AGU;
#include <CL/sycl.hpp>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <vector>
#include <random>

#include <sycl/ext/intel/fpga_extensions.hpp>

#include "store_queue.hpp"
#include "memory_utils.hpp"

using namespace sycl;

// Forward declare kernel name.
class MainKernel;

double histogram_kernel(queue &q, const std::vector<int> &h_idx, std::vector<float> &h_hist) {
  const int array_size = h_idx.size();

  int *idx = fpga_tools::toDevice(h_idx, q);
  float *hist = fpga_tools::toDevice(h_hist, q);


        using val_type_0 = float;
        constexpr int kNumLoads_0 = 1;
        constexpr int kNumStores_0 = 11;
        constexpr int kQueueSize_0 = 8;

        using ld_idx_pipes_0 = PipeArray<class ld_idx_pipes_0_class, request_lsq_t, 64, kNumLoads_0>;
        using ld_val_pipes_0 = PipeArray<class ld_val_pipes_0_class, val_type_0, 64, kNumLoads_0>;
        using st_idx_pipe_0 = pipe<class st_idx_pipe_0_class, request_lsq_t, 64>;
        using st_val_pipe_0 = pipe<class st_val_pipe_0_class, val_type_0, 64>;
        using end_lsq_signal_pipe_0 = pipe<class end_lsq_signal_pipe_0_class, int>;
        
        auto eventStoreQueue_0 = 
            LoadStoreQueue<val_type_0, ld_idx_pipes_0, ld_val_pipes_0, 
                           st_idx_pipe_0, st_val_pipe_0, end_lsq_signal_pipe_0, 
                           kNumLoads_0, kQueueSize_0>(q);
        
            using st_req_pipes_0 = PipeArray<class st_req_pipes_0_class, request_lsq_t, 64, kNumStores_0>;
            using st_val_pipes_0 = PipeArray<class st_val_pipes_0_class, st_val_pair<float>, 64, kNumStores_0>;
            
            using end_st_val_merge_pipe_0 = pipe<class end_st_val_merge_pipe_0_class, int>;
            using end_st_req_merge_pipe_0 = pipe<class end_st_req_merge_pipe_0_class, int>;
            


q.submit([&](handler &hnd) {
            hnd.single_task<class StoreReqMerge_0>([=]() [[intel::kernel_args_restrict]] {
            // int nextTag = 1;
            // constexpr int kNumStores = kNumStores_0;
            // NTuple<bool, kNumStores> flags;
            // UnrolledLoop<kNumStores>([&](auto k) {
            //     flags. template get<k>() = false;
            // });

            // NTuple<request_lsq_t, kNumStores> stReqs;

            // bool end = false;
            // int maxTag;
            // //[[intel::ivdep]]
            // // [[intel::initiation_interval(2)]]
            // while (!end || nextTag <= maxTag) {
            //     UnrolledLoop<kNumStores>([&](auto k) {
            //     bool succ = false;
            //     if (!flags. template get<k>()) {
            //         stReqs. template get<k>() = st_req_pipes_0::PipeAt<k>::read(succ);
            //         flags. template get<k>() = succ;
            //     }
            //     });
                
            //     request_lsq_t nextReq;
            //     bool gotNext = false;
            //     UnrolledLoop<kNumStores>([&](auto k) {
            //     if (flags. template get<k>() && stReqs. template get<k>().second == nextTag) {
            //         nextReq = stReqs. template get<k>();            
            //         flags. template get<k>() = false;
            //         gotNext = true;
            //     }
            //     });


            //     if (gotNext) {
            //         // PRINTF("nextTag REQ %d\n", nextTag);
            //         st_idx_pipe_0::write(nextReq);
            //         nextTag++;
            //     }
                
            //     if (!end) 
            //         maxTag = end_st_req_merge_pipe_0::read(end);
            // }
            // PRINTF("maxTag REQ %d\n", maxTag);
            // end_lsq_signal_pipe_0::write(maxTag);
            
            for (int i = 0; i < array_size - 11; ++i) {
              [[intel::ivdep]]
              for (int j = 0; j < kNumStores_0; ++j) {
                request_lsq_t req;
                UnrolledLoop<kNumStores_0>([&](auto k) {
                  if (k == j)
                    req = st_req_pipes_0::PipeAt<k>::read();
                });
                
                st_idx_pipe_0::write(req);
              }
            }
            });});

q.submit([&](handler &hnd) {
            hnd.single_task<class StoreValueMerge_0>([=]() [[intel::kernel_args_restrict]] {
            int nextTag = 1;
            constexpr int kNumStores = kNumStores_0;
            NTuple<bool, kNumStores> flags;
            UnrolledLoop<kNumStores>([&](auto k) {
                flags. template get<k>() = false;
            });

            NTuple<st_val_pair<float>, kNumStores> stVals;

            bool end = false;
            int maxTag;
            //[[intel::ivdep]]
            // [[intel::initiation_interval(2)]]
            // while (!end || nextTag <= maxTag) {
            //     UnrolledLoop<kNumStores>([&](auto k) {
            //     bool succ = false;
            //     if (!flags. template get<k>()) {
            //         stVals. template get<k>() = st_val_pipes_0::PipeAt<k>::read(succ);
            //         flags. template get<k>() = succ;
            //     }
            //     });
                
            //     st_val_pair<float> nextVal;
            //     bool gotNext = false;
            //     UnrolledLoop<kNumStores>([&](auto k) {
            //     if (flags. template get<k>() && stVals. template get<k>().second == nextTag) {
            //         nextVal = stVals. template get<k>();            
            //         flags. template get<k>() = false;
            //         gotNext = true;
            //     }
            //     });

            //     if (gotNext) {
            //         // PRINTF("nextTag VAL %d\n", nextTag);
            //         st_val_pipe_0::write(nextVal.first);
            //         nextTag++;
            //     }
                
            //     if (!end)
            //         maxTag = end_st_val_merge_pipe_0::read(end);
            // }

            // [[intel::ivdep]]
            for (int i = 0; i < array_size - 11; ++i) {
              [[intel::ivdep]]
              for (int j = 0; j < kNumStores_0; ++j) {
                float val;
                UnrolledLoop<kNumStores>([&](auto k) {
                  if (k == j)
                    val = st_val_pipes_0::PipeAt<k>::read().first;
                });
                
                st_val_pipe_0::write(val);
              }
            }
            // PRINTF("maxTag VAL %d\n", maxTag);
            });});

auto event = q.submit([&](handler &hnd) {
  hnd.single_task<MainKernel>([=]() [[intel::kernel_args_restrict]] {
    /////////////////////////////////// KERNEL CODE /////////////////////////////////////////////
    int tag = 0;
    [[intel::ivdep]]
    for (int i = 0; i < array_size - 11; ++i) {
      auto idx_scalar = idx[i];
      ld_idx_pipes_0::PipeAt<0>::write({(int64_t)(hist + idx_scalar), tag});
      auto x = ld_val_pipes_0::PipeAt<0>::read();

      tag++;
      st_req_pipes_0::PipeAt<0>::write({(int64_t)(hist + idx_scalar), tag});
      st_val_pipes_0::PipeAt<0>::write({x * 10.0f, tag});
      // st_idx_pipe_0::write({(int64_t)(hist + idx_scalar), tag});
      // st_val_pipe_0::write(x * 10.0f);

      tag++;
      st_req_pipes_0::PipeAt<1>::write({(int64_t)(hist + idx_scalar + 1), tag});
      st_val_pipes_0::PipeAt<1>::write({x * 10.0f, tag});
      // st_idx_pipe_0::write({(int64_t)(hist + idx_scalar + 1), tag});
      // st_val_pipe_0::write(x * 10.0f);

      tag++;
      st_req_pipes_0::PipeAt<2>::write({(int64_t)(hist + idx_scalar + 2), tag});
      st_val_pipes_0::PipeAt<2>::write({x * 10.0f, tag});
      // st_idx_pipe_0::write({(int64_t)(hist + idx_scalar + 2), tag});
      // st_val_pipe_0::write(x * 10.0f);

      tag++;
      st_req_pipes_0::PipeAt<3>::write({(int64_t)(hist + idx_scalar + 3), tag});
      st_val_pipes_0::PipeAt<3>::write({x * 10.0f, tag});
      // st_idx_pipe_0::write({(int64_t)(hist + idx_scalar + 3), tag});
      // st_val_pipe_0::write(x * 10.0f);

      tag++;
      st_req_pipes_0::PipeAt<4>::write({(int64_t)(hist + idx_scalar + 4), tag});
      st_val_pipes_0::PipeAt<4>::write({x * 10.0f, tag});
      // st_idx_pipe_0::write({(int64_t)(hist + idx_scalar + 4), tag});
      // st_val_pipe_0::write(x * 10.0f);

      tag++;
      st_req_pipes_0::PipeAt<5>::write({(int64_t)(hist + idx_scalar + 5), tag});
      st_val_pipes_0::PipeAt<5>::write({x * 10.0f, tag});
      // st_idx_pipe_0::write({(int64_t)(hist + idx_scalar + 5), tag});
      // st_val_pipe_0::write(x * 10.0f);

      tag++;
      st_req_pipes_0::PipeAt<6>::write({(int64_t)(hist + idx_scalar + 6), tag});
      st_val_pipes_0::PipeAt<6>::write({x * 10.0f, tag});
      // st_idx_pipe_0::write({(int64_t)(hist + idx_scalar + 6), tag});
      // st_val_pipe_0::write(x * 10.0f);

      tag++;
      st_req_pipes_0::PipeAt<7>::write({(int64_t)(hist + idx_scalar + 7), tag});
      st_val_pipes_0::PipeAt<7>::write({x * 10.0f, tag});
      // st_idx_pipe_0::write({(int64_t)(hist + idx_scalar + 7), tag});
      // st_val_pipe_0::write(x * 10.0f);

      tag++;
      st_req_pipes_0::PipeAt<8>::write({(int64_t)(hist + idx_scalar + 8), tag});
      st_val_pipes_0::PipeAt<8>::write({x * 10.0f, tag});
      // st_idx_pipe_0::write({(int64_t)(hist + idx_scalar + 8), tag});
      // st_val_pipe_0::write(x * 10.0f);

      tag++;
      st_req_pipes_0::PipeAt<9>::write({(int64_t)(hist + idx_scalar + 9), tag});
      st_val_pipes_0::PipeAt<9>::write({x * 10.0f, tag});
      // st_idx_pipe_0::write({(int64_t)(hist + idx_scalar + 9), tag});
      // st_val_pipe_0::write(x * 10.0f);

      tag++;
      st_req_pipes_0::PipeAt<10>::write({(int64_t)(hist + idx_scalar + 10), tag});
      st_val_pipes_0::PipeAt<10>::write({x * 10.0f, tag});
      // st_idx_pipe_0::write({(int64_t)(hist + idx_scalar + 10), tag});
      // st_val_pipe_0::write(x * 10.0f);
    }

    // end_st_req_merge_pipe_0::write(tag);
    // end_st_val_merge_pipe_0::write(tag);
    end_lsq_signal_pipe_0::write(tag);
    /////////////////////////////////// KERNEL CODE /////////////////////////////////////////////
  });
});

event.wait();
eventStoreQueue_0.wait();
  q.copy(hist, h_hist.data(), h_hist.size()).wait();

  sycl::free(idx, q);
  sycl::free(hist, q);

  auto start = event.get_profiling_info<info::event_profiling::command_start>();
  auto end = event.get_profiling_info<info::event_profiling::command_end>();
  double time_in_ms = static_cast<double>(end - start) / 1000000;

  return time_in_ms;
}

void histogram_cpu(const int *idx, float *hist, const int N) {
  for (int i = 0; i < N; ++i) {
    auto idx_scalar = idx[i];
    auto x = hist[idx_scalar];
    hist[idx_scalar] = x + 10.0;
  }
}

enum data_distribution { ALL_WAIT, NO_WAIT, PERCENTAGE_WAIT };
void init_data(std::vector<int> &feature, std::vector<float> &hist,
               const data_distribution distr, const int percentage) {
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0, 100);
  auto dice = std::bind (distribution, generator);

  int counter=0;
  for (int i = 0; i < feature.size(); i++) {
    if (distr == data_distribution::ALL_WAIT) {
      feature[i] = (feature.size() >= 4) ? i % 4 : 0;
    }
    else if (distr == data_distribution::NO_WAIT) {
      feature[i] = i;
    }
    else {
      feature[i] = (dice() <= percentage) ? feature[std::max(i-1, 0)] : i;
    }

    hist[i] = 0.0;
  }
}

// Create an exception handler for asynchronous SYCL exceptions
static auto exception_handler = [](sycl::exception_list e_list) {
  for (std::exception_ptr const &e : e_list) {
    try {
      std::rethrow_exception(e);
    } catch (std::exception const &e) {
#if _DEBUG
      std::cout << "Failure" << std::endl;
#endif
      std::terminate();
    }
  }
};

int main(int argc, char *argv[]) {
  // Get A_SIZE and forward/no-forward from args.
  int ARRAY_SIZE = 64;
  auto DATA_DISTR = data_distribution::ALL_WAIT;
  int PERCENTAGE = 5;
  try {
    if (argc > 1) {
      ARRAY_SIZE = int(atoi(argv[1]));
    }
    if (argc > 2) {
      DATA_DISTR = data_distribution(atoi(argv[2]));
    }
    if (argc > 3) {
      PERCENTAGE = int(atoi(argv[3]));
      std::cout << "Percentage is " << PERCENTAGE << "\n";
      if (PERCENTAGE < 0 || PERCENTAGE > 100)
        throw std::invalid_argument("Invalid percentage.");
    }
  } catch (exception const &e) {
    std::cout << "Incorrect argv.\nUsage:\n";
    std::cout << "  ./executable [ARRAY_SIZE] [data_distribution (0/1/2)] [PERCENTAGE (only for "
                 "data_distr 2)]\n";
    std::cout << "    0 - all_wait, 1 - no_wait, 2 - PERCENTAGE wait\n";
    std::terminate();
  }

#if FPGA_EMULATOR
  ext::intel::fpga_emulator_selector d_selector;
#elif FPGA
  ext::intel::fpga_selector d_selector;
#else
  default_selector d_selector;
#endif
  try {
    // Enable profiling.
    property_list properties{property::queue::enable_profiling()};
    queue q(d_selector, exception_handler, properties);

    // Print out the device information used for the kernel code.
    std::cout << "Running on device: " << q.get_device().get_info<info::device::name>() << "\n";

    std::vector<int> feature(ARRAY_SIZE);
    std::vector<float> hist(ARRAY_SIZE);
    std::vector<float> hist_cpu(ARRAY_SIZE);

    init_data(feature, hist, DATA_DISTR, PERCENTAGE);
    std::copy(hist.begin(), hist.end(), hist_cpu.begin());

    auto start = std::chrono::steady_clock::now();
    double kernel_time = 0;

    kernel_time = histogram_kernel(q, feature, hist);

    // Wait for all work to finish.
    q.wait();

    histogram_cpu(feature.data(), hist_cpu.data(), hist_cpu.size());

    std::cout << "\nKernel time (ms): " << kernel_time << "\n";

    if (std::equal(hist.begin(), hist.end(), hist_cpu.begin())) {
      std::cout << "Passed\n";
    } else {
      std::cout << "Failed\n";
      std::cout << "sum(fpga) = " << std::accumulate(hist.begin(), hist.end(), 0.0) << "\n";
      std::cout << "sum(cpu) = " << std::accumulate(hist_cpu.begin(), hist_cpu.end(), 0.0) << "\n";
    }
  } catch (exception const &e) {
    std::cout << "An exception was caught.\n";
    std::terminate();
  }

  return 0;
}
