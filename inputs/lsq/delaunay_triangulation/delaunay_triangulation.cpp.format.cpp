#include <CL/sycl.hpp>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <stdlib.h>
#include <vector>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include "memory_utils.hpp"
#include <limits>
#include <math.h>
namespace dt {
// TODO: use structs and combine it with the Scalar Replacement of Aggregates
// pass in LLVM so that each struct gets its own LSQ.
// template <typename T> struct Point2D {
//   T x;
//   T y;
// };
// struct Edge {
//   /// Store indexes into points[]
//   int fromPoint;
//   int toPoint;
//   bool isBad = false;
// };
// struct Triangle {
//   /// Store indexes into points[]
//   int aIdx;
//   int bIdx;
//   int cIdx;
//   bool isBad = false;
// };
inline bool almost_equal(const float x, const float y) {
  float ulpFloat = static_cast<float>(2);
  return fabsf(x - y) <= std::numeric_limits<float>::epsilon() * fabsf(x + y) * ulpFloat || fabsf(x - y) < std::numeric_limits<float>::min();
}
template <typename T> inline bool almost_equal(const T v1_x, const T v1_y, const T v2_x, const T v2_y) { return almost_equal(v1_x, v2_x) && almost_equal(v1_y, v2_y); }
template <typename T> inline bool almost_equal(const int &e1_fromPoint, const int &e1_toPoint, const int &e2_fromPoint, const int &e2_toPoint, const T *pointsX, const T *pointsY) { return (almost_equal(pointsX[e1_fromPoint], pointsY[e1_fromPoint], pointsX[e2_fromPoint], pointsY[e2_fromPoint]) && almost_equal(pointsX[e1_toPoint], pointsY[e1_toPoint], pointsX[e2_toPoint], pointsY[e2_toPoint])) || (almost_equal(pointsX[e1_toPoint], pointsY[e1_toPoint], pointsX[e2_fromPoint], pointsY[e2_fromPoint]) && almost_equal(pointsX[e1_fromPoint], pointsY[e1_fromPoint], pointsX[e2_fromPoint], pointsY[e2_fromPoint])); }
template <typename T> inline bool containsPoint(const T *points_x, const T *points_y, const int triangle_aIdx, const int triangle_bIdx, const int triangle_cIdx, const T point_x, const T point_y) { return almost_equal(points_x[triangle_aIdx], points_y[triangle_aIdx], point_x, point_y) || almost_equal(points_x[triangle_bIdx], points_y[triangle_bIdx], point_x, point_y) || almost_equal(points_x[triangle_cIdx], points_y[triangle_cIdx], point_x, point_y); }
template <typename T> T norm2(const T p_x, const T p_y) { return p_x * p_x + p_y * p_y; }
template <typename T> inline T dist2(const T p1_x, const T p1_y, const T p2_x, const T p2_y) {
  const T dx = p1_x - p2_x;
  const T dy = p1_y - p2_y;
  return dx * dx + dy * dy;
}
template <typename T> inline bool circumCircleContains(const int triangle_aIdx, const int triangle_bIdx, const int triangle_cIdx, const T point_x, const T point_y, const T *points_x, const T *points_y) {
  const T ab = norm2(points_x[triangle_aIdx], points_y[triangle_aIdx]);
  const T cd = norm2(points_x[triangle_bIdx], points_x[triangle_bIdx]);
  const T ef = norm2(points_x[triangle_cIdx], points_x[triangle_cIdx]);
  const T ax = points_x[triangle_aIdx];
  const T ay = points_y[triangle_aIdx];
  const T bx = points_x[triangle_bIdx];
  const T by = points_y[triangle_bIdx];
  const T cx = points_x[triangle_cIdx];
  const T cy = points_y[triangle_cIdx];
  const T circum_x = (ab * (cy - by) + cd * (ay - cy) + ef * (by - ay)) / (ax * (cy - by) + bx * (ay - cy) + cx * (by - ay));
  const T circum_y = (ab * (cx - bx) + cd * (ax - cx) + ef * (bx - ax)) / (ay * (cx - bx) + by * (ax - cx) + cy * (bx - ax));
  const T circum_radius = dist2(points_x[triangle_aIdx], points_y[triangle_aIdx], circum_x / 2, circum_y / 2);
  const T dist = dist2(point_x, point_y, circum_x / 2, circum_y / 2);
  return dist <= circum_radius;
}
} // namespace dt
using namespace fpga_tools;
using namespace sycl;
class MainKernel;
double dt_k(queue &q, std::vector<float> &h_pointsX, std::vector<float> &h_pointsY, std::vector<int> &h_trianglesA, std::vector<int> &h_trianglesB, std::vector<int> &h_trianglesC, std::vector<int8_t> &h_trianglesIsBad, const int numPoints) {
  float *pointsX = fpga_tools::toDevice(h_pointsX.data(), h_pointsX.size(), q);
  float *pointsY = fpga_tools::toDevice(h_pointsY.data(), h_pointsY.size(), q);
  int *trianglesA = fpga_tools::toDevice(h_trianglesA.data(), h_trianglesA.size(), q);
  int *trianglesB = fpga_tools::toDevice(h_trianglesB.data(), h_trianglesB.size(), q);
  int *trianglesC = fpga_tools::toDevice(h_trianglesC.data(), h_trianglesC.size(), q);
  int8_t *trianglesIsBad = fpga_tools::toDevice(h_trianglesIsBad.data(), h_trianglesIsBad.size(), q);
  int *polygonFromPoint = sycl::malloc_device<int>(h_trianglesA.size() * 3, q);
  int *polygonToPoint = sycl::malloc_device<int>(h_trianglesA.size() * 3, q);
  int8_t *polygonIsBad = sycl::malloc_device<int8_t>(h_trianglesA.size() * 3, q);
  // struct Edge {
  //   /// Store indexes into points[]
  //   int fromPoint;
  //   int toPoint;
  //   bool isBad = false;
  // };
  // struct Triangle {
  //   /// Store indexes into points[]
  //   int aIdx;
  //   int bIdx;
  //   int cIdx;
  //   bool isBad = false;
  // };
  auto event = q.single_task<MainKernel>([=]() [[intel::kernel_args_restrict]] {
    int numTriangles = 1;
    for (int iP = 0; iP < numPoints; ++iP) {
      // std::vector<EdgeType> polygon;
      int numEdgesInPolygon = 0;
      // [[intel::ivdep]]
      for (int iT = 0; iT < numTriangles; ++iT) {
        if (dt::circumCircleContains(trianglesA[iT], trianglesB[iT], trianglesC[iT], pointsX[iP], pointsY[iP], pointsX, pointsY)) {
          trianglesIsBad[iT] = true;
          polygonFromPoint[numEdgesInPolygon] = trianglesA[iT];
          polygonToPoint[numEdgesInPolygon] = trianglesB[iT];
          polygonIsBad[numEdgesInPolygon] = false;
          numEdgesInPolygon++;
          polygonFromPoint[numEdgesInPolygon] = trianglesB[iT];
          polygonToPoint[numEdgesInPolygon] = trianglesC[iT];
          polygonIsBad[numEdgesInPolygon] = false;
          numEdgesInPolygon++;
          polygonFromPoint[numEdgesInPolygon] = trianglesC[iT];
          polygonToPoint[numEdgesInPolygon] = trianglesA[iT];
          polygonIsBad[numEdgesInPolygon] = false;
          numEdgesInPolygon++;
        }
      }
      // Delete bad triangles.
      int numGoodTriangles = 0;
      // [[intel::ivdep]]
      for (int iT = 0; iT < numTriangles; ++iT) {
        if (!trianglesIsBad[iT]) {
          trianglesA[numGoodTriangles] = trianglesA[iT];
          trianglesB[numGoodTriangles] = trianglesB[iT];
          trianglesC[numGoodTriangles] = trianglesC[iT];
          trianglesIsBad[numGoodTriangles] = trianglesIsBad[iT];
          numGoodTriangles++;
        }
      }
      // Mark bad polygons.
      for (int iE1 = 0; iE1 < numEdgesInPolygon; ++iE1) {
        // [[intel::ivdep]]
        for (int iE2 = iE1 + 1; iE2 < numEdgesInPolygon; ++iE2) {
          if (dt::almost_equal(polygonFromPoint[iE1], polygonToPoint[iE1], polygonFromPoint[iE2], polygonToPoint[iE2], pointsX, pointsY)) {
            polygonIsBad[iE1] = true;
            polygonIsBad[iE2] = true;
          }
        }
      }
      // Add new triangles.
      // [[intel::ivdep]]
      for (int iE = 0; iE < numEdgesInPolygon; ++iE) {
        if (!polygonIsBad[iE]) {
          trianglesA[numGoodTriangles] = polygonFromPoint[iE];
          trianglesB[numGoodTriangles] = polygonToPoint[iE];
          trianglesC[numGoodTriangles] = iP;
          trianglesIsBad[numGoodTriangles] = false;
          numGoodTriangles++;
        }
      }
      numTriangles = numGoodTriangles;
    }
    // Remove triangles containing points from the supertriangle.
    for (int iT = 0; iT < numTriangles; ++iT) {
      if (dt::containsPoint(pointsX, pointsY, trianglesA[iT], trianglesB[iT], trianglesC[iT], pointsX[numPoints], pointsY[numPoints]) || dt::containsPoint(pointsX, pointsY, trianglesA[iT], trianglesB[iT], trianglesC[iT], pointsX[numPoints + 1], pointsY[numPoints + 1]) || dt::containsPoint(pointsX, pointsY, trianglesA[iT], trianglesB[iT], trianglesC[iT], pointsX[numPoints + 2], pointsY[numPoints + 2])) {
        trianglesIsBad[iT] = true;
      }
    }
    // ext::oneapi::experimental::printf("Exiting..\n");
    // ext::oneapi::experimental::printf("Num tra %d\n", numTriangles);
  });
  event.wait();
  q.copy(trianglesA, h_trianglesA.data(), h_trianglesA.size()).wait();
  q.copy(trianglesB, h_trianglesB.data(), h_trianglesB.size()).wait();
  q.copy(trianglesC, h_trianglesC.data(), h_trianglesC.size()).wait();
  q.copy(trianglesIsBad, h_trianglesIsBad.data(), h_trianglesIsBad.size()).wait();
  // sycl::free(pointsX, q);
  // sycl::free(pointsY, q);
  // sycl::free(polygonFromPoint, q);
  // sycl::free(polygonToPoint, q);
  // sycl::free(polygonIsBad, q);
  // sycl::free(trianglesA, q);
  // sycl::free(trianglesB, q);
  // sycl::free(trianglesC, q);
  // sycl::free(trianglesIsBad, q);
  auto start = event.get_profiling_info<info::event_profiling::command_start>();
  auto end = event.get_profiling_info<info::event_profiling::command_end>();
  double time_in_ms = static_cast<double>(end - start) / 1000000;
  return time_in_ms;
}
enum data_distribution { ALL_WAIT, NO_WAIT, PERCENTAGE_WAIT };
template <typename T> void init_data(std::vector<T> &pointsX, std::vector<T> &pointsY, std::vector<int> &trianglesA, std::vector<int> &trianglesB, std::vector<int> &trianglesC, std::vector<int8_t> &trianglesIsBad, const int numPoints, const data_distribution distr, const int percentage) {
  int max = 1000, min = 50;
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution((int)min, (int)max);
  auto dice = std::bind(distribution, generator);
  // Determinate the super triangle
  T minX = max, minY = max, maxX = min, maxY = min;
  for (int i = 0; i < pointsX.size(); ++i) {
    pointsX[i] = T(dice());
    pointsY[i] = T(dice());
    if (pointsX[i] < minX)
      minX = pointsX[i];
    if (pointsY[i] < minY)
      minY = pointsY[i];
    if (pointsX[i] > maxX)
      maxX = pointsX[i];
    if (pointsY[i] > maxY)
      maxY = pointsY[i];
  }
  const T dx = maxX - minX;
  const T dy = maxY - minY;
  const T deltaMax = std::max(dx, dy);
  const T midx = (minX + maxX) / 2;
  const T midy = (minY + maxY) / 2;
  pointsX[numPoints] = midx - 20 * deltaMax;
  pointsY[numPoints] = midy - deltaMax;
  pointsX[numPoints + 1] = midx;
  pointsY[numPoints + 1] = midy + 20 * deltaMax;
  pointsX[numPoints + 2] = midx + 20 * deltaMax;
  pointsY[numPoints + 2] = midy - deltaMax;
  // Supertriangle for the initial condition
  std::fill(trianglesA.begin(), trianglesA.end(), numPoints);
  std::fill(trianglesB.begin(), trianglesB.end(), numPoints);
  std::fill(trianglesC.begin(), trianglesC.end(), numPoints);
  std::fill(trianglesIsBad.begin(), trianglesIsBad.end(), 0);
  trianglesA[0] = numPoints;
  trianglesB[0] = numPoints + 1;
  trianglesC[0] = numPoints + 2;
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
  int N = 20;
  auto DATA_DISTR = data_distribution::ALL_WAIT;
  int PERCENTAGE = 5;
  try {
    if (argc > 1) {
      N = int(atoi(argv[1]));
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
  // Create device selector for the device of your interest.
#if FPGA_EMULATOR
  // DPC++ extension: FPGA emulator selector on systems without FPGA card.
  ext::intel::fpga_emulator_selector d_selector;
#elif FPGA
  // DPC++ extension: FPGA selector on systems with FPGA card.
  ext::intel::fpga_selector d_selector;
#elif GPU
  gpu_selector d_selector;
#else
  // The default device selector will select the most performant device.
  default_selector d_selector;
#endif
  try {
    // Enable profiling.
    property_list properties{property::queue::enable_profiling()};
    queue q(d_selector, exception_handler, properties);
    // Print out the device information used for the kernel code.
    std::cout << "Running on device: " << q.get_device().get_info<info::device::name>() << "\n";
    // std::vector<dt::Point2D<float>> points(N + 3); // +3 for the supertriangle
    std::vector<float> pointsX(N * 2);
    std::vector<float> pointsY(N * 2);
    const int MAX_NUM_T = N * N + 1; // teha ctual max is probably lower...
    std::vector<int> trianglesA(MAX_NUM_T);
    std::vector<int> trianglesB(MAX_NUM_T);
    std::vector<int> trianglesC(MAX_NUM_T);
    std::vector<int8_t> trianglesIsBad(MAX_NUM_T);
    // std::vector<dt::Triangle> triangles(N*3 + 1);
    init_data(pointsX, pointsY, trianglesA, trianglesB, trianglesC, trianglesIsBad, N, DATA_DISTR, PERCENTAGE);
    auto start = std::chrono::steady_clock::now();
    double kernel_time = 0;
    kernel_time = dt_k(q, pointsX, pointsY, trianglesA, trianglesB, trianglesC, trianglesIsBad, N);
    // Wait for all work to finish.
    q.wait();
    std::cout << "\nKernel time (ms): " << kernel_time << "\n";
    int count = 0;
    for (auto &t : trianglesIsBad)
      count += int(t != 1);
    std::cout << "Num good triangles: " << count << "\n";
    auto stop = std::chrono::steady_clock::now();
    double total_time = (std::chrono::duration<double>(stop - start)).count() * 1000.0;
    // std::cout << "Total time (ms): " << total_time << "\n";
  } catch (exception const &e) {
    std::cout << "An exception was caught.\n";
    std::terminate();
  }
  return 0;
}
