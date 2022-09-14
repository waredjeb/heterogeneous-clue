#include <CL/sycl.hpp>

#include "DataFormats/PointsCloud.h"

#include "CLUEAlgoSYCL.h"
#include "CLUEAlgoKernels.h"

CLUEAlgoSYCL::CLUEAlgoSYCL(float const &dc,
                           float const &rhoc,
                           float const &outlierDeltaFactor,
                           sycl::queue const &stream,
                           int const &numberOfPoints)
    : d_points{stream, numberOfPoints}, dc_{dc}, rhoc_{rhoc}, outlierDeltaFactor_{outlierDeltaFactor}, queue_{stream} {
  init_device();
}

CLUEAlgoSYCL::~CLUEAlgoSYCL() {
  if (queue_) {
    free_device();
  }
}

void CLUEAlgoSYCL::init_device() {
  d_hist = sycl::malloc_device<LayerTilesSYCL>(NLAYERS, *queue_);
  d_seeds = sycl::malloc_device<cms::sycltools::VecArray<int, maxNSeeds>>(1, *queue_);
  d_followers = sycl::malloc_device<cms::sycltools::VecArray<int, maxNFollowers>>(reserve, *queue_);
}

void CLUEAlgoSYCL::free_device() {
  sycl::free(d_hist, *queue_);
  sycl::free(d_seeds, *queue_);
  sycl::free(d_followers, *queue_);
}

void CLUEAlgoSYCL::setup(PointsCloud const &host_pc) {
  // input variables
  (*queue_).memcpy(d_points.x.get(), host_pc.x.data(), sizeof(float) * host_pc.n);
  (*queue_).memcpy(d_points.y.get(), host_pc.y.data(), sizeof(float) * host_pc.n);
  (*queue_).memcpy(d_points.layer.get(), host_pc.layer.data(), sizeof(int) * host_pc.n);
  (*queue_).memcpy(d_points.weight.get(), host_pc.weight.data(), sizeof(float) * host_pc.n);
  // result and internal variables
  (*queue_).memset(d_points.rho.get(), 0x00, sizeof(float) * host_pc.n);
  (*queue_).memset(d_points.delta.get(), 0x00, sizeof(float) * host_pc.n);
  (*queue_).memset(d_points.nearestHigher.get(), 0x00, sizeof(int) * host_pc.n);
  (*queue_).memset(d_points.clusterIndex.get(), 0x00, sizeof(int) * host_pc.n);
  (*queue_).memset(d_points.isSeed.get(), 0x00, sizeof(int) * host_pc.n);
  (*queue_).memset(d_hist, 0x00, sizeof(LayerTilesSYCL) * NLAYERS);
  (*queue_).memset(d_seeds, 0x00, sizeof(cms::sycltools::VecArray<int, maxNSeeds>));
  (*queue_).memset(d_followers, 0x00, sizeof(cms::sycltools::VecArray<int, maxNFollowers>) * host_pc.n).wait();
}

void CLUEAlgoSYCL::makeClusters(PointsCloud const &host_pc) {
  setup(host_pc);
  const int numThreadsPerBlock = 256;  // ThreadsPerBlock = work-group size
  const sycl::range<1> blockSize(numThreadsPerBlock);
  const sycl::range<1> gridSize(ceil(d_points.n / static_cast<float>(blockSize[0])));
  PointsCloudSYCL::PointsCloudSYCLView *d_points_view = d_points.view();

  (*queue_).submit([&](sycl::handler &cgh) {
    //SYCL kernels cannot capture by reference - need to reassign pointers inside the submit to pass by value
    auto d_hist_kernel = d_hist;
    auto num_points_kernel = d_points.n;
    cgh.parallel_for(sycl::nd_range<1>(gridSize * blockSize, blockSize), [=](sycl::nd_item<1> item) {
      kernel_compute_histogram(d_hist_kernel, d_points_view, num_points_kernel, item);
    });
  });

  (*queue_).submit([&](sycl::handler &cgh) {
    auto d_hist_kernel = d_hist;
    auto dc_kernel = dc_;
    auto num_points_kernel = d_points.n;
    cgh.parallel_for(sycl::nd_range<1>(gridSize * blockSize, blockSize), [=](sycl::nd_item<1> item) {
      kernel_calculate_density(d_hist_kernel, d_points_view, dc_kernel, num_points_kernel, item);
    });
  });

  (*queue_).submit([&](sycl::handler &cgh) {
    auto d_hist_kernel = d_hist;
    auto outlierDeltaFactor_kernel = outlierDeltaFactor_;
    auto dc_kernel = dc_;
    auto num_points_kernel = d_points.n;
    cgh.parallel_for(sycl::nd_range<1>(gridSize * blockSize, blockSize), [=](sycl::nd_item<1> item) {
      kernel_calculate_distanceToHigher(
          d_hist_kernel, d_points_view, outlierDeltaFactor_kernel, dc_kernel, num_points_kernel, item);
    });
  });

  (*queue_).submit([&](sycl::handler &cgh) {
    auto d_seeds_kernel = d_seeds;
    auto d_followers_kernel = d_followers;
    auto outlierDeltaFactor_kernel = outlierDeltaFactor_;
    auto dc_kernel = dc_;
    auto rhoc_kernel = rhoc_;
    auto num_points_kernel = d_points.n;
    cgh.parallel_for(sycl::nd_range<1>(gridSize * blockSize, blockSize), [=](sycl::nd_item<1> item) {
      kernel_find_clusters(d_seeds_kernel,
                           d_followers_kernel,
                           d_points_view,
                           outlierDeltaFactor_kernel,
                           dc_kernel,
                           rhoc_kernel,
                           num_points_kernel,
                           item);
    });
  });

  const sycl::range<1> gridSize_nseeds(ceil(maxNSeeds / static_cast<double>(blockSize[0])));
  (*queue_).submit([&](sycl::handler &cgh) {
    auto d_seeds_kernel = d_seeds;
    auto d_followers_kernel = d_followers;
    cgh.parallel_for(sycl::nd_range<1>(gridSize_nseeds * blockSize, blockSize), [=](sycl::nd_item<1> item) {
      kernel_assign_clusters(d_seeds_kernel, d_followers_kernel, d_points_view, item);
    });
  });
  (*queue_).wait();
}