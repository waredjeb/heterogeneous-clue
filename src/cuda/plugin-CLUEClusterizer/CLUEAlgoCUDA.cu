#include <math.h>
#include <limits>
#include <iostream>

// GPU Add
#include <cuda_runtime.h>
#include <cuda.h>
// for timing
#include <chrono>
#include <ctime>
// user include

#include "CLUEAlgoCUDA.h"
#include "CLUEAlgoKernels.h"

void CLUEAlgoCUDA::init_device() {
  d_hist = cms::cuda::make_device_unique<LayerTilesCUDA[]>(NLAYERS, stream_);
  d_seeds = cms::cuda::make_device_unique<cms::cuda::VecArray<int, maxNSeeds>>(
      stream_);
  d_followers =
      cms::cuda::make_device_unique<cms::cuda::VecArray<int, maxNFollowers> []>(
          reserve, stream_);

  hist_ = d_hist.get();
  seeds_ = d_seeds.get();
  followers_ = d_followers.get();
}

void CLUEAlgoCUDA::setup(PointsCloud const& host_pc) {
  // copy input variables
  cudaMemcpy(d_points.x.get(), host_pc.x.data(), sizeof(float) * host_pc.n,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_points.y.get(), host_pc.y.data(), sizeof(float) * host_pc.n,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_points.layer.get(), host_pc.layer.data(),
             sizeof(int) * host_pc.n, cudaMemcpyHostToDevice);
  cudaMemcpy(d_points.weight.get(), host_pc.weight.data(),
             sizeof(float) * host_pc.n, cudaMemcpyHostToDevice);
  // initialize result and internal variables
  // // result variables
  cudaMemset(d_points.rho.get(), 0x00, sizeof(float) * host_pc.n);
  cudaMemset(d_points.delta.get(), 0x00, sizeof(float) * host_pc.n);
  cudaMemset(d_points.nearestHigher.get(), 0x00, sizeof(int) * host_pc.n);
  cudaMemset(d_points.clusterIndex.get(), 0x00, sizeof(int) * host_pc.n);
  cudaMemset(d_points.isSeed.get(), 0x00, sizeof(int) * host_pc.n);
  // algorithm internal variables
  cudaMemset(d_hist.get(), 0x00, sizeof(LayerTilesCUDA) * NLAYERS);
  cudaMemset(d_seeds.get(), 0x00, sizeof(cms::cuda::VecArray<int, maxNSeeds>));
  cudaMemset(d_followers.get(), 0x00,
             sizeof(cms::cuda::VecArray<int, maxNFollowers>) * host_pc.n);

  cudaStreamSynchronize(stream_);
}

void CLUEAlgoCUDA::makeClusters(PointsCloud const& host_pc) {

  setup(host_pc);
  ////////////////////////////////////////////
  // calculate rho, delta and find seeds
  // 1 point per thread
  ////////////////////////////////////////////
  const dim3 blockSize(1024, 1, 1);
  const dim3 gridSize(ceil(host_pc.n / static_cast<float>(blockSize.x)), 1, 1);
  kernel_compute_histogram << <gridSize, blockSize>>>
      (d_hist.get(), d_points.view(), host_pc.n);
  kernel_calculate_density << <gridSize, blockSize>>>
      (d_hist.get(), d_points.view(), dc_, host_pc.n);
  kernel_calculate_distanceToHigher << <gridSize, blockSize>>>
      (d_hist.get(), d_points.view(), outlierDeltaFactor_, dc_, host_pc.n);
  kernel_find_clusters << <gridSize, blockSize>>>
      (d_seeds.get(), d_followers.get(), d_points.view(), outlierDeltaFactor_,
       dc_, rhoc_, host_pc.n);

  ////////////////////////////////////////////
  // assign clusters
  // 1 point per seeds
  ////////////////////////////////////////////
  const dim3 gridSize_nseeds(ceil(maxNSeeds / 1024.0), 1, 1);
  kernel_assign_clusters << <gridSize_nseeds, blockSize>>>
      (d_seeds.get(), d_followers.get(), d_points.view(), host_pc.n);
}
