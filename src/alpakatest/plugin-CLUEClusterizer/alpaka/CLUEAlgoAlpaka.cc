#include "DataFormats/PointsCloud.h"

#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/alpakaWorkDiv.h"
#include "CLUEAlgoAlpaka.h"
#include "CLUEAlgoKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  void CLUEAlgoAlpaka::init_device() {
    d_points.view_d->x = alpaka::getPtrNative(d_points.x);
    d_points.view_d->y = alpaka::getPtrNative(d_points.y);
    d_points.view_d->layer = alpaka::getPtrNative(d_points.layer);
    d_points.view_d->weight = alpaka::getPtrNative(d_points.weight);
    d_points.view_d->rho = alpaka::getPtrNative(d_points.rho);
    d_points.view_d->delta = alpaka::getPtrNative(d_points.delta);
    d_points.view_d->nearestHigher = alpaka::getPtrNative(d_points.nearestHigher);
    d_points.view_d->clusterIndex = alpaka::getPtrNative(d_points.clusterIndex);
    d_points.view_d->isSeed = alpaka::getPtrNative(d_points.isSeed);

    hist_ = alpaka::getPtrNative(d_hist);
    seeds_ = alpaka::getPtrNative(d_seeds);
    followers_ = alpaka::getPtrNative(d_followers);
  }

  void CLUEAlgoAlpaka::setup(PointsCloud const &host_pc) {
    // copy input variables
    alpaka::memcpy(queue_, d_points.x, cms::alpakatools::make_host_view(host_pc.x.data(), host_pc.n));
    alpaka::memcpy(queue_, d_points.y, cms::alpakatools::make_host_view(host_pc.y.data(), host_pc.n));
    alpaka::memcpy(queue_, d_points.layer, cms::alpakatools::make_host_view(host_pc.layer.data(), host_pc.n));
    alpaka::memcpy(queue_, d_points.weight, cms::alpakatools::make_host_view(host_pc.weight.data(), host_pc.n));
    // initialize result and internal variables
    alpaka::memset(queue_, d_points.rho, 0x00, d_points.n);
    alpaka::memset(queue_, d_points.delta, 0x00, d_points.n);
    alpaka::memset(queue_, d_points.nearestHigher, 0x00, d_points.n);
    alpaka::memset(queue_, d_points.clusterIndex, 0x00, d_points.n);
    alpaka::memset(queue_, d_points.isSeed, 0x00, d_points.n);
    alpaka::memset(queue_, d_hist, 0x00, static_cast<uint32_t>(NLAYERS));
    alpaka::memset(queue_, d_seeds, 0x00, static_cast<uint32_t>(1));
    alpaka::memset(queue_, d_followers, 0x00, d_points.n);

    alpaka::wait(queue_);
  }

  void CLUEAlgoAlpaka::makeClusters(PointsCloud const &host_pc) {
    setup(host_pc);
    // calculate rho, delta and find seeds
    // 1 point per thread
    const Idx blockSize = 1024;
    const Idx gridSize = ceil(d_points.n / static_cast<float>(blockSize));
    auto WorkDiv1D = cms::alpakatools::make_workdiv<Acc1D>(gridSize, blockSize);
    alpaka::enqueue(
        queue_,
        alpaka::createTaskKernel<Acc1D>(WorkDiv1D, KernelComputeHistogram(), hist_, d_points.view_d, d_points.n));
    // alpaka::enqueue(
    //     queue_,
    //     alpaka::createTaskKernel<Acc1D>(WorkDiv1D, KernelCalculateDensity(), hist_, d_points.view_d, dc_, d_points.n));
    // alpaka::enqueue(
    //     queue_,
    //     alpaka::createTaskKernel<Acc1D>(
    //         WorkDiv1D, KernelComputeDistanceToHigher(), hist_, d_points.view_d, outlierDeltaFactor_, dc_, d_points.n));
    // alpaka::enqueue(queue_,
    //                 alpaka::createTaskKernel<Acc1D>(WorkDiv1D,
    //                                                 KernelFindClusters(),
    //                                                 seeds_,
    //                                                 followers_,
    //                                                 d_points.view_d,
    //                                                 outlierDeltaFactor_,
    //                                                 dc_,
    //                                                 rhoc_,
    //                                                 d_points.n));
    // alpaka::enqueue(queue_,
    //                 alpaka::createTaskKernel<Acc1D>(
    //                     WorkDiv1D, KernelAssignClusters(), seeds_, followers_, d_points.view_d, d_points.n));

    alpaka::wait(queue_);
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE