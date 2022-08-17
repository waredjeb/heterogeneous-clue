#ifndef POINTS_CLOUD_ALPAKA_H
#define POINTS_CLOUD_ALPAKA_H

#include <memory>

#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/alpakaMemory.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  constexpr unsigned int reserve = 1000000;

  class PointsCloudAlpaka {
  public:
    PointsCloudAlpaka() = delete;
    explicit PointsCloudAlpaka(Queue &stream, int numberOfPoints)
        //input variables
        : x{cms::alpakatools::make_device_buffer<float[]>(stream, reserve)},
          y{cms::alpakatools::make_device_buffer<float[]>(stream, reserve)},
          layer{cms::alpakatools::make_device_buffer<int[]>(stream, reserve)},
          weight{cms::alpakatools::make_device_buffer<float[]>(stream, reserve)},
          //result variables
          rho{cms::alpakatools::make_device_buffer<float[]>(stream, reserve)},
          delta{cms::alpakatools::make_device_buffer<float[]>(stream, reserve)},
          nearestHigher{cms::alpakatools::make_device_buffer<int[]>(stream, reserve)},
          clusterIndex{cms::alpakatools::make_device_buffer<int[]>(stream, reserve)},
          isSeed{cms::alpakatools::make_device_buffer<int[]>(stream, reserve)},
          n{numberOfPoints} {
      // auto view = std::make_unique<PointsCloudAlpakaView>();
      // view->x = x.get();
      // view->y = y.get();
      // view->layer = layer.get();
      // view->weight = weight.get();
      // view->rho = rho.get();
      // view->delta = delta.get();
      // view->nearestHigher = nearestHigher.get();
      // view->clusterIndex = clusterIndex.get();
      // view->isSeed = isSeed.get();
      // view->n = numberOfPoints;

      // view_d = cms::sycltools::make_device_unique<PointsCloudSYCLView>(stream);
      // stream.memcpy(view_d.get(), view.get(), sizeof(PointsCloudSYCLView)).wait();

      //// either make host and device view to each buffer or use Marco's strategy
    }
    PointsCloudAlpaka(PointsCloudAlpaka const &) = delete;
    PointsCloudAlpaka(PointsCloudAlpaka &&) = default;
    PointsCloudAlpaka &operator=(PointsCloudAlpaka const &) = delete;
    PointsCloudAlpaka &operator=(PointsCloudAlpaka &&) = default;

    ~PointsCloudAlpaka() = default;

    cms::alpakatools::device_buffer<Device, float[]> x;
    cms::alpakatools::device_buffer<Device, float[]> y;
    cms::alpakatools::device_buffer<Device, int[]> layer;
    cms::alpakatools::device_buffer<Device, float[]> weight;
    cms::alpakatools::device_buffer<Device, float[]> rho;
    cms::alpakatools::device_buffer<Device, float[]> delta;
    cms::alpakatools::device_buffer<Device, int[]> nearestHigher;
    cms::alpakatools::device_buffer<Device, int[]> clusterIndex;
    cms::alpakatools::device_buffer<Device, int[]> isSeed;
    int n;

    // class PointsCloudAlpakaView {
    // public:
    //   float *x;
    //   float *y;
    //   int *layer;
    //   float *weight;
    //   float *rho;
    //   float *delta;
    //   int *nearestHigher;
    //   int *clusterIndex;
    //   int *isSeed;
    //   int n;
    // };

    // PointsCloudAlpakaView *view() const { return view_d.get(); }

  // private:
    // cms::sycltools::device::unique_ptr<PointsCloudSYCLView> view_d;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE
#endif