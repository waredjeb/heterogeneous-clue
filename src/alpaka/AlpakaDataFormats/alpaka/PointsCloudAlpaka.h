#ifndef Points_Cloud_Alpaka_h
#define Points_Cloud_Alpaka_h

#include <memory>

#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/alpakaMemory.h"
#include "DataFormats/PointsCloud.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  constexpr unsigned int reserve = 1000000;

  class PointsCloudAlpaka {
  public:
    PointsCloudAlpaka() = delete;
    explicit PointsCloudAlpaka(Queue &stream, uint32_t numberOfPoints)
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
          n{numberOfPoints},
          view_d{cms::alpakatools::make_device_buffer<PointsCloudAlpakaView>(stream)} {
      auto view_h = cms::alpakatools::make_host_buffer<PointsCloudAlpakaView>(stream);
      view_h->x = x.data();
      view_h->y = y.data();
      view_h->layer = layer.data();
      view_h->weight = weight.data();
      view_h->rho = rho.data();
      view_h->delta = delta.data();
      view_h->nearestHigher = nearestHigher.data();
      view_h->clusterIndex = clusterIndex.data();
      view_h->isSeed = isSeed.data();
      view_h->n = numberOfPoints;

      alpaka::memcpy(stream, view_d, view_h);
      alpaka::wait(stream);
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
    uint32_t n;

    class PointsCloudAlpakaView {
    public:
      float *x;
      float *y;
      int *layer;
      float *weight;
      float *rho;
      float *delta;
      int *nearestHigher;
      int *clusterIndex;
      int *isSeed;
      uint32_t n;
    };

    PointsCloudAlpakaView *view() { return view_d.data(); }

  private:
    cms::alpakatools::device_buffer<Device, PointsCloudAlpakaView> view_d;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif