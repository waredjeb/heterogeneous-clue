#ifndef CLUEAlgo_CUDA_h
#define CLUEAlgo_CUDA_h

// #include <optional>

#include "CUDACore/alpakaMemory.h"
#include "CUDADataFormats/alpaka/PointsCloudCUDA.h"
#include "CUDADataFormats/LayerTilesCUDA.h"


class CLUEAlgoCUDA {
public:
  // constructor
  CLUEAlgoCUDA() = delete;
  explicit CLUEAlgoCUDA(float const &dc,
                          float const &rhoc,
                          float const &outlierDeltaFactor,
                          cudaStream_t stream,
                          uint32_t const &numberOfPoints)
      : d_points{stream, numberOfPoints},
        dc_{dc},
        rhoc_{rhoc},
      outlierDeltaFactor_{outlierDeltaFactor} {
    :
    d_hist = cms::cuda::make_device_unique<LayerTilesCUDA[]>(numberOfPoints, stream);
    dc_ = cms::cuda::make_device_unique<cms::cuda::VecArray<int, maxNSeeds>>(dc, stream);
    rhoc_ = cms::cuda::make_device_unique<cms::cuda::VecArray<int, maxNFollowers>>(rhoc, stream);
    init_device();
  }

  ~CLUEAlgoCUDA() = default;

  void makeClusters(PointsCloud const &host_pc);

  PointsCloudCUDA d_points;

  LayerTilesCUDA<Acc1D> *hist_;
  cms::cuda::VecArray<int, maxNSeeds> *seeds_;
  cms::cuda::VecArray<int, maxNFollowers> *followers_;

private:
  float dc_;
  float rhoc_;
  float outlierDeltaFactor_;

  cms::cuda::device_unique_pointer<LayerTilesCUDA[]> d_hist;
  cms::cuda::device_unique_pointer<cms::cuda::VecArray<int, maxNSeeds> d_seeds;
  cms::cuda::device_unique_pointer<cms::cuda::VecArray<int, maxNFollowers>[]> d_followers;

  // private methods
  void init_device();

  void setup(PointsCloud const &host_pc);
};
