#ifndef CLUEAlgoSYCL_h
#define CLUEAlgoSYCL_h

#include <optional>

#include <CL/sycl.hpp>

#include "SYCLDataFormats/PointsCloudSYCL.h"
#include "SYCLDataFormats/LayerTilesSYCL.h"

class CLUEAlgoSYCL {
public:
  CLUEAlgoSYCL(float const &dc,
               float const &rhoc,
               float const &outlierDeltaFactor,
               sycl::queue const &stream,
               int const &numberOfPoints);

  ~CLUEAlgoSYCL();

  void makeClusters(PointsCloud const &host_pc);

  PointsCloudSYCL d_points;

private:
  float dc_;
  float rhoc_;
  float outlierDeltaFactor_;

  LayerTilesSYCL *d_hist;
  cms::sycltools::VecArray<int, maxNSeeds> *d_seeds;
  cms::sycltools::VecArray<int, maxNFollowers> *d_followers;

  std::optional<sycl::queue> queue_;

  void init_device();

  void free_device();

  void setup(PointsCloud const &host_pc);
};

#endif