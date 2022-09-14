#ifndef CLUEAlgo_Serial_h
#define CLUEAlgo_Serial_h

#include "DataFormats/PointsCloud.h"
#include "DataFormats/LayerTilesSerial.h"

class CLUEAlgoSerial {
public:
  // constructor
  CLUEAlgoSerial() = delete;
  explicit CLUEAlgoSerial(float const &dc,
                          float const &rhoc,
                          float const &outlierDeltaFactor,
                          uint32_t const &numberOfPoints)
      : dc_{dc}, rhoc_{rhoc}, outlierDeltaFactor_{outlierDeltaFactor} {}

  ~CLUEAlgoSerial() = default;

  void makeClusters(PointsCloud const &host_pc);

  PointsCloudSerial d_points;

  std::array<LayerTilesSerial, NLAYERS> hist_;

private:
  float dc_;
  float rhoc_;
  float outlierDeltaFactor_;

  void setup(PointsCloud const &host_pc);
};

#endif