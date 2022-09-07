#include <iostream>
#include "DataFormats/PointsCloud.h"

#include "CLUEAlgoSerial.h"
#include "CLUEAlgoKernels.h"

void CLUEAlgoSerial::setup(PointsCloud &host_pc) {
  // resize output variables
  host_pc.outResize(host_pc.n);
}

void CLUEAlgoSerial::makeClusters(PointsCloud &host_pc) {
  setup(host_pc);
  // calculate rho, delta and find seeds
  std::cout << "[B] Trying to print a point.x : " << host_pc.x[10] << std::endl;
  std::cout << "[B] Trying to print a point.y : " << host_pc.y[10] << std::endl;
  std::cout << "[B] Trying to print a point.rho : " << host_pc.rho[10] << std::endl;

  KernelComputeHistogram(hist_, host_pc);
  KernelCalculateDensity(hist_, host_pc, dc_);
  KernelComputeDistanceToHigher(hist_, host_pc, outlierDeltaFactor_, dc_);
  KernelFindAndAssignClusters(host_pc, outlierDeltaFactor_, dc_, rhoc_);
}