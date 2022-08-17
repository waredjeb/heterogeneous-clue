#include <iostream>

#include "DataFormats/FEDRawDataCollection.h"
#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/EventSetup.h"
#include "Framework/PluginFactory.h"

#include "AlpakaCore/Product.h"
#include "AlpakaCore/ScopedContext.h"
#include "AlpakaCore/alpakaMemory.h"

#include "AlpakaDataFormats/PointsCloudAlpaka.h"
#include "../ValidatorData.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class ValidatorPointsCloudToAlpaka : public edm::EDProducer {
  public:
    explicit ValidatorPointsCloudToAlpaka(edm::ProductRegistry& reg);
    // ~ValidatorPointsCloudToAlpaka() = default;


  private:
    void produce(edm::Event& event, edm::EventSetup const& eventSetup) override;
    template <class T>
    bool arraysAreEqual(cms::alpakatools::device_buffer<Device, T[]> deviceArr,
                        std::vector<T> trueDataArr,
                        int n,
                        Queue stream);
    edm::EDGetTokenT<cms::alpakatools::Product<Queue, PointsCloudAlpaka>> tokenPC_;
  };

  ValidatorPointsCloudToAlpaka::ValidatorPointsCloudToAlpaka(edm::ProductRegistry& reg)
      : tokenPC_(reg.consumes<cms::alpakatools::Product<Queue, PointsCloudAlpaka>>()) {}

  template <class T>
  bool ValidatorPointsCloudToAlpaka::arraysAreEqual(cms::alpakatools::device_buffer<Device, T[]> deviceArr,
                                                    std::vector<T> trueDataArr,
                                                    int n,
                                                    Queue stream) {
    bool sameValue = true;

    auto host = cms::alpakatools::make_host_buffer<T[]>(n);
    alpaka::memcpy(stream, host, deviceArr);

    for (int i = 0; i < n; i++) {
      if (host[i] != trueDataArr[i]) {
        sameValue = false;
        break;
      }
    }

    // delete[] host;

    return sameValue;
  }

  void ValidatorPointsCloudToAlpaka::produce(edm::Event& event, edm::EventSetup const& eventSetup) {
    auto pcTrueData = std::make_unique<ValidatorPointsCloud>();
    *pcTrueData = eventSetup.get<ValidatorPointsCloud>();

    auto const& pcDeviceProduct = event.get(tokenPC_);
    cms::alpakatools::ScopedContextProduce ctx{pcDeviceProduct};
    auto const& pcDevice = ctx.get(pcDeviceProduct);

    std::cout << "Checking input data on device" << '\n';

    assert(pcDevice.n == pcTrueData->n);
    std::cout << "Number of points -> Ok" << '\n';
    assert(arraysAreEqual(pcDevice.x, pcTrueData->x, pcDevice.n, ctx.stream()));
    std::cout << "x -> Ok" << '\n';
    assert(arraysAreEqual(pcDevice.y, pcTrueData->y, pcDevice.n, ctx.stream()));
    std::cout << "y -> Ok" << '\n';
    assert(arraysAreEqual(pcDevice.layer, pcTrueData->layer, pcDevice.n, ctx.stream()));
    std::cout << "layer -> Ok" << '\n';
    assert(arraysAreEqual(pcDevice.weight, pcTrueData->weight, pcDevice.n, ctx.stream()));
    std::cout << "weight -> Ok" << '\n';

    std::cout << "Input data copied correctly from host to device" << std::endl;
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(ValidatorPointsCloudToAlpaka);