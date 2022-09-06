#include <iostream>

#include "DataFormats/PointsCloud.h"
#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/EventSetup.h"
#include "Framework/PluginFactory.h"

#include "AlpakaCore/alpakaConfig.h"
#include "AlpakaCore/ScopedContext.h"
#include "AlpakaCore/Product.h"

#include "alpakaAlgo1.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class TestProducer : public edm::EDProducer {
  public:
    explicit TestProducer(edm::ProductRegistry& reg);

  private:
    void produce(edm::Event& event, edm::EventSetup const& eventSetup) override;

    edm::EDGetTokenT<PointsCloud> rawGetToken_;
    edm::EDPutTokenT<cms::alpakatools::Product<Queue, cms::alpakatools::device_buffer<Device, float[]>>> putToken_;
  };

  TestProducer::TestProducer(edm::ProductRegistry& reg)
      : rawGetToken_(reg.consumes<PointsCloud>()),
        putToken_(reg.produces<cms::alpakatools::Product<Queue, cms::alpakatools::device_buffer<Device, float[]>>>()) {}

  void TestProducer::produce(edm::Event& event, edm::EventSetup const& eventSetup) {
    auto const value = event.get(rawGetToken_);
    std::cout << "Number of points: " << value.n << '\n';
    std::cout << "TestProducer  Event " << event.eventID() << " stream " << event.streamID() << " ES int "
              << eventSetup.get<int>() << std::endl;
              
    cms::alpakatools::ScopedContextProduce<Queue> ctx(event.streamID());
    ctx.emplace(event, putToken_, alpakaAlgo1(ctx.stream()));
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(TestProducer);
