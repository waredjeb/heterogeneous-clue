#include <iostream>

#include "DataFormats/FEDRawDataCollection.h"
#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/EventSetup.h"
#include "Framework/PluginFactory.h"

#include "AlpakaCore/Product.h"
#include "AlpakaCore/ScopedContext.h"
#include "AlpakaCore/alpakaMemory.h"

#include "DataFormats/PointsCloud.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class PointsCloudProducer : public edm::EDProducer {
  public:
    explicit PointsCloudProducer(edm::ProductRegistry& reg);
    // ~PointsCloudProducer() override = default;

  private:
    void produce(edm::Event& event, edm::EventSetup const& eventSetup) override;

    edm::EDPutTokenT<cms::alpakatools::Product<Queue, PointsCloud>> pcPutToken_;
  };

  PointsCloudProducer::PointsCloudProducer(edm::ProductRegistry& reg)
      : pcPutToken_(reg.produces<cms::alpakatools::Product<Queue, PointsCloud>>()) {}

  void PointsCloudProducer::produce(edm::Event& event, edm::EventSetup const& eventSetup) {
    cms::alpakatools::ScopedContextProduce<Queue> ctx(event.streamID());
    auto pcHost = eventSetup.get<PointsCloud>();
    ctx.emplace(event, pcPutToken_, std::move(pcHost));
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(PointsCloudProducer);