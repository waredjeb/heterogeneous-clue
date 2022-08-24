#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"

#include "AlpakaCore/ScopedContext.h"
#include "AlpakaCore/Product.h"

#include "DataFormats/PointsCloud.h"
#include "DataFormats/CLUE_config.h"
#include "AlpakaDataFormats/alpaka/PointsCloudAlpaka.h"
#include "CLUEAlgoAlpaka.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class CLUEAlpakaClusterizer : public edm::EDProducer {
  public:
    explicit CLUEAlpakaClusterizer(edm::ProductRegistry& reg);
    ~CLUEAlpakaClusterizer() override = default;

  private:
    void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

    edm::EDGetTokenT<cms::alpakatools::Product<Queue, PointsCloud>> tokenPointsCloudAlpaka_;
    edm::EDPutTokenT<cms::alpakatools::Product<Queue, PointsCloudAlpaka>> tokenClusters_;
  };

  CLUEAlpakaClusterizer::CLUEAlpakaClusterizer(edm::ProductRegistry& reg)
      : tokenPointsCloudAlpaka_{reg.consumes<cms::alpakatools::Product<Queue, PointsCloud>>()},
        tokenClusters_{reg.produces<cms::alpakatools::Product<Queue, PointsCloudAlpaka>>()} {}

  void CLUEAlpakaClusterizer::produce(edm::Event& event, const edm::EventSetup& eventSetup) {
    auto const& pcProduct = event.get(tokenPointsCloudAlpaka_);
    cms::alpakatools::ScopedContextProduce<Queue> ctx(pcProduct);
    PointsCloud const& pc = ctx.get(pcProduct);
    Parameters const& par = eventSetup.get<Parameters>();
    auto stream = ctx.stream();
    CLUEAlgoAlpaka clueAlgo(par.dc, par.rhoc, par.outlierDeltaFactor, stream, pc.n);
    clueAlgo.makeClusters(pc);

    ctx.emplace(event, tokenClusters_, std::move(clueAlgo.d_points));
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(CLUEAlpakaClusterizer);