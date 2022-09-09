#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"

#include "DataFormats/PointsCloud.h"
#include "DataFormats/CLUE_config.h"
#include "CLUEAlgoSerial.h"

class CLUESerialClusterizer : public edm::EDProducer {
public:
  explicit CLUESerialClusterizer(edm::ProductRegistry& reg);
  ~CLUESerialClusterizer() override = default;

private:
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  edm::EDGetTokenT<PointsCloud> pointsCloudToken_;
  edm::EDPutTokenT<PointsCloudSerial> clusterToken_;
};

CLUESerialClusterizer::CLUESerialClusterizer(edm::ProductRegistry& reg)
    : pointsCloudToken_{reg.consumes<PointsCloud>()}, clusterToken_{reg.produces<PointsCloudSerial>()} {}

void CLUESerialClusterizer::produce(edm::Event& event, const edm::EventSetup& eventSetup) {
  auto const& pc = event.get(pointsCloudToken_);
  Parameters const& par = eventSetup.get<Parameters>();
  CLUEAlgoSerial clueAlgo(par.dc, par.rhoc, par.outlierDeltaFactor, pc.n);
  clueAlgo.makeClusters(pc);

  event.emplace(clusterToken_, std::move(clueAlgo.d_points));
}

DEFINE_FWK_MODULE(CLUESerialClusterizer);