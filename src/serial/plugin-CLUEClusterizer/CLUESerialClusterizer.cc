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
  // edm::EDPutTokenT<PointsCloud> resultsToken_;
};

CLUESerialClusterizer::CLUESerialClusterizer(edm::ProductRegistry& reg)
    : pointsCloudToken_(reg.consumes<PointsCloud>()) {} //, resultsToken_(reg.produces<PointsCloud>()) {}

void CLUESerialClusterizer::produce(edm::Event& event, const edm::EventSetup& eventSetup) {
  // auto const& pc = event.get(pointsCloudToken_);
  auto pc = event.get(pointsCloudToken_);
  Parameters const& par = eventSetup.get<Parameters>();
  CLUEAlgoSerial clueAlgo(par.dc, par.rhoc, par.outlierDeltaFactor, pc.n);  //removed stream argument
  clueAlgo.makeClusters(pc);

  std::cout << "[A] Trying to print a point.x : " << pc.x[10] << std::endl;
  std::cout << "[A] Trying to print a point.y : " << pc.y[10] << std::endl;
  std::cout << "[A] Trying to print a point.rho : " << pc.rho[10] << std::endl;
  // ctx.emplace(event, clusterToken_, std::move(clueAlgo.d_points));
  // event.emplace(resultsToken_, pc);  // need to check if instead I should put clueAlgo data member -> depends on what makeClusters returns
}

DEFINE_FWK_MODULE(CLUESerialClusterizer);