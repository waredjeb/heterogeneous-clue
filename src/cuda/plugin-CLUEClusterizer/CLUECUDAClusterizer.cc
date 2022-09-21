#include "Framework/EventSetup.h"
#include "Framework/Event.h"
#include "Framework/PluginFactory.h"
#include "Framework/EDProducer.h"

#include "CUDACore/ScopedContext.h"
#include "CUDACore/Product.h"

#include "DataFormats/PointsCloud.h"
#include "DataFormats/CLUE_config.h"
#include "CUDADataFormats/PointsCloudCUDA.h"
#include "CLUEAlgoCUDA.h"

class CLUECUDAClusterizer : public edm::EDProducer {
public:
  explicit CLUECUDAClusterizer(edm::ProductRegistry& reg);
  ~CLUECUDAClusterizer() override = default;

private:
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  edm::EDGetTokenT<PointsCloud> pointsCloudToken_;
  edm::EDPutTokenT<cms::cuda::Product<PointsCloudCUDA>> clusterToken_;
};

CLUECUDAClusterizer::CLUECUDAClusterizer(edm::ProductRegistry& reg)
    : pointsCloudToken_{reg.consumes<PointsCloud>()},
      clusterToken_{reg.produces<cms::cuda::Product<PointsCloudCUDA>>()} {}

void CLUECUDAClusterizer::produce(edm::Event& event, const edm::EventSetup& eventSetup) {
  auto const& pc = event.get(pointsCloudToken_);
  cms::cuda::ScopedContextProduce ctx(event.streamID());
  Parameters const& par = eventSetup.get<Parameters>();
  auto stream = ctx.stream();
  CLUEAlgoCUDA clueAlgo(par.dc, par.rhoc, par.outlierDeltaFactor, stream, pc.n);
  clueAlgo.makeClusters(pc);

  ctx.emplace(event, clusterToken_, std::move(clueAlgo.d_points));
}

DEFINE_FWK_MODULE(CLUECUDAClusterizer);
