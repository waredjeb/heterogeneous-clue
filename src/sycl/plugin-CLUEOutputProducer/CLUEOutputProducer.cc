#include <iostream>
#include <memory>
#include <filesystem>
#include <fstream>

#include "DataFormats/FEDRawDataCollection.h"
#include "Framework/EDProducer.h"
#include "Framework/Event.h"
#include "Framework/EventSetup.h"
#include "Framework/PluginFactory.h"

#include "DataFormats/CLUE_config.h"
#include "DataFormats/PointsCloud.h"

#include "SYCLCore/Product.h"
#include "SYCLCore/ScopedContext.h"

#include "SYCLDataFormats/PointsCloudSYCL.h"

class CLUEOutputProducer : public edm::EDProducer {
public:
  explicit CLUEOutputProducer(edm::ProductRegistry& reg);

private:
  void produce(edm::Event& event, edm::EventSetup const& eventSetup) override;
  edm::EDGetTokenT<cms::sycltools::Product<PointsCloudSYCL>> deviceClustersToken_;
  edm::EDPutTokenT<cms::sycltools::Product<PointsCloud>> resultsToken_;
  edm::EDGetTokenT<PointsCloud> pointsCloudToken_;
};

CLUEOutputProducer::CLUEOutputProducer(edm::ProductRegistry& reg)
    : deviceClustersToken_(reg.consumes<cms::sycltools::Product<PointsCloudSYCL>>()),
      resultsToken_(reg.produces<cms::sycltools::Product<PointsCloud>>()),
      pointsCloudToken_(reg.consumes<PointsCloud>()) {}

void CLUEOutputProducer::produce(edm::Event& event, edm::EventSetup const& eventSetup) {
  auto outDir = eventSetup.get<std::filesystem::path>();
  PointsCloud results = event.get(pointsCloudToken_);
  auto const& pcProduct = event.get(deviceClustersToken_);
  cms::sycltools::ScopedContextProduce ctx{pcProduct};
  auto const& device_clusters = ctx.get(pcProduct);
  auto stream = ctx.stream();

  results.outResize(device_clusters.n);
  stream.memcpy(results.rho.data(), device_clusters.rho.get(), device_clusters.n * sizeof(float));
  stream.memcpy(results.delta.data(), device_clusters.delta.get(), device_clusters.n * sizeof(float));
  stream.memcpy(results.nearestHigher.data(), device_clusters.nearestHigher.get(), device_clusters.n * sizeof(int));
  stream.memcpy(results.isSeed.data(), device_clusters.isSeed.get(), device_clusters.n * sizeof(int));
  stream.memcpy(results.clusterIndex.data(), device_clusters.clusterIndex.get(), device_clusters.n * sizeof(int)).wait();

  std::cout << "Data transferred back to host" << std::endl;

  Parameters par;
  par = eventSetup.get<Parameters>();
  if (par.produceOutput) {
    auto temp_outDir = eventSetup.get<std::filesystem::path>();
    std::string input_file_name = temp_outDir.filename();
    std::string output_file_name = create_outputfileName(input_file_name, par.dc, par.rhoc, par.outlierDeltaFactor);
    std::filesystem::path outDir = temp_outDir.parent_path() / output_file_name;

    std::ofstream clueOut(outDir);

    clueOut << "index,x,y,layer,weight,rho,delta,nh,isSeed,clusterId\n";
    for (int i = 0; i < device_clusters.n; i++) {
      clueOut << i << "," << results.x[i] << "," << results.y[i] << "," << results.layer[i] << "," << results.weight[i]
              << "," << results.rho[i] << "," << (results.delta[i] > 999 ? 999 : results.delta[i]) << ","
              << results.nearestHigher[i] << "," << results.isSeed[i] << "," << results.clusterIndex[i] << "\n";
    }

    clueOut.close();

    std::cout << "Ouput was saved in " << outDir << std::endl;
  }

  ctx.emplace(event, resultsToken_, std::move(results));
}
DEFINE_FWK_MODULE(CLUEOutputProducer);
