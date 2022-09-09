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

class CLUEOutputProducer : public edm::EDProducer {
public:
  explicit CLUEOutputProducer(edm::ProductRegistry& reg);

private:
  void produce(edm::Event& event, edm::EventSetup const& eventSetup) override;
  edm::EDGetTokenT<PointsCloudSerial> clustersToken_;
};

CLUEOutputProducer::CLUEOutputProducer(edm::ProductRegistry& reg) : clustersToken_(reg.consumes<PointsCloudSerial>()) {}

void CLUEOutputProducer::produce(edm::Event& event, edm::EventSetup const& eventSetup) {
  auto outDir = eventSetup.get<std::filesystem::path>();
  auto const& results = event.get(clustersToken_);

  Parameters par;
  par = eventSetup.get<Parameters>();
  if (par.produceOutput) {
    auto temp_outDir = eventSetup.get<std::filesystem::path>();
    std::string input_file_name = temp_outDir.filename();
    std::string output_file_name = create_outputfileName(input_file_name, par.dc, par.rhoc, par.outlierDeltaFactor);
    std::filesystem::path outDir = temp_outDir.parent_path() / output_file_name;

    std::ofstream clueOut(outDir);

    clueOut << "index,x,y,layer,weight,rho,delta,nh,isSeed,clusterId\n";
    for (unsigned int i = 0; i < results.n; i++) {
      clueOut << i << "," << results.x[i] << "," << results.y[i] << "," << results.layer[i] << "," << results.weight[i]
              << "," << results.rho[i] << "," << (results.delta[i] > 999 ? 999 : results.delta[i]) << ","
              << results.nearestHigher[i] << "," << results.isSeed[i] << "," << results.clusterIndex[i] << "\n";
    }

    clueOut.close();

    std::cout << "Ouput was saved in " << outDir << std::endl;
  }
}

DEFINE_FWK_MODULE(CLUEOutputProducer);