#include <fstream>
#include <memory>
#include "Framework/ESProducer.h"
#include "Framework/EventSetup.h"
#include "Framework/ESPluginFactory.h"
#include "DataFormats/LayerTilesConstants.h"
#include "DataFormats/PointsCloud.h"

class PointsCloudESProducer : public edm::ESProducer {
public:
  PointsCloudESProducer(std::filesystem::path const& inputFile) : data_{inputFile} {}
  void produce(edm::EventSetup& eventSetup);

private:
  std::filesystem::path data_;
};

void PointsCloudESProducer::produce(edm::EventSetup& eventSetup) {
  // Create empty PointsCloud
  PointsCloud cloud;
  for (int l = 0; l < NLAYERS; l++) {
    // open csv file
    std::ifstream iFile(data_);
    std::string value = "";
    // Iterate through each line and split the content using delimeter
    while (getline(iFile, value, ',')) {
      cloud.x.push_back(std::stof(value));
      getline(iFile, value, ',');
      cloud.y.push_back(std::stof(value));
      getline(iFile, value, ',');
      cloud.layer.push_back(std::stoi(value) + l);
      getline(iFile, value);
      cloud.weight.push_back(std::stof(value));
    }
    iFile.close();
  }
  cloud.n = cloud.x.size();
  auto pc = std::make_unique<PointsCloud>(cloud);
  eventSetup.put(std::move(pc));
}

DEFINE_FWK_EVENTSETUP_MODULE(PointsCloudESProducer);