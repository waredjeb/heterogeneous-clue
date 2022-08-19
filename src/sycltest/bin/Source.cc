#include <cassert>
#include <iostream>
#include <fstream>
#include <filesystem>

#include "Source.h"

namespace edm {
  Source::Source(int maxEvents, int runForMinutes, ProductRegistry &reg, std::filesystem::path const &inputFile)
      : maxEvents_(maxEvents), runForMinutes_(runForMinutes), cloudToken_(reg.produces<PointsCloud>()) {
    for (int l = 0; l < NLAYERS; l++) {
      // open csv file
      std::ifstream iFile(inputFile);
      std::string value = "";
      // Iterate through each line and split the content using delimeter
      while (getline(iFile, value, ',')) {
        cloud_.x.push_back(std::stof(value));
        getline(iFile, value, ',');
        cloud_.y.push_back(std::stof(value));
        getline(iFile, value, ',');
        cloud_.layer.push_back(std::stoi(value) + l);
        getline(iFile, value);
        cloud_.weight.push_back(std::stof(value));
      }
      iFile.close();
    }
    cloud_.n = cloud_.x.size();

    if (runForMinutes_ < 0 and maxEvents_ < 0) {
      maxEvents_ = cloud_.n / NLAYERS;
    }
  }

  void Source::startProcessing() {
    if (runForMinutes_ >= 0) {
      startTime_ = std::chrono::steady_clock::now();
    }
  }

  std::unique_ptr<Event> Source::produce(int streamId, ProductRegistry const &reg) {
    if (shouldStop_) {
      return nullptr;
    }

    const int old = numEvents_.fetch_add(1);
    const int iev = old + 1;
    if (runForMinutes_ < 0) {
      if (old >= maxEvents_) {
        shouldStop_ = true;
        --numEvents_;
        return nullptr;
      }
    } else {
      if (numEvents_ - numEventsTimeLastCheck_ > static_cast<int>(cloud_.n)) {
        std::scoped_lock lock(timeMutex_);
        // if some other thread beat us, no need to do anything
        if (numEvents_ - numEventsTimeLastCheck_ > static_cast<int>(cloud_.n)) {
          auto processingTime = std::chrono::steady_clock::now() - startTime_;
          if (std::chrono::duration_cast<std::chrono::minutes>(processingTime).count() >= runForMinutes_) {
            shouldStop_ = true;
          }
          numEventsTimeLastCheck_ = (numEvents_ / cloud_.n) * cloud_.n;
        }
        if (shouldStop_) {
          --numEvents_;
          return nullptr;
        }
      }
    }
    auto ev = std::make_unique<Event>(streamId, iev, reg);

    ev->emplace(cloudToken_, cloud_);

    return ev;
  }
}  // namespace edm
