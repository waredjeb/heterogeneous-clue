#include <iostream>
#include <vector>

#include <CL/sycl.hpp>

#include "chooseDevice.h"

namespace cms::sycltools {
  std::vector<sycl::device> const& enumerateDevices(bool verbose) {
    static const std::vector<sycl::device> devices = sycl::device::get_devices(sycl::info::device_type::all);

    if (verbose) {
      std::cerr << "Found " << devices.size() << " SYCL devices:" << std::endl;
      for (auto const& device : devices)
        std::cerr << "  - " << device.get_backend() << ' ' << device.get_info<cl::sycl::info::device::name>()
                  << std::endl;
      std::cerr << std::endl;
    }
    return devices;
  }

  sycl::device chooseDevice(edm::StreamID id) {
    auto const& devices = enumerateDevices();
    auto const& device = devices[id % devices.size()];
    if (device.is_gpu() and device.get_backend() == sycl::backend::ext_oneapi_level_zero) {
      try {
        std::vector<sycl::device> subDevices =
            device.create_sub_devices<sycl::info::partition_property::partition_by_affinity_domain>(
                sycl::info::partition_affinity_domain::next_partitionable);
        auto const& subDevice = subDevices[id % subDevices.size()];
        std::cerr << "EDM stream " << id << " offload to tile " << id % subDevices.size() << " on device "
                  << id % devices.size() << std::endl;
        return subDevice;
      } catch (sycl::exception const& e) {
        std::cerr << "This GPU does not support splitting in multiple sub devices" << std::endl;
        std::cerr << "EDM stream " << id << " offload to " << device.get_info<cl::sycl::info::device::name>()
                  << " on backend " << device.get_backend() << std::endl;
        return device;
      }
    } else {
      std::cerr << "EDM stream " << id << " offload to " << device.get_info<cl::sycl::info::device::name>()
                << " on backend " << device.get_backend() << std::endl;
      return device;
    }
  }
}  // namespace cms::sycltools
