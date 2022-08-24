#ifndef Points_Cloud_h
#define Points_Cloud_h

#include <vector>

struct PointsCloud {
  PointsCloud() = default;
  PointsCloud(unsigned int const& n) {
    x.resize(n);
    y.resize(n);
    layer.resize(n);
    weight.resize(n);

    rho.resize(n);
    delta.resize(n);
    nearestHigher.resize(n);
    clusterIndex.resize(n);
    followers.resize(n);
    isSeed.resize(n);
  }
  std::vector<float> x;
  std::vector<float> y;
  std::vector<int> layer;
  std::vector<float> weight;

  std::vector<float> rho;
  std::vector<float> delta;
  std::vector<int> nearestHigher;
  std::vector<std::vector<int>> followers;
  std::vector<int> isSeed;
  std::vector<int> clusterIndex;
  // why use int instead of bool?
  // https://en.cppreference.com/w/cpp/container/vector_bool
  // std::vector<bool> behaves similarly to std::vector, but in order to be space efficient, it:
  // Does not necessarily store its elements as a contiguous array (so &v[0] + n != &v[n])

  unsigned int n;
};

#endif