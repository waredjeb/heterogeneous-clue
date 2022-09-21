#ifndef CLUE_config_h
#define CLUE_config_h

#include <vector>
#include <filesystem>
#include <regex>

struct Parameters {
  float dc = 20;
  float rhoc = 25;
  float outlierDeltaFactor = 2;
  bool produceOutput = true;
};

template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 6) {
  std::ostringstream out;
  out.precision(n);
  out << std::fixed << a_value;
  return out.str();
}

inline std::string create_outputfileName(std::string inputFileName, float dc, float rhoc, float outlierDeltaFactor) {
  std::string underscore = "_", suffix = "";
  suffix.append(underscore);
  suffix.append(to_string_with_precision(dc, 2));
  suffix.append(underscore);
  suffix.append(to_string_with_precision(rhoc, 2));
  suffix.append(underscore);
  suffix.append(to_string_with_precision(outlierDeltaFactor, 2));
  suffix.append(".csv");

  std::string tmpFileName;
  std::regex regexp("input");
  std::regex_replace(back_inserter(tmpFileName), inputFileName.begin(), inputFileName.end(), regexp, "output");

  std::string outputFileName;
  std::regex regexp2(".csv");
  std::regex_replace(back_inserter(outputFileName), tmpFileName.begin(), tmpFileName.end(), regexp2, suffix);

  return outputFileName;
}

#endif
