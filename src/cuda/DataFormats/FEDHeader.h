#ifndef DataFormats_FEDRawData_FEDHeader_h
#define DataFormats_FEDRawData_FEDHeader_h

/** \class FEDHeader
 *  Helper class to interpret/create FED header words.
 *
 *  \author N. Amapane - CERN, R. Mommsen - FNAL
 */

#include <cstdint>

struct fedh_struct;

class FEDHeader {
public:
  /// Constructor
  FEDHeader(const unsigned char* header);

  /// Destructor
  ~FEDHeader();

  /// Event Trigger type identifier
  uint8_t triggerType() const;

  /// Level-1 event number generated by the TTC system
  uint32_t lvl1ID() const;

  /// The bunch crossing number
  uint16_t bxID() const;

  /// Identifier of the FED
  uint16_t sourceID() const;

  /// Version identifier of the FED data format
  uint8_t version() const;

  /// 0 -> the current header word is the last one.
  /// 1-> other header words can follow
  /// (always 1 for ECAL)
  bool moreHeaders() const;

  /// Check that the header is OK
  bool check() const;

  /// Set all fields in the header
  static void set(unsigned char* header, uint8_t triggerType, uint32_t lvl1ID,
                  uint16_t bxID, uint16_t sourceID, uint8_t version = 0,
                  bool moreHeaders = false);

  static const uint32_t length;

private:
  const fedh_struct* theHeader;
};
#endif  // DataFormats_FEDRawData_FEDHeader_h
