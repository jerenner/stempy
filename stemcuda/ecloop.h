// ecloop.h
#ifndef _ECLOOP_H_
#define _ECLOOP_H_

#include <vector>
#include "stempy/reader.h"
#include "stempy/electron.h"

namespace stempy {

void gpu_proc(uint16_t *&h_frames, int *&h_imgNums, int& nfilled,
  bool& done,
  bool& buffer_full, std::mutex& buffer_mutex, std::condition_variable& cd_buffer_full,
  Events& events, std::unique_ptr<std::mutex[]>& positionMutexes,
  uint16_t device, uint32_t npixelsperframe, uint32_t nframesperproc,
  uint16_t backgroundThreshold, uint16_t xRayThreshold,
  uint16_t fsparse = 10);

// template <typename FrameType>
// void ecloop_wrapper(std::vector<FrameType>& frame, uint32_t nvalues,
//                   double backgroundThreshold, double xRayThreshold);

}
#endif
