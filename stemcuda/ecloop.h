// ecloop.h
#ifndef _ECLOOP_H_
#define _ECLOOP_H_

#include <vector>
#include "stempy/reader.h"
#include "stempy/electron.h"

namespace stempy {

void gpu_proc(uint16_t *&h_frames, int *&h_imgNums, int& nfilled,
  bool& done,
  bool& buffer_full, std::mutex& buffer_mutex,
  std::condition_variable& cd_buffer_full,
  Events& events, std::unique_ptr<std::mutex[]>& positionMutexes,
  int device, int npixelsperframe, int nframesperproc,
  int backgroundThreshold, int xRayThreshold,
  int fsparse = 10, int frame_dim1 = 576, int frame_dim2 = 576,
  bool verbose = false);

}
#endif
