//ecloop.cu
#ifndef _ECLOOP_
#define _ECLOOP_

#include <iostream>
#include <thread>

#include "ecloop.h"

namespace stempy {

/*
  Performs the thresholding.

  frames: the address of the beginning of the frames array
  n: the number of elements in the entire frames array
  bgt: the background threshold
  xrt: the x-ray threshold
*/
__global__ void threshold(uint16_t *frames, uint64_t n, uint16_t bgt, uint16_t xrt)
{
  // Parallelize a loop over all elements in the frames array.
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = blockDim.x * gridDim.x;

  //printf("BlockIdx = %d, blockDim.x = %d, gridDim.x = %d; thread = %d\n\n",blockIdx.x, blockDim.x, gridDim.x, threadIdx.x);

  for (int j = index; j < n; j += stride) {
    //printf("Thresholding frame value %d",frames[j]);

    // Threshold the electron events
    if (frames[j] <= bgt || frames[j] >= xrt) {
      frames[j] = 0;
    }
  }
}


/*
  Performs the electron counting.

  frames: the address of the beginning of the frames array
  buf: the address of the beginning of a buffer array (same size as frames array) to contain the counted electrons
  width: the frame width
  height: the frame height
  nframes: the total number of frames
  npixelsperframe: the total number of pixels per frame
*/
__global__ void count_sparse(uint16_t *frames, int * buf,
                        uint16_t width, uint16_t height,
                        uint32_t npixelsperframe)
{

    // Loop over all frames, processing 1 frame per block.
    const int f = blockIdx.x;
    const uint32_t frameoffset = f*npixelsperframe;

    // Compute the number of entries in the sparse matrix per frame, and the number handled by each thread.
    const int thread = threadIdx.x;
    const uint32_t nentries_sparse_frame  = (uint32_t) (npixelsperframe/10);
    const uint32_t nentries_sparse_thread = (uint32_t) (nentries_sparse_frame/blockDim.x);
    const uint32_t bufoffset = f*nentries_sparse_frame + thread*nentries_sparse_thread;

    // Initialize the region of interest of this thread in the buffer to -1.
    for(uint32_t i = 0; i < nentries_sparse_thread; i++) {
      buf[bufoffset + i] = -1;
    }

    // Process this frame with blockDim threads.
    const int stride = blockDim.x;
    uint32_t ncounts = 0;
    for (uint32_t i = thread; i < npixelsperframe; i += stride) {
      int tx = 0, ty = 0;
      int row = i / width;
      int column = i % width;

      tx = (column + 1); ty = width;
      int rightNeighbourColumn = ((tx % ty) + ty) % ty; // mod((column + 1), width);

      tx = (column - 1); ty = width;
      int leftNeighbourColumn = ((tx % ty) + ty) % ty; // mod((column - 1), width);

      tx = (row - 1); ty = height;
      int topNeighbourRow = ((tx % ty) + ty) % ty;     // mod((row - 1), height);

      tx = (row + 1); ty = height;
      int bottomNeighbourRow = ((tx % ty) + ty) % ty;  // mod((row + 1), height);

      uint16_t pixelValue = frames[frameoffset + i];
      int bottomNeighbourRowIndex = bottomNeighbourRow * width;
      int topNeighbourRowIndex = topNeighbourRow * width;
      int rowIndex = row * width;

      bool event = true;

      // If we are on row 0, there are no pixels above this one
      if (row != 0) {
        // Check top sections
        // top
        event = event && pixelValue > frames[frameoffset + topNeighbourRowIndex + column];
        // top left
        event = event &&
                (column == 0 ||
                 pixelValue > frames[frameoffset + topNeighbourRowIndex + leftNeighbourColumn]);
        // top right
        event = event &&
                (column == width - 1 ||
                 pixelValue > frames[frameoffset + topNeighbourRowIndex + rightNeighbourColumn]);
      }

      // If we are on the bottom row, there are no pixels below this one
      if (event && row != height - 1) {
        // Check bottom sections
        // bottom
        event = event && pixelValue > frames[frameoffset + bottomNeighbourRowIndex + column];
        // bottom left
        event = event && (column == 0 ||
                        pixelValue >
                          frames[frameoffset + bottomNeighbourRowIndex + leftNeighbourColumn]);
        // bottom right
        event =
          event &&
          (column == width - 1 ||
           pixelValue > frames[frameoffset + bottomNeighbourRowIndex + rightNeighbourColumn]);
      }

      // left
      event = event &&
              (column == 0 || pixelValue > frames[frameoffset + rowIndex + leftNeighbourColumn]);
      // right
      event = event && (column == width - 1 ||
                      pixelValue > frames[frameoffset + rowIndex + rightNeighbourColumn]);

      // If this index corresponds to an electron, save it in the buffer.
      if (event) {

        if(ncounts < nentries_sparse_thread) {
          buf[bufoffset + ncounts] = i;
          ncounts++;
        }
        else {
          printf("** Too many counts %d in frame %d, thread = %d\\n",ncounts,f,thread);
        }

      }

    } // end loop over current frame

    // Ensure that all threads have reached this point before continuing.
    __syncthreads();

}

void gpu_proc(uint16_t *&h_frames, int *&h_imgNums, int& nfilled,
  bool& done,
  bool& buffer_full, std::mutex& buffer_mutex, std::condition_variable& cd_buffer_full,
  Events& events, std::unique_ptr<std::mutex[]>& positionMutexes,
  uint16_t device, uint32_t npixelsperframe, uint32_t nframesperproc,
  uint16_t backgroundThreshold, uint16_t xRayThreshold,
  uint16_t fsparse)
{

  bool debug = false;

  // Set the device this thread will interact with.
  cudaSetDevice(device);

  // Compute the number of bytes in the frames and buffers arrays.
  uint32_t nsparsepixelsperframe = (uint32_t) (npixelsperframe/fsparse);
  uint64_t bytes_frames = nframesperproc*npixelsperframe*sizeof(uint16_t);
  uint64_t bytes_sparse = nframesperproc*nsparsepixelsperframe*sizeof(int);
  uint64_t bytes_imgNums = nframesperproc*sizeof(int);
  uint64_t bytes_move   = 0;
  int nprocess = 0;

  // Pointers to the memory on the host (CPU) and device (GPU).
  int *h_imgNums_temp;
  uint16_t *d_frames;
  int *h_sparse, *d_sparse;

  // Create a new CUDA stream.
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // Lock the buffer mutex for allocation.
  {
    std::unique_lock<std::mutex> lk(buffer_mutex);
    cd_buffer_full.wait(lk, [&buffer_full]{ return buffer_full; });

    // Allocate (pinned) memory for the frames and sparse buffers on the CPU.
    if(debug) std::cout << "\n[GPU THREAD] Allocating pinned memory..." << std::endl;
    cudaMallocHost((void**)&h_frames, bytes_frames);
    cudaMallocHost((void**)&h_sparse, bytes_sparse);

    // Allocate space for a temporary array to store the image numbers.
    h_imgNums_temp = (int*) malloc(bytes_imgNums);

    // Allocate memory for the frames and sparse output buffer on the device.
    if(debug) std::cout << "\n[GPU THREAD] Allocating device memory..." << std::endl;
    cudaMalloc((void**)&d_frames, bytes_frames);
    cudaMalloc((void**)&d_sparse, bytes_sparse);

    // Set the buffer to empty and release the lock.
    buffer_full = false;
    lk.unlock();
    cd_buffer_full.notify_all();

  }

  // Loop until all events have been processed.
  for(;;) {

    // Only proceed when the frames buffer is full, or end the process if we are done with all frames.
    {
      if(debug) std::cout << "\n[GPU THREAD] Checking for full buffer..." << std::endl;
      std::unique_lock<std::mutex> lk(buffer_mutex);
      cd_buffer_full.wait(lk, [&buffer_full, &done]{ return buffer_full; });

      // Count the frames in the buffer if it is full.
      if(buffer_full) {

        // Save the number of frames to process.
        nprocess = nfilled;

        // Save the image numbers.
        std::copy(h_imgNums, h_imgNums + nprocess, h_imgNums_temp);

        // Move all frames to the GPU.
        if(debug) std::cout << "\n[GPU THREAD] Buffer full: moving " << nprocess << " frames to GPU" << std::endl;
        bytes_move = nprocess*npixelsperframe*sizeof(uint16_t);
        // if(debug) {
        //   for(int nn = 0; nn < npixelsperframe; nn++) std::cout << "[elem " << nn << "] = " << h_frames[nn + 10*npixelsperframe] << " ";
        //   std::cout << "\n" << std::endl;
        // }
        cudaMemcpyAsync(d_frames, h_frames, bytes_move, cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);

        // Set the frames buffer to empty again so that the reader threads can resume.
        if(debug) std::cout << "\n[GPU THREAD] Setting buffer to empty" << std::endl;
        nfilled = 0;
        buffer_full = false;
        lk.unlock();
        cd_buffer_full.notify_all();
      }
    }

    // Run the GPU-based execution.
    if(debug) std::cout << "\n[GPU THREAD] Running kernels" << std::endl;
    threshold<<<1, 1024, 0, stream>>>(d_frames, nprocess*npixelsperframe, backgroundThreshold, xRayThreshold);
    count_sparse<<<nprocess, 1024, 0, stream>>>(d_frames, d_sparse, 576, 576, npixelsperframe);

    // Retrieve the output sparse array.
    bytes_move = nprocess*nsparsepixelsperframe*sizeof(int);
    if(debug) std::cout << "\n[GPU THREAD] Transferring " << bytes_move << " bytes for " << nprocess << " sparse events" << std::endl;
    cudaMemcpyAsync(h_sparse, d_sparse, bytes_move, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    // if(debug) {
    //   for(int nn = 0; nn < nsparsepixelsperframe; nn++) std::cout << "[elem " << nn << "] = " << h_sparse[nn + 1*nsparsepixelsperframe] << " ";
    //   std::cout << "\n" << std::endl;
    // }

    // Process the output sparse array.
    if(debug) std::cout << "\n[GPU THREAD] Processing " << nprocess << " sparse events" << std::endl;

    // Here we need to calculate a modified number of sparse pixels per frame because we divided the
    //  array into 1024 threads. Although the sparse entries for each frame will be spaced by nsparsepixelsperframe,
    //  there will only be nsparseentriesperthread*1024 of them that are actually relevant, where nsparsepixelsperthread
    //  is the integer division of nsparsepixelsperframe by 1024. (Essentially there is some "left over" space in the sparse
    //  array because it is not evenly divisible by the number of threads).
    uint32_t nsparsepixelsperthread = (uint32_t) (nsparsepixelsperframe/1024);
    uint32_t nsparseentriesperframe = (uint32_t) (nsparsepixelsperthread*1024);
    for(uint16_t i = 0; i < nprocess; i++) {

      // Construct the sparse array (remove all -1s from the GPU output).
      std::vector<uint32_t> sparseFrame;
      uint16_t j = 0;
      int lastElem = -1;
      //if(debug) std::cout << "\n[GPU THREAD] Constructing sparse vector" << i << std::endl;
      while(j < nsparseentriesperframe) {
          lastElem = h_sparse[j + i * nsparsepixelsperframe];
          if(lastElem >= 0) sparseFrame.push_back(lastElem);
          j++;
      }

      // Place the sparse array in the events list (with a mutex lock at the corresponding position).
      {
        int position = h_imgNums_temp[i];
        //if(debug) std::cout << "\n[GPU THREAD] Placing sparse vector " << i << " at position " << position << std::endl;
        std::vector<uint32_t>& eventAtPosition = events[position];
        std::mutex& mutex = positionMutexes[position];
        std::unique_lock<std::mutex> positionLock(mutex);
        eventAtPosition.insert(eventAtPosition.end(), sparseFrame.begin(),
                                 sparseFrame.end());
      }
    }

    // done is true: all frames have been counted, so end the process.
    if(done) {
      return;
    }

  } // end main loop

}


// template <typename FrameType>
// void ecloop_wrapper(std::vector<FrameType>& frame, uint32_t nvalues, double backgroundThreshold, double xRayThreshold)
// {
//
//   uint16_t *h_frames;
//
//   //
//   Events events;
//   // Mutexes to protect writing to events object.
//   std::unique_ptr<std::mutex[]> positionMutexes;
//
//
//   // Allocate Unified Memory – accessible from CPU or GPU
//   cudaMallocManaged(&c_frame, nvalues*sizeof(FrameType));
//
//   // Copy the arrays into CUDA memory.
//   cudaMemcpy(c_frame, &frame[0], nvalues*sizeof(FrameType), cudaMemcpyHostToDevice);
//
//   // Execute the thresholding.
//   threshold<<<1024, 1>>>(c_frame, nvalues, backgroundThreshold, xRayThreshold);
//
//   // Wait for GPU to finish before accessing on host
//   cudaDeviceSynchronize();
//
//   // Copy data back to vector.
//   cudaMemcpy(&frame[0], c_frame, nvalues*sizeof(FrameType), cudaMemcpyDeviceToHost);
//
// }
// template void ecloop_wrapper(std::vector<uint16_t>& frame, uint32_t nvalues, double backgroundThreshold, double xRayThreshold);

}
#endif
