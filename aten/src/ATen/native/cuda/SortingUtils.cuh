#pragma once

#include <THC/THCNumerics.cuh>
#include <ATen/NumericUtils.h>

// Collection of kernel sort routines
namespace at{

namespace {

template <typename T>
__device__ inline void swapVars(T& t1, T& t2) {
  T tmp = t1;
  t1 = t2;
  t2 = tmp;
}

template <typename Comparator, typename K, typename V>
__device__ inline void bitonicSwap(K& key1, V& value1, bool& valid1,
                                   K& key2, V& value2, bool& valid2,
                                   bool direction,
                                   const Comparator& comp) {
  // Invalid entries always sort to the end
  bool swap = (comp(key1, key2) && valid1) || !valid2;
  if (swap == direction) {
    swapVars(key1, key2);
    swapVars(value1, value2);
    swapVars(valid1, valid2);
  }
};

template <typename Comparator, typename K, typename V,
          typename IndexType, int Power2SortSize>
__device__ inline void bitonicSort(K keys[Power2SortSize],
                                   V values[Power2SortSize],
                                   bool valid[Power2SortSize],
                                   const Comparator& comp) {
#ifndef __HIP_PLATFORM_HCC__
#pragma unroll
#endif
  for (unsigned int size = 2; size < Power2SortSize; size *= 2) {
    bool flag = ((threadIdx.x & (size / 2)) != 0);

#ifndef __HIP_PLATFORM_HCC__
#pragma unroll
#endif
    for (unsigned int stride = size / 2; stride > 0; stride /= 2) {

      __syncthreads();

      unsigned int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
      bitonicSwap<Comparator, K, V>(
        keys[pos], values[pos], valid[pos],
        keys[pos + stride], values[pos + stride], valid[pos + stride],
        flag, comp);
    }
  }

#ifndef __HIP_PLATFORM_HCC__
#pragma unroll
#endif
  for (unsigned int stride = Power2SortSize / 2; stride > 0; stride /= 2) {

    __syncthreads();

    unsigned int pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
    bitonicSwap<Comparator, K, V>(
      keys[pos], values[pos], valid[pos],
      keys[pos + stride], values[pos + stride], valid[pos + stride],
      false, comp);
  }

  __syncthreads();

}

} // anonymous namespace
} // at namespace