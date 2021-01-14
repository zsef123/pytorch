#include <ATen/native/Sorting.h>
#include <ATen/native/DispatchStub.h>

#include <ATen/ATen.h>
#include <c10/macros/Macros.h>
#include <ATen/Dispatch.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/native/SortingUtils.h>
#include <ATen/native/cuda/SortingUtils.cuh>
#include <ATen/cuda/CUDAApplyUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <ATen/native/cuda/SortingCommon.cuh>
#include <ATen/MemoryOverlap.h>

#include <THC/THCThrustAllocator.cuh>

#include <THC/THCSortUtils.cuh>

namespace at { namespace native {

namespace {

template <typename T, bool handleNaN = false>
struct LTComp {
  __device__ inline bool operator()(const T& lhs, const T& rhs) const {
      return (handleNaN && _isnan<T>(rhs) && !_isnan<T>(lhs)) || lhs < rhs;
  }
};

template <typename T, bool handleNaN = false>
struct GTComp {
  __device__ inline bool operator()(const T& lhs, const T& rhs) const {
      return (handleNaN && _isnan<T>(rhs) && !_isnan<T>(lhs)) || lhs > rhs;
  }
};

template <typename T, typename IndexType, bool handleNaN = true>
struct ThrustSliceLTOp {
ThrustSliceLTOp(int64_t size) : sliceSize(size) {}
  __device__ bool operator()(const thrust::tuple<int64_t, T>& lhs, const thrust::tuple<int64_t, T>& rhs) const {
    IndexType segA = static_cast<IndexType>(thrust::get<0>(lhs) / sliceSize);
    IndexType segB = static_cast<IndexType>(thrust::get<0>(rhs) / sliceSize);
    if (segA != segB)
        return segA < segB;
    else
        return (handleNaN && _isnan<T>(thrust::get<1>(rhs)) && !_isnan<T>(thrust::get<1>(lhs))) || thrust::get<1>(lhs) < thrust::get<1>(rhs);
  }
  const IndexType sliceSize;
};

template <typename T, typename IndexType, bool handleNaN = true>
struct ThrustSliceGTOp {
ThrustSliceGTOp(int64_t size) : sliceSize(size) {}
  __device__ bool operator()(const thrust::tuple<int64_t, T>& lhs, const thrust::tuple<int64_t, T>& rhs) const {
    IndexType segA = static_cast<IndexType>(thrust::get<0>(lhs) / sliceSize);
    IndexType segB = static_cast<IndexType>(thrust::get<0>(rhs) / sliceSize);
    if (segA != segB)
        return segA < segB;
    else
        return (handleNaN && _isnan<T>(thrust::get<1>(rhs)) && !_isnan<T>(thrust::get<1>(lhs))) || thrust::get<1>(lhs) > thrust::get<1>(rhs);
  }
  const IndexType sliceSize;
};

// Sorts (key, values) pairs (in different tensors) in-place; i.e.,
// modifies the input `keys` and `values`
template <typename K, typename V,
          int KeyDims, int ValueDims,
          typename Comparator, typename IndexType, int Power2SortSize>
C10_LAUNCH_BOUNDS_1(1024)
__global__ void
bitonicSortKVInPlace(cuda::detail::TensorInfo<K, IndexType> keys,
                     IndexType keySlices,
                     IndexType keySliceSize,
                     IndexType keySliceStride,
                     cuda::detail::TensorInfo<V, IndexType> values,
                     IndexType valueSliceStride,
                     Comparator comp) {
    // Find the slice of the tensor that we are sorting
    const IndexType linearIndex = getLinearBlockId<IndexType>();
    // Tiling the slices could have us be out of bounds, if there are a
    // lot of slices to sort
    if (linearIndex >= keySlices)
        return;

    __shared__ K sharedKeys[Power2SortSize];
    __shared__ V sharedValues[Power2SortSize];
    __shared__ bool sharedValid[Power2SortSize];

    const IndexType keyStartOffset =
        cuda::detail::IndexToOffset<K, IndexType, KeyDims>::get(linearIndex, keys);
    const IndexType valueStartOffset =
        cuda::detail::IndexToOffset<V, IndexType, ValueDims>::get(linearIndex, values);

    // If the sort size is 1, the data is already sorted
    if (Power2SortSize == 1)
        return;

    // Otherwise, each thread is responsible for loading and storing 2
    // elements. The sort size is guaranteed to be >= 2
    const IndexType elem1 = threadIdx.x;
    const IndexType elem2 = threadIdx.x + (Power2SortSize / 2);

    bool valid1 = (elem1 < keySliceSize);

    sharedKeys[elem1] = valid1 ?
        keys.data[keyStartOffset + elem1 * keySliceStride] : static_cast<K>(0);
    sharedValues[elem1] = valid1 ?
        values.data[valueStartOffset + elem1 * valueSliceStride] : static_cast<V>(0);
    sharedValid[elem1] = valid1;

    bool valid2 = (elem2 < keySliceSize);
    sharedKeys[elem2] = valid2 ?
        keys.data[keyStartOffset + elem2 * keySliceStride] : static_cast<K>(0);
    sharedValues[elem2] = valid2 ?
        values.data[valueStartOffset + elem2 * valueSliceStride] : static_cast<V>(0);
    sharedValid[elem2] = valid2;

    bitonicSort<Comparator, K, V, IndexType, Power2SortSize>(
        sharedKeys, sharedValues, sharedValid, comp);

    // elem1 and elem2 values might be out-of-range, if the data size we are
    // sorting is smaller than half the power2 size
    if (valid1) {
        keys.data[keyStartOffset + elem1 * keySliceStride] = sharedKeys[elem1];
        values.data[valueStartOffset + elem1 * valueSliceStride] = sharedValues[elem1];
    }

    if (valid2) {
        keys.data[keyStartOffset + elem2 * keySliceStride] = sharedKeys[elem2];
        values.data[valueStartOffset + elem2 * valueSliceStride] = sharedValues[elem2];
    }
}
// In alignment with default sort on a c++ map, this function
// will permute key and value tensors identically, and
// in such a way that the 'key' tensor is ordered numerically
template<typename scalar_t>
void sort_key_value_inplace(
        Tensor& values,
        Tensor& indices,
        int64_t dim,
        bool descending) {
    checkSameSize("sort_key_value_inplace", {values, "values", 1}, {indices, "indices", 2});
    TORCH_CHECK((values.dim() == 0 ? 1 : values.dim()) <= MAX_CUTORCH_DIMS,
                CUTORCH_DIM_WARNING);
    TORCH_CHECK((indices.dim() == 0 ? 1 : indices.dim()) <= MAX_CUTORCH_DIMS,
                CUTORCH_DIM_WARNING);

    int64_t in_elements = values.numel();
    if (in_elements == 0)
        return;

    int64_t key_slice_size = values.size(dim);
    int64_t key_slices = in_elements / key_slice_size;

    int64_t ceil_pow_of_2 = nextHighestPowerOf2(key_slice_size);
    if (ceil_pow_of_2 > 2048)
        AT_ERROR("sortKeyValueInplace only works for sizes <= 2048 at present");

    dim3 grid;
    if (!getGridFromTiles(key_slice_size, grid))
        AT_ERROR("sortKeyValueInplace Slice to sort is too large");

#define HANDLE_CASE(INDEX_TYPE, KEY_DIMS, SORT_SIZE)                               \
  do {                                                                             \
    dim3 block(SORT_SIZE <= 2 ? 1 : SORT_SIZE / 2);                                \
                                                                                   \
    if (descending) {                                                              \
        bitonicSortKVInPlace<scalar_t, int64_t, KEY_DIMS, -1,                      \
                            GTComp<scalar_t, true>, INDEX_TYPE, SORT_SIZE>         \
            <<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(               \
            value_info,                                                            \
            static_cast<INDEX_TYPE>(key_slices),                                   \
            static_cast<INDEX_TYPE>(key_slice_size),                               \
            static_cast<INDEX_TYPE>(value_info.strides[collapse_value_dims]),      \
            indices_info,                                                          \
            static_cast<INDEX_TYPE>(indices_info.strides[collapse_indice_dims]),   \
            GTComp<scalar_t, true>());                                             \
    } else {                                                                       \
        bitonicSortKVInPlace<scalar_t, int64_t, KEY_DIMS, -1,                      \
                            LTComp<scalar_t, true>, INDEX_TYPE, SORT_SIZE>         \
            <<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(               \
            value_info,                                                            \
            static_cast<INDEX_TYPE>(key_slices),                                   \
            static_cast<INDEX_TYPE>(key_slice_size),                               \
            static_cast<INDEX_TYPE>(value_info.strides[collapse_value_dims]),      \
            indices_info,                                                          \
            static_cast<INDEX_TYPE>(indices_info.strides[collapse_indice_dims]),   \
            LTComp<scalar_t, true>());                                             \
    }                                                                              \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                                \
  } while (0)

#define HANDLE_SORT_CASE(INDEX_TYPE, KEY_DIMS)          \
  {                                                     \
    switch (ceil_pow_of_2) {                            \
      case 2048:                                        \
        HANDLE_CASE(INDEX_TYPE, KEY_DIMS, 2048);        \
        break;                                          \
      case 1024: case 512: case 256:                    \
        HANDLE_CASE(INDEX_TYPE, KEY_DIMS, 1024);        \
        break;                                          \
      case 128: case 64:                                \
        HANDLE_CASE(INDEX_TYPE, KEY_DIMS, 128);         \
        break;                                          \
      case 32: case 16: case 8: case 4: case 2:         \
        HANDLE_CASE(INDEX_TYPE, KEY_DIMS, 32);          \
        break;                                          \
      case 1:                                           \
        /* Nothing to do, data already sorted */        \
        break;                                          \
      default:                                          \
        TORCH_INTERNAL_ASSERT(false);                   \
    }                                                   \
  }

    // The constructed key/values tensor info is used to select the slice
    // we are sorting on a per-block basis
    if (cuda::detail::canUse32BitIndexMath(values)) {
        auto value_info = cuda::detail::getTensorInfo<scalar_t, uint32_t>(values);
        value_info.reduceDim(dim);
        int collapse_value_dims = value_info.collapseDims(dim);

        auto indices_info = cuda::detail::getTensorInfo<int64_t, uint32_t>(indices);
        indices_info.reduceDim(dim);
        int collapse_indice_dims = indices_info.collapseDims(dim);

        if (value_info.isContiguous()) {
            HANDLE_SORT_CASE(uint32_t, -2);
        }
        else {
            switch (value_info.dims) {
                case 2:
                HANDLE_SORT_CASE(uint32_t, 2);
                break;
                default:
                HANDLE_SORT_CASE(uint32_t, -1);
                break;
            }
        }
    }
    else {
        auto value_info = cuda::detail::getTensorInfo<scalar_t, uint64_t>(values);
        value_info.reduceDim(dim);
        int collapse_value_dims = value_info.collapseDims(dim);

        auto indices_info = cuda::detail::getTensorInfo<int64_t, uint64_t>(indices);
        indices_info.reduceDim(dim);
        int collapse_indice_dims = indices_info.collapseDims(dim);

        // int64_t case is rare, just instantiate the generic version
        HANDLE_SORT_CASE(uint64_t, -1);
    }

#undef HANDLE_CASE
#undef HANDLE_SORT_CASE
}

template<typename scalar_t>
void sort_via_thrust(
        Tensor& values,
        Tensor& indices,
        int64_t dim,
        bool descending) {
    int64_t nDims = values.dim() == 0 ? 1 : values.dim();

    int64_t totalElements = values.numel();
    int64_t sliceSize = values.dim() == 0 ? 1 : values.size(dim);
    int64_t sliceStride = values.dim() == 0 ? 1 : values.stride(dim);

    // We perform a vectorized segmented sort in Thrust.
    // Say we are sorting a (2, 3) tensor. We have in flattened form:
    // values 0.4 1.2 5.3 6.2 1.3 2.3
    // indices  0   1   2   3   4   5
    // where indices is a global index (across all slices)

    // First we sort by values, globally:
    // values 6.2 5.3 2.3 1.2 1.3 0.4
    // indices  3   2   5   1   4   0

    // Then we stable sort by segment, which is index / 3:
    // values 5.3 1.2 0.4 6.2 2.3 1.3
    // indices  2   1   0   3   5   4

    // Then we translate the global index to a per-slice Lua index
    // (index % 3) + 1:
    // values 5.3 1.2 0.4 6.2 2.3 1.3
    // indices  3   2   1   1   3   2

    // This method can only work if the slice we are sorting (`dim`) is
    // innermost, and both values and indices are contiguous. We do this
    // by re-arranging the input into this form as needed, which will
    // unfortunately allocate memory if the request is not in this form.
    // Vectorized sort is slower than iterated sort if the number of
    // slices is small (since we're sorting twice, instead of invoking a
    // smaller sort `numSlices` times), but the Thrust sort
    // implementation here is a catch-all, so we're not looking for
    // efficiency, but instead correctness.

    auto trKeys = at::alias(values);
    auto trIndices = at::alias(indices);
    if (dim != nDims - 1) {
        trKeys.transpose_(dim, nDims - 1);
        trIndices.transpose_(dim, nDims - 1);
    }
    auto trContigKey = trKeys.contiguous();
    auto trContigIndices = trIndices.contiguous();

    auto thrustAlloc = THCThrustAllocator(globalContext().lazyInitCUDA());

    thrust::device_ptr<scalar_t> keyIter(trContigKey.data_ptr<scalar_t>());
    // Since we are composing a global index across all segments rather
    // than a per-segment index, we treat the memory as int so we don't
    // have problems sorting slices < 2^24 but where the entire tensor
    // has more than 2^24 elements
    thrust::device_ptr<int64_t> indexIter(trContigIndices.data_ptr<int64_t>());

    // Fill the indices with a global index across all slices
    thrust::counting_iterator<int64_t> countIter(0);

    thrust::copy(
#if CUDA_VERSION >= 7000 || defined __HIP_PLATFORM_HCC__
        thrust::cuda::par(thrustAlloc).on(c10::cuda::getCurrentCUDAStream()),
#endif
        countIter, countIter + totalElements, indexIter);

    auto begin = thrust::make_zip_iterator(thrust::make_tuple(indexIter, keyIter));

    auto thrust_sort = [&] (auto comp) {
        thrust::sort(
#if CUDA_VERSION >= 7000 || defined __HIP_PLATFORM_HCC__
            thrust::cuda::par(thrustAlloc).on(c10::cuda::getCurrentCUDAStream()),
#endif
            begin, begin + totalElements, comp);
    };

    if (descending) {
        if (cuda::detail::canUse32BitIndexMath(trContigKey))
            thrust_sort(ThrustSliceGTOp<scalar_t, uint32_t, true>(sliceSize));
        else
            thrust_sort(ThrustSliceGTOp<scalar_t, uint64_t, true>(sliceSize));
    }
    else {
        if (cuda::detail::canUse32BitIndexMath(trContigKey))
            thrust_sort(ThrustSliceLTOp<scalar_t, uint32_t, true>(sliceSize));
        else
            thrust_sort(ThrustSliceLTOp<scalar_t, uint64_t, true>(sliceSize));
    }

    // Translate the global integer 0-based index to a per-slice real
    // Lua index
    thrust::for_each(
#if CUDA_VERSION >= 7000 || defined __HIP_PLATFORM_HCC__
        thrust::cuda::par(thrustAlloc).on(c10::cuda::getCurrentCUDAStream()),
#endif
        indexIter, indexIter + totalElements,
        GlobalIndexToPerSliceIndex(sliceSize));

  // Reverse the transposition as needed
  if (dim != nDims - 1) {
    trContigKey.transpose_(dim, nDims - 1);
    trContigIndices.transpose_(dim, nDims - 1);
  }

  // Then copy back to the expected output
  values.copy_(trContigKey);
  indices.copy_(trContigIndices);
}

static void sort_kernel(
        Tensor& values,
        Tensor& indices,
        int64_t dim,
        bool descending) {
    auto indices_arg = TensorArg(indices, "indices", 2);
    checkSameGPU("sort_kernel", {values, "values", 1}, indices_arg);
    checkScalarType("sort_kernel", indices_arg, kLong);

    TORCH_CHECK((values.dim() == 0 ? 1 : values.dim()) <= MAX_CUTORCH_DIMS,
                CUTORCH_DIM_WARNING);
    TORCH_CHECK((indices.dim() == 0 ? 1 : indices.dim()) <= MAX_CUTORCH_DIMS,
                CUTORCH_DIM_WARNING);

    dim = at::maybe_wrap_dim(dim, values);

    int64_t slice_size = values.dim() == 0 ? 1 : values.size(dim);

#if CUDA_VERSION >= 8000
#if defined(THC_REAL_IS_DOUBLE) || defined(THC_REAL_IS_LONG)
    const int64_t max_slice_size = 1024;
#else
    const int64_t max_slice_size = 2048;
#endif
#else
    const int64_t max_slice_size = 2048;
#endif

    if (slice_size <= max_slice_size) {
        // Fill `indices` (the values) with the slice-relative index.
        Long_fillSliceWithIndex(indices, dim);

        // Sort using our in-place k/v kernel that supports arbitrary layout
        AT_DISPATCH_ALL_TYPES_AND(
            ScalarType::Half, values.scalar_type(),
            "sort_key_value_inplace", [&]() {
                sort_key_value_inplace<scalar_t>(values, indices, dim, descending);
        });
    }
    else {
        AT_DISPATCH_ALL_TYPES_AND(
            ScalarType::Half, values.scalar_type(),
            "sort_via_thrust", [&]() {
                sort_via_thrust<scalar_t>(values, indices, dim, descending);
        });
    }
  AT_CUDA_CHECK(cudaGetLastError());
}

} // anonymous namespace

REGISTER_DISPATCH(sort_stub, &sort_kernel);

}} // at::native
