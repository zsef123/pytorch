#include <ATen/native/cuda/SortingCommon.cuh>


namespace at { namespace native {

void Long_fillSliceWithIndex(Tensor& out, int64_t dim) {
  TORCH_CHECK((out.dim() == 0 ? 1 : out.dim()) <= MAX_CUTORCH_DIMS,
              CUTORCH_DIM_WARNING);
  TORCH_CHECK(dim == 0 || dim < out.dim(),
              "Long_fillSliceWithIndex(): Indexing dim ", dim, " is out of bounds of tensor");

  int64_t in_elements = out.numel();
  if (in_elements <= 0) {
    return;
  }

  int64_t slice_size = out.dim() == 0 ? 1 : out.size(dim);
  int64_t num_slices = in_elements / slice_size;

  dim3 grid;
  if (!getGridFromTiles(num_slices, grid)) {
    AT_ERROR("Slice to fill with indices is too large");
  }

  dim3 block(std::min(
    slice_size,
    static_cast<int64_t>(cuda::getCurrentDeviceProperties()->maxThreadsPerBlock)
  ));

#define FILL_INDEX(INDEX_T, DIM)                                  \
  fillSliceWithIndex_kernel<INDEX_T, DIM>                         \
    <<<grid, block, 0, c10::cuda::getCurrentCUDAStream()>>>(      \
      info, num_slices, slice_size, info.strides[collapseDim]);   \
  C10_CUDA_KERNEL_LAUNCH_CHECK()

  if (cuda::detail::canUse32BitIndexMath(out)) {
    auto info = cuda::detail::getTensorInfo<int64_t, uint32_t>(out);
    info.reduceDim(dim);
    int collapseDim = info.collapseDims(dim);

    if (info.isContiguous()) {
      FILL_INDEX(uint32_t, -2);
    }
    else {
      if (info.dims == 1) {
        FILL_INDEX(uint32_t, 1);
      }
      else if (info.dims == 2) {
        FILL_INDEX(uint32_t, 2);
      }
      else {
        FILL_INDEX(uint32_t, -1);
      }
    }
  }
  else {
    auto info = cuda::detail::getTensorInfo<int64_t, uint64_t>(out);
    info.reduceDim(dim);
    int collapseDim = info.collapseDims(dim);

    // catch-all implementation
    FILL_INDEX(uint64_t, -1);
  }

#undef FILL_INDEX
}

}} // at::native
