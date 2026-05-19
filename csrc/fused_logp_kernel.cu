#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

template <typename scalar_t>
__device__ __forceinline__ scalar_t blockReduceMax(scalar_t val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    float f_val = static_cast<float>(val);

    for (int offset = 16; offset > 0; offset /= 2)
        f_val = max(f_val, __shfl_down_sync(0xffffffff, f_val, offset));

    if (lane == 0) shared[wid] = f_val;
    __syncthreads();

    f_val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : -1e20f;
    if (wid == 0) {
        for (int offset = 16; offset > 0; offset /= 2)
            f_val = max(f_val, __shfl_down_sync(0xffffffff, f_val, offset));
    }
    return static_cast<scalar_t>(f_val);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t blockReduceSum(scalar_t val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    float f_val = static_cast<float>(val);

    for (int offset = 16; offset > 0; offset /= 2)
        f_val += __shfl_down_sync(0xffffffff, f_val, offset);

    if (lane == 0) shared[wid] = f_val;
    __syncthreads();

    f_val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (wid == 0) {
        for (int offset = 16; offset > 0; offset /= 2)
            f_val += __shfl_down_sync(0xffffffff, f_val, offset);
    }
    return static_cast<scalar_t>(f_val);
}

template <typename scalar_t>
__global__ void fused_logp_forward_kernel(
    const scalar_t* __restrict__ logits,      // [TotalTokens, VocabSize]
    const int64_t* __restrict__ token_ids,   // [TotalTokens]
    scalar_t* __restrict__ output,           // [TotalTokens]
    int vocab_size) {

    int row = blockIdx.x;
    const scalar_t* row_logits = logits + row * vocab_size;

    float local_max = -1e20f;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        local_max = max(local_max, static_cast<float>(row_logits[i]));
    }
    float max_val = blockReduceMax<float>(local_max);

    __shared__ float res_max;
    if (threadIdx.x == 0) res_max = max_val;
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
        local_sum += expf(static_cast<float>(row_logits[i]) - res_max);
    }
    float sum_val = blockReduceSum<float>(local_sum);

    __shared__ float res_sum;
    if (threadIdx.x == 0) res_sum = sum_val;
    __syncthreads();

    if (threadIdx.x == 0) {
        int64_t target_id = token_ids[row];
        if (target_id >= 0 && target_id < vocab_size) {
            float target_logit = static_cast<float>(row_logits[target_id]);
            output[row] = static_cast<scalar_t>(target_logit - res_max - logf(res_sum));
        } else {
            output[row] = static_cast<scalar_t>(0.0f);
        }
    }
}

torch::Tensor fused_logp_forward(torch::Tensor logits, torch::Tensor token_ids) {
    TORCH_CHECK(logits.is_cuda(), "logits must be a CUDA tensor");
    TORCH_CHECK(token_ids.is_cuda(), "token_ids must be a CUDA tensor");

    auto logits_contig = logits.contiguous();
    auto token_ids_contig = token_ids.contiguous();

    int64_t total_tokens = logits.size(0);
    int64_t vocab_size = logits.size(1);
    auto output = torch::empty({total_tokens}, logits.options());

    const int threads = 256;
    const int blocks = total_tokens;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
        logits.scalar_type(), "fused_logp_kernel", ([&] {
        fused_logp_forward_kernel<scalar_t><<<blocks, threads>>>(
            logits_contig.data_ptr<scalar_t>(),
            token_ids_contig.data_ptr<int64_t>(),
            output.data_ptr<scalar_t>(),
            vocab_size);
    }));

    return output;
}
