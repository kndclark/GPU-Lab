// Proves, at runtime, which SM the kernel in this image was actually compiled for.
// If the image was built for the wrong arch, this fails loudly instead of silently
// falling back to PTX JIT -- which is exactly the confound the harness must avoid.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void probe(int *out) {
#if defined(__CUDA_ARCH__)
    out[0] = __CUDA_ARCH__;
#else
    out[0] = -1;
#endif
}

int main() {
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, 0) != cudaSuccess) {
        printf("FAIL: no CUDA device visible\n");
        return 1;
    }
    int *d = nullptr, h = 0;
    cudaMalloc(&d, sizeof(int));
    probe<<<1, 1>>>(d);
    // The launch error surfaces here, NOT from cudaDeviceSynchronize(). Checking only
    // the sync lets cudaErrorNoKernelImageForDevice slip through as an apparent success
    // with an unwritten output buffer.
    cudaError_t e = cudaGetLastError();
    if (e == cudaSuccess) e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        printf("FAIL: kernel did not run: %s\n", cudaGetErrorString(e));
        if (e == cudaErrorNoKernelImageForDevice)
            printf("      This image has no cubin for sm_%d%d and no PTX to JIT from --\n"
                   "      i.e. it was built for a different architecture. Correct by design.\n",
                   p.major, p.minor);
        return 2;
    }
    cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost);

    const char *tcal = getenv("TORCH_CUDA_ARCH_LIST");
    const char *narch = getenv("NVCC_GENCODE");
    printf("device               : %s\n", p.name);
    printf("hardware compute cap : %d.%d  (sm_%d%d)\n", p.major, p.minor, p.major, p.minor);
    printf("kernel compiled for  : __CUDA_ARCH__ = %d\n", h);
    printf("NVCC_GENCODE         : %s\n", narch ? narch : "(unset)");
    printf("TORCH_CUDA_ARCH_LIST : %s\n", tcal ? tcal : "(unset)");
    int match = (h / 10 == p.major * 10 + p.minor);
    printf("RESULT               : %s\n",
           match ? "OK - native cubin matches hardware"
                 : "MISMATCH - kernel arch != hardware arch (JIT fallback?)");
    return match ? 0 : 3;
}
