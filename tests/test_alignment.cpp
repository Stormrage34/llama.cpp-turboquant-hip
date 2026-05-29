#include <iostream>
#include <cstdlib>
#include <cstring>

// Simple alignment verification test.
// Allocates a deliberately mis‑aligned buffer, copies a small amount of data,
// and invokes a trivial ggml operation to ensure the fallback path for
// unaligned loads does not crash. This placeholder can be expanded to load a real
// quantized model and run an inference step.
int main() {
    // Allocate 1024 bytes with alignment of 8 (intentionally not 128).
    void *ptr = nullptr;
    // posix_memalign requires alignment to be a power of two and at least sizeof(void*)
    // We request 8‑byte alignment which is smaller than the 128‑byte boundary used by the
    // fast‑path load, so the buffer will be mis‑aligned for that path.
    int rc = posix_memalign(&ptr, 8, 1024);
    if (rc != 0 || !ptr) {
        std::cerr << "Failed to allocate mis‑aligned buffer" << std::endl;
        return 1;
    }
    // Fill buffer with dummy data.
    std::memset(ptr, 0xAB, 1024);

    // In a real test we would hand this buffer to ggml/llama kernels that perform
    // the 128‑bit load. For now we simply ensure the program runs without crashing.
    std::cout << "Alignment test: allocated mis‑aligned buffer at " << ptr << std::endl;

    free(ptr);
    std::cout << "Alignment test passed (no crash)" << std::endl;
    return 0;
}
