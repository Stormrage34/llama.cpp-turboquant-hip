#include "llama-infinity-cache.h"
#include "llama-model-loader.h"
#include "llama-util.h"

#include "gguf.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#ifdef __linux__
#include <sys/file.h> // For flock
#include <sys/mman.h> // For MAP_LOCKED, MAP_POPULATE
#endif

#ifdef _WIN32
#include <windows.h> // For LockFileEx, VirtualLock
#endif

namespace fs = std::filesystem;

// Magic for the Infinity Cache header: 'INFT'
static constexpr uint32_t LLAMA_INFINITY_CACHE_MAGIC = 0x494E4654;

// RDNA2 swizzle block size constants (mirrors llama-model-loader.cpp)
#ifdef RDNA2_CACHE_SWIZZLE
static constexpr uint32_t IQ4_XS_BLOCK_SIZE    = 136;
static constexpr uint32_t IQ4_XS_QS_SIZE_HOST   = 128;
static constexpr uint32_t IQ4_XS_META_SIZE_HOST  = 8;

static constexpr uint32_t Q4_K_BLOCK_SIZE        = 144;
static constexpr uint32_t Q4_K_QS_SIZE_HOST       = 128;
static constexpr uint32_t Q4_K_META_SIZE_HOST      = 16;

static constexpr uint32_t Q5_K_BLOCK_SIZE        = 176;
static constexpr uint32_t Q5_K_QS_SIZE_HOST       = 128;
static constexpr uint32_t Q5_K_META_SIZE_HOST      = 48;

// Check if a ggml_type should be swizzled based on compile flags.
static bool tensor_needs_swizzle(ggml_type tensor_type, uint32_t compile_flags) {
    switch (tensor_type) {
        case GGML_TYPE_IQ4_XS:  return true;  // Always swizzle IQ4_XS
        case GGML_TYPE_Q4_K:    return true;  // Always swizzle Q4_K (intra-block)
        case GGML_TYPE_Q5_K:    return compile_flags & 0x02; // ALL_SWIZZLE
        default:                return false;
    }
}

// SoA layout for IQ4_XS: [qs_blk0(128B)]...[qs_blkN-1(128B)] [meta_blk0(8B)]...[meta_blkN-1(8B)]
static void swizzle_iq4_xs_host(void * data, size_t nbytes) {
    const size_t n_blocks = nbytes / IQ4_XS_BLOCK_SIZE;
    if (n_blocks == 0) return;

    std::vector<uint8_t> tmp(nbytes);
    uint8_t * qs_dst   = tmp.data();
    uint8_t * meta_dst = tmp.data() + n_blocks * IQ4_XS_QS_SIZE_HOST;

    const uint8_t * src = (const uint8_t *)data;
    for (size_t b = 0; b < n_blocks; b++) {
        const uint8_t * blk = src + b * IQ4_XS_BLOCK_SIZE;
        memcpy(qs_dst + b * IQ4_XS_QS_SIZE_HOST, blk + 8, IQ4_XS_QS_SIZE_HOST);
        memcpy(meta_dst + b * IQ4_XS_META_SIZE_HOST, blk, IQ4_XS_META_SIZE_HOST);
    }
    memcpy(data, tmp.data(), nbytes);
}

// Q4_K intra-block: [qs(128B)] + [dm(4B)] + [scales(12B)]
static void swizzle_q4_K_host(void * data, size_t nbytes) {
    const size_t n_blocks = nbytes / Q4_K_BLOCK_SIZE;
    if (n_blocks == 0) return;

    uint8_t tmp[Q4_K_BLOCK_SIZE];
    constexpr uint32_t aos_qs_offset = 16; // dm(4B) + scales(12B)
    for (size_t b = 0; b < n_blocks; b++) {
        uint8_t * blk = (uint8_t *)data + b * Q4_K_BLOCK_SIZE;
        memcpy(tmp, blk, Q4_K_BLOCK_SIZE);
        memcpy(blk, tmp + aos_qs_offset, Q4_K_QS_SIZE_HOST);
        memcpy(blk + Q4_K_QS_SIZE_HOST, tmp, aos_qs_offset);
    }
}

// SoA layout for Q5_K: [qs_blk0(128B)]...[qs_blkN-1(128B)] [meta_blk0(48B)]...[meta_blkN-1(48B)]
static void swizzle_q5_K_host(void * data, size_t nbytes) {
    const size_t n_blocks = nbytes / Q5_K_BLOCK_SIZE;
    if (n_blocks == 0) return;

    std::vector<uint8_t> tmp(nbytes);
    uint8_t * qs_dst   = tmp.data();
    uint8_t * meta_dst = tmp.data() + n_blocks * Q5_K_QS_SIZE_HOST;

    const uint8_t * src = (const uint8_t *)data;
    for (size_t b = 0; b < n_blocks; b++) {
        const uint8_t * blk = src + b * Q5_K_BLOCK_SIZE;
        memcpy(qs_dst + b * Q5_K_QS_SIZE_HOST, blk + 48, Q5_K_QS_SIZE_HOST);
        memcpy(meta_dst + b * Q5_K_META_SIZE_HOST, blk, Q5_K_META_SIZE_HOST);
    }
    memcpy(data, tmp.data(), nbytes);
}

// Dispatch swizzle transform for a tensor of the given type.
static void apply_swizzle(ggml_type tensor_type, void * data, size_t nbytes) {
    switch (tensor_type) {
        case GGML_TYPE_IQ4_XS:  swizzle_iq4_xs_host(data, nbytes); break;
        case GGML_TYPE_Q4_K:    swizzle_q4_K_host(data, nbytes);   break;
        case GGML_TYPE_Q5_K:    swizzle_q5_K_host(data, nbytes);   break;
        default: break;
    }
}

#endif // RDNA2_CACHE_SWIZZLE

// Helper to compute SHA-256 hash of a file's content (e.g., GGUF header)
static std::string calculate_file_hash(const std::string &filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        return "";
    }

    // For GGUF, we only hash the header for model identity.
    // Read the GGUF header (up to the first tensor, or a fixed size if parsing is too complex here)
    // For simplicity, let's read the first 1MB of the file for hashing. A robust solution would parse GGUF.
    std::vector<char> buffer(1024 * 1024); // 1MB buffer for hashing
    file.read(buffer.data(), buffer.size());
    size_t bytes_read = file.gcount();

    return llama_sha256(buffer.data(), bytes_read);
}

// Generates the full path for a cache file.
std::string llama_infty_cache_path(const std::string &model_hash, const std::string &cache_dir) {
    return fs::path(cache_dir) / (model_hash + ".cache");
}

bool llama_infty_cache_create_or_open(
    const std::string    &gguf_path,
    const std::string    &cache_dir,
    const llama_mmap_flags &mmap_flags,
    const uint32_t       current_compile_flags,
    const size_t         max_cache_size,
    llama_mmap           **out_mapping
) {
    // Ensure cache directory exists
    if (!fs::exists(cache_dir)) {
        try {
            fs::create_directories(cache_dir);
        } catch (const fs::filesystem_error &e) {
            LLAMA_LOG_ERROR("%s: Failed to create cache directory '%s': %s
", __func__, cache_dir.c_str(), e.what());
            return false;
        }
    }

    // 1. Calculate model hash (SHA-256 of GGUF header)
    std::string model_hash = calculate_file_hash(gguf_path);
    if (model_hash.empty()) {
        LLAMA_LOG_WARN("%s: Failed to calculate hash for GGUF file '%s'. Falling back to normal mmap.
", __func__, gguf_path.c_str());
        // Fallback to normal mmap logic (caller's responsibility)
        return false;
    }

    const std::string cache_file_path = llama_infty_cache_path(model_hash, cache_dir);

    // Get original GGUF file size for header validation
    struct llama_file gguf_file(gguf_path.c_str(), true);
    if (!gguf_file.is_open()) {
        LLAMA_LOG_ERROR("%s: Failed to open GGUF file '%s' for size check.\n", __func__, gguf_path.c_str());
        return false;
    }
    const uint64_t gguf_model_size = gguf_file.size;
    gguf_file.close();

    // 2. Attempt to open existing cache file (cache hit)
    if (fs::exists(cache_file_path)) {
        struct llama_file cache_file(cache_file_path.c_str(), true); // Read-only
        if (!cache_file.is_open()) {
            LLAMA_LOG_WARN("%s: Failed to open existing cache file '%s'. Proceeding to create new cache.\n", __func__, cache_file_path.c_str());
            goto cache_miss; // Jump to cache miss path
        }

    #ifdef __linux__
        // Acquire shared lock
        if (flock(cache_file.fd, LOCK_SH | LOCK_NB) != 0) {
            if (errno == EWOULDBLOCK) {
                LLAMA_LOG_INFO("%s: Cache file '%s' is exclusively locked by another process. Falling back to normal mmap.\n", __func__, cache_file_path.c_str());
            } else {
                LLAMA_LOG_ERROR("%s: Failed to acquire shared lock on cache file '%s': %s. Falling back to normal mmap.\n", __func__, cache_file_path.c_str(), strerror(errno));
            }
            cache_file.close();
            return false; // Fallback to normal mmap
        }
    #endif

        // Read and validate header
        infty_cache_header header;
        if (cache_file.read(&header, sizeof(header)) != sizeof(header)) {
            LLAMA_LOG_WARN("%s: Failed to read header from cache file '%s'. Invalidating cache and recreating.\n", __func__, cache_file_path.c_str());
            cache_file.close();
            fs::remove(cache_file_path); // Invalidate
            goto cache_miss;
        }

        bool header_valid = true;
        if (header.magic != LLAMA_INFINITY_CACHE_MAGIC) {
            LLAMA_LOG_WARN("%s: Cache file '%s' has invalid magic (0x%x != 0x%x). Invalidating.\n", __func__, cache_file_path.c_str(), header.magic, LLAMA_INFINITY_CACHE_MAGIC);
            header_valid = false;
        }
        if (std::memcmp(header.gguf_hash, model_hash.data(), 32) != 0) {
            LLAMA_LOG_WARN("%s: Cache file '%s' has mismatched GGUF hash. Invalidating.\n", __func__, cache_file_path.c_str());
            header_valid = false;
        }
        if (header.compile_flags != current_compile_flags) {
            LLAMA_LOG_WARN("%s: Cache file '%s' has mismatched compile flags (0x%x != 0x%x). Invalidating.\n", __func__, cache_file_path.c_str(), header.compile_flags, current_compile_flags);
            header_valid = false;
        }
        if (header.model_data_size != gguf_model_size) {
            LLAMA_LOG_WARN("%s: Cache file '%s' has mismatched model data size (%" PRIu64 " != %" PRIu64 "). Invalidating.\n", __func__, cache_file_path.c_str(), header.model_data_size, gguf_model_size);
            header_valid = false;
        }

        if (header_valid) {
            LLAMA_LOG_INFO("%s: Successfully opened and validated Infinity Cache file '%s'.\n", __func__, cache_file_path.c_str());
            // Create llama_mmap object for the cache file. Offset by sizeof(header).
            // The mmap_flags already contain whether to lock or use hugetlb.
            try {
                *out_mapping = new llama_mmap(&cache_file, mmap_flags.prefetch ? -1 : 0, mmap_flags.numa, mmap_flags.use_hugetlb, mmap_flags.mem_lock, sizeof(header));
                cache_file.release_fd(); // llama_mmap takes ownership of fd
                return true;
            } catch (const std::runtime_error &e) {
                LLAMA_LOG_ERROR("%s: Failed to mmap cache file '%s': %s. Falling back to normal mmap.\n", __func__, cache_file_path.c_str(), e.what());
                return false;
            }
        } else {
            LLAMA_LOG_WARN("%s: Cache file '%s' is invalid. Removing and recreating.\n", __func__, cache_file_path.c_str());
            cache_file.close();
            fs::remove(cache_file_path); // Invalidate
            // Note: flock is released when fd is closed.
        }
    }

    cache_miss:
    // Placeholder for cache miss and creation logic
    }

    cache_miss:
    LLAMA_LOG_INFO("%s: Cache miss or invalid cache for '%s'. Creating new cache.\n", __func__, gguf_path.c_str());

    // 3. Cache Eviction (if max_cache_size > 0)
    if (max_cache_size > 0) {
        LLAMA_LOG_INFO("%s: Checking for cache eviction. Max size: %" PRIu64 " bytes.\n", __func__, (uint64_t)max_cache_size);
        size_t current_cache_size = 0;
        std::vector<fs::path> cache_files;

        for (const auto& entry : fs::directory_iterator(cache_dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".cache") {
                current_cache_size += entry.file_size();
                cache_files.push_back(entry.path());
            }
        }

        std::sort(cache_files.begin(), cache_files.end(), [] (const fs::path& a, const fs::path& b) {
            return fs::last_write_time(a) < fs::last_write_time(b);
        });

        while (current_cache_size > max_cache_size && !cache_files.empty()) {
            const fs::path& oldest_file = cache_files.front();
            size_t file_size = fs::file_size(oldest_file);
            try {
                fs::remove(oldest_file);
                current_cache_size -= file_size;
                LLAMA_LOG_INFO("%s: Evicted oldest cache file '%s' (size: %" PRIu64 " bytes). Current cache size: %" PRIu64 " bytes.\n", __func__, oldest_file.c_str(), (uint64_t)file_size, (uint64_t)current_cache_size);
            } catch (const fs::filesystem_error &e) {
                LLAMA_LOG_ERROR("%s: Failed to evict cache file '%s': %s\n", __func__, oldest_file.c_str(), e.what());
                // Continue trying to evict other files
            }
            cache_files.erase(cache_files.begin());
        }
    }

    const std::string tmp_cache_file_path = cache_file_path + ".tmp";

    // Acquire exclusive lock on a temporary file for creation
    {
        struct llama_file tmp_file(tmp_cache_file_path.c_str(), false); // Create/Write
        if (!tmp_file.is_open()) {
            LLAMA_LOG_ERROR("%s: Failed to create temporary cache file '%s'. Falling back to normal mmap.\n", __func__, tmp_cache_file_path.c_str());
            return false;
        }

#ifdef __linux__
        if (flock(tmp_file.fd, LOCK_EX | LOCK_NB) != 0) {
            if (errno == EWOULDBLOCK) {
                LLAMA_LOG_INFO("%s: Another process is already creating cache '%s'. Falling back to normal mmap.\n", __func__, cache_file_path.c_str());
            } else {
                LLAMA_LOG_ERROR("%s: Failed to acquire exclusive lock on temporary cache file '%s': %s. Falling back to normal mmap.\n", __func__, tmp_cache_file_path.c_str(), strerror(errno));
            }
            tmp_file.close();
            return false; // Fallback to normal mmap
        }
#endif

        // Open original GGUF file for reading data
        struct llama_file gguf_file_read(gguf_path.c_str(), true);
        if (!gguf_file_read.is_open()) {
            LLAMA_LOG_ERROR("%s: Failed to open GGUF file '%s' for reading data. Falling back to normal mmap.\n", __func__, gguf_path.c_str());
#ifdef __linux__
            flock(tmp_file.fd, LOCK_UN); // Release lock
#endif
            tmp_file.close();
            fs::remove(tmp_cache_file_path); // Clean up temp file
            return false;
        }

        // Calculate total size for allocation (header + model data)
        const uint64_t total_cache_size = sizeof(infty_cache_header) + gguf_model_size;

        // Allocate space in the temporary file
#ifdef __linux__
        if (posix_fallocate(tmp_file.fd, 0, total_cache_size) != 0) {
            LLAMA_LOG_ERROR("%s: Failed to fallocate '%s': %s. Falling back to normal mmap.\n", __func__, tmp_cache_file_path.c_str(), strerror(errno));
#ifdef __linux__
            flock(tmp_file.fd, LOCK_UN); // Release lock
#endif
            tmp_file.close();
            gguf_file_read.close();
            fs::remove(tmp_cache_file_path); // Clean up temp file
            return false;
        }
#else
        // Windows equivalent of fallocate (simplified)
        _lseek(tmp_file.fd, total_cache_size - 1, SEEK_SET);
        _write(tmp_file.fd, "", 1);
#endif

        // mmap the temporary file for writing
        llama_mmap *tmp_mapping = nullptr;
        try {
            // The llama_mmap constructor will handle MAP_LOCKED, MAP_POPULATE, HUGETLB.
            // Start mapping from offset 0, as we will write the header first.
            tmp_mapping = new llama_mmap(&tmp_file, mmap_flags.prefetch ? -1 : 0, mmap_flags.numa, mmap_flags.use_hugetlb, mmap_flags.mem_lock, 0, total_cache_size);
            tmp_file.release_fd(); // llama_mmap takes ownership of fd
        } catch (const std::runtime_error &e) {
            LLAMA_LOG_ERROR("%s: Failed to mmap temporary cache file '%s': %s. Falling back to normal mmap.\n", __func__, tmp_cache_file_path.c_str(), e.what());
#ifdef __linux__
            flock(tmp_file.fd, LOCK_UN); // Release lock
#endif
            tmp_file.close(); // Close the fd if mmap failed before ownership transfer
            gguf_file_read.close();
            fs::remove(tmp_cache_file_path); // Clean up temp file
            return false;
        }

        // Copy data from GGUF to cache, applying swizzle transform per tensor.
        // The GGUF file contains tensors in AoS layout; we must convert to SoA (or
        // intra-block) layout so GPU kernels can read them directly without runtime
        // transformation. Without this, rdna2-cache-swizzle kernels will read garbage.
        LLAMA_LOG_INFO("%s: Copying model data from '%s' to '%s' (with per-tensor swizzle).\n", __func__, gguf_path.c_str(), tmp_cache_file_path.c_str());
        gguf_file_read.seek(0, SEEK_SET);

        // Parse GGUF header. Format: [magic][version][n_tensors][tensor0_header...][tensor0_data...]
        // We do a single-pass parse: read all tensor headers first, then read data in order.
        uint32_t magic = 0;
        gguf_file_read.read_raw(&magic, sizeof(magic));
        const uint32_t gguf_version = (magic == 0x67677566) ? 3u : 1u;
        (void)gguf_version;

        uint32_t n_tensors = 0;
        if (gguf_version >= 3) {
            uint64_t val_u64 = 0;
            gguf_file_read.read_raw(&val_u64, sizeof(val_u64));
            n_tensors = (uint32_t)val_u64;
        } else {
            uint32_t val_u32 = 0;
            gguf_file_read.read_raw(&val_u32, sizeof(val_u32));
            n_tensors = val_u32;
        }

        // Store tensor headers in memory (no need to seek back for data).
        struct tensor_header {
            uint32_t name_len;
            char     name[4096];
            uint32_t n_dims;
            uint64_t ne[4];
            ggml_type type;
            uint64_t data_size; // precomputed
            // offset is implicit: sum of data_size of all previous tensors in file order.
        };

        std::vector<tensor_header> headers;
        headers.reserve(n_tensors);

        for (uint32_t t = 0; t < n_tensors && gguf_file_read.is_open(); ++t) {
            tensor_header hdr{};
            uint32_t name_len = 0;
            gguf_file_read.read_raw(&name_len, sizeof(name_len));
            if (name_len >= sizeof(hdr.name) || name_len > 4096 * 1024) {
                LLAMA_LOG_ERROR("%s: Invalid tensor name length %u\n", __func__, name_len);
                goto cleanup_and_fail;
            }
            gguf_file_read.read_raw(hdr.name, name_len);
            hdr.name_len = name_len;

            uint32_t n_dims = 0;
            gguf_file_read.read_raw(&n_dims, sizeof(n_dims));
            hdr.n_dims = n_dims;
            for (uint32_t d = 0; d < n_dims; ++d) {
                gguf_file_read.read_raw(&hdr.ne[d], sizeof(hdr.ne[d]));
            }

            uint32_t type_u = 0;
            gguf_file_read.read_raw(&type_u, sizeof(type_u));
            hdr.type = (ggml_type)type_u;

            // Compute data size from tensor shape and type (mirrors ggml_nbytes)
            uint64_t sz = 1;
            for (uint32_t d = 0; d < n_dims; ++d) {
                if (sz > gguf_model_size / hdr.ne[d]) {
                    LLAMA_LOG_ERROR("%s: Tensor overflow\n", __func__);
                    goto cleanup_and_fail;
                }
                sz *= hdr.ne[d];
            }
            hdr.data_size = ggml_nbytes((ggml_type)type_u);

            headers.push_back(std::move(hdr));
        }

        // Now read each tensor's data in file order, apply swizzle, write to cache.
        char* dest_ptr = (char*)tmp_mapping->addr + sizeof(infty_cache_header);
        uint64_t total_written = 0;

        for (const auto &hdr : headers) {
            // Read raw tensor data from GGUF (data follows headers in the file).
            std::vector<uint8_t> data(hdr.data_size);
            gguf_file_read.read_raw(data.data(), hdr.data_size);

            // Apply swizzle if this tensor type is configured for it.
            if (tensor_needs_swizzle(hdr.type, current_compile_flags)) {
                apply_swizzle(hdr.type, data.data(), data.size());
            }

            std::memcpy(dest_ptr + total_written, data.data(), data.size());
            total_written += data.size();
        }

        // Verify all model data was written.
        if (total_written != gguf_model_size) {
            LLAMA_LOG_ERROR("%s: Tensor data size mismatch: expected %lu, got %lu.\n",
                __func__, (unsigned long)gguf_model_size, (unsigned long)total_written);
            goto cleanup_and_fail;
        }

        // Zero-fill remaining space (padding between last tensor and header).
        memset(dest_ptr + total_written, 0, gguf_model_size - total_written);

        // Write header to the beginning of the mmap'd temporary file
        infty_cache_header new_header = {
            .magic = LLAMA_INFINITY_CACHE_MAGIC,
            .compile_flags = current_compile_flags,
            .model_data_size = gguf_model_size,
        };
        std::memcpy(new_header.gguf_hash, model_hash.data(), 32);
        std::memcpy(tmp_mapping->addr, &new_header, sizeof(infty_cache_header));

        // Flush changes to disk
#ifdef __linux__
        if (msync(tmp_mapping->addr, total_cache_size, MS_SYNC) != 0) {
            LLAMA_LOG_ERROR("%s: Failed to msync cache file '%s': %s. Data might not be fully flushed.\n", __func__, tmp_cache_file_path.c_str(), strerror(errno));
        }
#endif

        // Rename .tmp to .cache
        try {
            fs::rename(tmp_cache_file_path, cache_file_path);
            LLAMA_LOG_INFO("%s: Successfully created Infinity Cache file '%s'.\n", __func__, cache_file_path.c_str());
        } catch (const fs::filesystem_error &e) {
            LLAMA_LOG_ERROR("%s: Failed to rename temporary cache file '%s' to '%s': %s. Falling back to normal mmap.\n", __func__, tmp_cache_file_path.c_str(), cache_file_path.c_str(), e.what());
            goto cleanup_and_fail;
        }

#ifdef __linux__
        // Release lock (now that rename is done, the new file has no lock if not reopened)
        // The lock was on tmp_file.fd, which is now closed by llama_mmap owning it. Renaming takes care of it.
#endif
        gguf_file_read.close();
        *out_mapping = tmp_mapping; // Transfer ownership
        return true;

    cleanup_and_fail:
#ifdef __linux__
        if (tmp_file.is_open()) {
            flock(tmp_file.fd, LOCK_UN); // Release lock if still held
        }
#endif
        if (tmp_mapping) {
            delete tmp_mapping; // This will unmap and close fd
            tmp_mapping = nullptr;
        }
        if (fs::exists(tmp_cache_file_path)) {
            fs::remove(tmp_cache_file_path); // Ensure temp file is removed
        }
        gguf_file_read.close();
        return false;
    return false; // Temporary fallback
}
