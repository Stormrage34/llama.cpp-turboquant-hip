// SPDX-FileCopyrightText: 2025 llama.cpp authors
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdio>
#include <dlfcn.h>
#include <string>

//
// ROCTx — optional runtime AMD ROCm profiling markers.
//
// Provides RAII range markers (push/pop) and pause/resume guards that
// rocprofv3's --selected-regions flag recognises.
//
// All calls are no-ops when the libroctx64.so library is absent or a
// required symbol is missing from the loaded version.  No compile-time
// dependency on ROCTx headers is needed — symbols are resolved via
// dlopen/dlsym at first use.
//
// Usage:
//   {
//       roctx_marker _("attention_layer_5");   // push on construction
//       compute_attention();                    // ← GPU kernels run here
//   }                                           // pop on destruction
//
// Shell (with rocprofv3):
//   rocprofv3 --hip-trace --stats --selected-regions -d ./telemetry_output -- \
//       ./build/bin/llama-bench -m model.gguf -n 1000 -p 512
//
// ROCTx library versions:
//   ROCTX 4.1 (ROCm 5.x):  roctxRangePushA, roctxRangePop
//   ROCTX 4.2+ (ROCm 6.x): roctxProfilerPause, roctxProfilerResume
//

namespace roctx {

// ---------------------------------------------------------------------------
// Internal: lazy library + symbol resolution
// ---------------------------------------------------------------------------
namespace detail {

// Resolve a single ROCTx function pointer by name using RTLD_DEFAULT.
// This searches the global symbol table, so rocprofv3's LD_PRELOAD
// interception (librocprofiler-sdk-roctx.so) takes priority over the
// old libroctx64.so.  All calls are no-ops when the symbol is absent.
template <typename Fn>
Fn sym(const char * name) {
    union { void * ptr; Fn fn; } u;
    u.ptr = dlsym(RTLD_DEFAULT, name);
    if (!u.ptr) {
        // Only warn once (first call) — subsequent lookups are cached
        // via the static local in each fn_* getter below.
        static bool warned = false;
        if (!warned) {
            fprintf(stderr, "[roctx] symbols not found: install ROCm or "
                            "check LD_PRELOAD for librocprofiler-sdk-roctx.so\n");
            warned = true;
        }
    }
    return u.fn;
}

// function pointer typedefs
using push_fn  = void (*)(const char *);
using pop_fn   = void (*)();
using ctl_fn   = void (*)(int);

inline push_fn  fn_push()  { static auto p = sym<push_fn>("roctxRangePushA");    return p; }
inline pop_fn   fn_pop()   { static auto p = sym<pop_fn >("roctxRangePop");       return p; }
inline ctl_fn   fn_pause() { static auto p = sym<ctl_fn >("roctxProfilerPause");  return p; }
inline ctl_fn   fn_resume(){ static auto p = sym<ctl_fn >("roctxProfilerResume"); return p; }

} // namespace detail

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Push a named range onto the rocprofv3 timeline.
inline void range_push(const char * name) {
    auto fn = detail::fn_push();
    if (fn) {
        fn(name);
    }
}

/// Pop the current range from the timeline.
inline void range_pop() {
    auto fn = detail::fn_pop();
    if (fn) {
        fn();
    }
}

/// Pause profiling data collection (requires ROCTX 4.2+).
inline void pause() {
    auto fn = detail::fn_pause();
    if (fn) {
        fn(0); // 0 = all threads
    }
}

/// Resume profiling data collection (requires ROCTX 4.2+).
inline void resume() {
    auto fn = detail::fn_resume();
    if (fn) {
        fn(0);
    }
}

// ---------------------------------------------------------------------------
// RAII marker
// ---------------------------------------------------------------------------
class marker {
public:
    explicit marker(const char * name) {
        range_push(name);
    }
    explicit marker(const std::string & name) {
        range_push(name.c_str());
    }
    ~marker() {
        range_pop();
    }
    // non-copyable, non-movable
    marker(const marker &) = delete;
    marker & operator=(const marker &) = delete;
};

// ---------------------------------------------------------------------------
// Scoped pause/resume guard
// ---------------------------------------------------------------------------
class pause_guard {
public:
    pause_guard()  { pause(); }
    ~pause_guard() { resume(); }
    pause_guard(const pause_guard &) = delete;
    pause_guard & operator=(const pause_guard &) = delete;
};

} // namespace roctx
