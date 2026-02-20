#include "profiler.h"

#include <cstdio>
#include <cstring>

#ifdef MGPU_ANDROID
#include <android/log.h>
#define MGPU_LOG(...) __android_log_print(ANDROID_LOG_INFO, "MGPU", __VA_ARGS__)
#define MGPU_ERR(...) __android_log_print(ANDROID_LOG_ERROR, "MGPU", __VA_ARGS__)
#else
#define MGPU_LOG(...) fprintf(stdout, __VA_ARGS__)
#define MGPU_ERR(...) fprintf(stderr, __VA_ARGS__)
#endif

namespace mgpu {

cl_event profile_begin() {
    // Return a user event placeholder — the actual event is obtained
    // by passing &event to clEnqueueNDRangeKernel.
    // Caller should pass the returned event pointer to the enqueue call.
    // This function exists for symmetry; the real event comes from OpenCL.
    return nullptr;
}

double profile_end(cl_event event) {
    if (!event) return -1.0;

    // Wait for the event to complete
    cl_int err = clWaitForEvents(1, &event);
    if (err != CL_SUCCESS) {
        MGPU_ERR("clWaitForEvents failed (err=%d)\n", err);
        return -1.0;
    }

    cl_ulong start = 0, end = 0;

    err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                                 sizeof(cl_ulong), &start, nullptr);
    if (err != CL_SUCCESS) {
        MGPU_ERR("Failed to get profiling start time (err=%d)\n", err);
        return -1.0;
    }

    err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                                 sizeof(cl_ulong), &end, nullptr);
    if (err != CL_SUCCESS) {
        MGPU_ERR("Failed to get profiling end time (err=%d)\n", err);
        return -1.0;
    }

    // Convert nanoseconds to milliseconds
    return (double)(end - start) / 1.0e6;
}

void print_profile(const char* name, cl_event event) {
    double ms = profile_end(event);
    if (ms >= 0.0) {
        MGPU_LOG("  %-40s %8.3f ms\n", name, ms);
    } else {
        MGPU_LOG("  %-40s   FAILED\n", name);
    }
}

void profile_session_init(ProfileSession* session) {
    session->count = 0;
}

void profile_session_add(ProfileSession* session, const char* name, cl_event event) {
    if (session->count >= ProfileSession::MAX_ENTRIES) {
        MGPU_ERR("Profile session full (max %d entries)\n", ProfileSession::MAX_ENTRIES);
        return;
    }

    double ms = profile_end(event);
    ProfileSession::Entry* entry = &session->entries[session->count];

    // Copy name safely
    size_t len = strlen(name);
    if (len >= sizeof(entry->name)) len = sizeof(entry->name) - 1;
    memcpy(entry->name, name, len);
    entry->name[len] = '\0';

    entry->time_ms = ms;
    session->count++;
}

void profile_session_print(const ProfileSession* session) {
    if (session->count == 0) {
        MGPU_LOG("Profile session: no entries\n");
        return;
    }

    double total = 0.0;
    for (int i = 0; i < session->count; i++) {
        if (session->entries[i].time_ms > 0.0)
            total += session->entries[i].time_ms;
    }

    MGPU_LOG("╔═══════════════════════════════════════════════════════════════╗\n");
    MGPU_LOG("║                    MGPU Profile Session                      ║\n");
    MGPU_LOG("╠════════════════════════════════════╤════════════╤════════════╣\n");
    MGPU_LOG("║ Kernel                             │   Time(ms) │        %%   ║\n");
    MGPU_LOG("╠════════════════════════════════════╪════════════╪════════════╣\n");

    for (int i = 0; i < session->count; i++) {
        double pct = (total > 0.0 && session->entries[i].time_ms > 0.0)
                     ? (session->entries[i].time_ms / total * 100.0) : 0.0;
        MGPU_LOG("║ %-34s │ %10.3f │ %8.1f%%  ║\n",
                 session->entries[i].name, session->entries[i].time_ms, pct);
    }

    MGPU_LOG("╠════════════════════════════════════╪════════════╪════════════╣\n");
    MGPU_LOG("║ TOTAL                              │ %10.3f │   100.0%%  ║\n", total);
    MGPU_LOG("╚════════════════════════════════════╧════════════╧════════════╝\n");
}

} // namespace mgpu
