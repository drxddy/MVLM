#pragma once

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

namespace mgpu {

struct KernelTimer {
    cl_event event;
    const char* name;
};

cl_event profile_begin();
double profile_end(cl_event event);   // returns milliseconds
void print_profile(const char* name, cl_event event);

struct ProfileSession {
    static const int MAX_ENTRIES = 256;
    struct Entry {
        char name[64];
        double time_ms;
    };
    Entry entries[MAX_ENTRIES];
    int count;
};

void profile_session_init(ProfileSession* session);
void profile_session_add(ProfileSession* session, const char* name, cl_event event);
void profile_session_print(const ProfileSession* session);

} // namespace mgpu
