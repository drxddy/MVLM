#include <jni.h>
#include <string>
#include <cstring>
#include <android/log.h>

#include "models/moondream2.h"
#include "engine/device.h"

#define LOG_TAG "MGPU"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

using namespace mgpu;

static Moondream2Model* g_model = nullptr;
static DeviceInfo* g_device = nullptr;

extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_mgpu_MainActivity_loadModel(JNIEnv* env, jobject thiz,
                                      jstring model_path, jstring kernel_dir) {
    const char* model_path_str = env->GetStringUTFChars(model_path, nullptr);
    const char* kernel_dir_str = env->GetStringUTFChars(kernel_dir, nullptr);

    LOGI("Loading model from: %s", model_path_str);

    // Initialize device
    g_device = new DeviceInfo();
    if (!init_device(g_device)) {
        LOGE("Failed to initialize OpenCL device");
        env->ReleaseStringUTFChars(model_path, model_path_str);
        env->ReleaseStringUTFChars(kernel_dir, kernel_dir_str);
        return JNI_FALSE;
    }

    // Load model
    g_model = new Moondream2Model();
    if (!moondream2_load(g_model, g_device, model_path_str, kernel_dir_str)) {
        LOGE("Failed to load model");
        destroy_device(g_device);
        delete g_device;
        g_device = nullptr;
        delete g_model;
        g_model = nullptr;
        env->ReleaseStringUTFChars(model_path, model_path_str);
        env->ReleaseStringUTFChars(kernel_dir, kernel_dir_str);
        return JNI_FALSE;
    }

    // Upload weights
    if (!moondream2_upload_weights(g_model, g_device)) {
        LOGE("Failed to upload weights");
    }

    // Initialize RoPE
    if (!moondream2_init_rope(g_model, g_device)) {
        LOGE("Failed to initialize RoPE");
    }

    // Allocate buffers
    if (!moondream2_alloc_buffers(g_model, g_device)) {
        LOGE("Failed to allocate buffers");
    }

    LOGI("Model loaded successfully!");
    env->ReleaseStringUTFChars(model_path, model_path_str);
    env->ReleaseStringUTFChars(kernel_dir, kernel_dir_str);
    return JNI_TRUE;
}

JNIEXPORT jstring JNICALL
Java_com_mgpu_MainActivity_generateText(JNIEnv* env, jobject thiz,
                                         jstring prompt, jint max_tokens) {
    if (!g_model || !g_device) {
        return env->NewStringUTF("Error: Model not loaded");
    }

    const char* prompt_str = env->GetStringUTFChars(prompt, nullptr);

    LOGI("Generating text for prompt: %s", prompt_str);

    int result_tokens[512];
    int num_tokens = moondream2_generate(g_model, g_device, prompt_str,
                                          max_tokens, nullptr);

    // Decode tokens to string
    // For now, just return a placeholder
    std::string result = "Generated " + std::to_string(num_tokens) + " tokens";

    env->ReleaseStringUTFChars(prompt, prompt_str);
    return env->NewStringUTF(result.c_str());
}

JNIEXPORT jstring JNICALL
Java_com_mgpu_MainActivity_getDeviceInfo(JNIEnv* env, jobject thiz) {
    if (!g_device) {
        return env->NewStringUTF("Device not initialized");
    }

    std::string info = "OpenCL Device: ";
    info += g_device->device_name;
    info += "\nVendor: ";
    info += g_device->vendor;
    info += "\nCompute Units: ";
    info += std::to_string(g_device->compute_units);

    return env->NewStringUTF(info.c_str());
}

JNIEXPORT void JNICALL
Java_com_mgpu_MainActivity_unloadModel(JNIEnv* env, jobject thiz) {
    if (g_model) {
        moondream2_destroy(g_model);
        delete g_model;
        g_model = nullptr;
    }

    if (g_device) {
        destroy_device(g_device);
        delete g_device;
        g_device = nullptr;
    }

    LOGI("Model unloaded");
}

} // extern "C"
