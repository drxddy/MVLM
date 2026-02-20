#include "../engine/device.h"
#include "../models/moondream2.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

static void print_usage(const char* program) {
    printf("MGPU - On-Device Vision-Language Model Inference Engine\n\n");
    printf("Usage: %s [options]\n\n", program);
    printf("Options:\n");
    printf("  --model <path>      Path to GGUF model file\n");
    printf("  --prompt <text>     Text prompt for the model\n");
    printf("  --image <path>      Path to input image\n");
    printf("  --kernels <dir>     Path to OpenCL kernel directory\n");
    printf("  --vocab <path>      Path to tokenizer vocabulary file\n");
    printf("  --max-tokens <n>    Maximum tokens to generate (default: 128)\n");
    printf("  --benchmark         Run benchmark mode\n");
    printf("  --help              Show this help message\n");
    printf("\nExamples:\n");
    printf("  %s --benchmark\n", program);
    printf("  %s --model weights/moondream2.gguf --kernels src/kernels --prompt \"Hello\"\n", program);
    printf("  %s --model weights/moondream2.gguf --kernels src/kernels --vocab vocab.txt --prompt \"Describe this image\"\n", program);
}

int main(int argc, char** argv) {
    const char* model_path = nullptr;
    const char* prompt = nullptr;
    const char* image_path = nullptr;
    const char* kernel_dir = nullptr;
    const char* vocab_path = nullptr;
    int max_tokens = 128;
    bool benchmark = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "--image") == 0 && i + 1 < argc) {
            image_path = argv[++i];
        } else if (strcmp(argv[i], "--kernels") == 0 && i + 1 < argc) {
            kernel_dir = argv[++i];
        } else if (strcmp(argv[i], "--vocab") == 0 && i + 1 < argc) {
            vocab_path = argv[++i];
        } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--benchmark") == 0) {
            benchmark = true;
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Error: Unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (argc == 1) {
        print_usage(argv[0]);
        return 0;
    }

    // Initialize OpenCL device
    mgpu::DeviceInfo device;
    if (!mgpu::init_device(&device)) {
        fprintf(stderr, "Error: Failed to initialize OpenCL device\n");
        return 1;
    }
    mgpu::print_device_info(&device);

    if (benchmark) {
        printf("\n=== Benchmark Mode ===\n");
        printf("Device initialized successfully.\n");
        printf("Run mgpu_bench for GEMM benchmarks.\n");
        mgpu::destroy_device(&device);
        return 0;
    }

    if (model_path) {
        mgpu::Moondream2Model model;
        if (!mgpu::moondream2_load(&model, &device, model_path, kernel_dir)) {
            fprintf(stderr, "Error: Failed to load model: %s\n", model_path);
            mgpu::destroy_device(&device);
            return 1;
        }

        if (prompt) {
            int n = mgpu::moondream2_generate(&model, &device, prompt,
                                               max_tokens, vocab_path);
            if (n < 0) {
                fprintf(stderr, "Error: generation failed\n");
            }
        } else if (image_path) {
            printf("Image: %s\n", image_path);
            printf("(Vision pipeline not yet wired — provide --prompt for text-only generation)\n");
        } else {
            printf("Model loaded successfully. Use --prompt to generate text.\n");
        }

        mgpu::moondream2_destroy(&model);
    }

    mgpu::destroy_device(&device);
    return 0;
}
