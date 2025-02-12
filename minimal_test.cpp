#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include "GpuANSCodec.h"
#include "StackDeviceMemory.h"

using namespace dietgpu;

std::vector<uint8_t> readFileWithOffset(const std::string& filename, size_t offset, size_t size) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    file.seekg(offset);
    std::vector<uint8_t> data(size);
    file.read(reinterpret_cast<char*>(data.data()), size);
    
    return data;
}

void runBenchmark(const std::string& filename, size_t offset, size_t dataSize) {
    auto inputData = readFileWithOffset(filename, offset, dataSize);
    auto res = makeStackMemory();
    auto stream = CudaStream::makeNonBlocking();

    auto startUpload = std::chrono::high_resolution_clock::now();
    auto data_dev = res.copyAlloc(stream, inputData);
    cudaStreamSynchronize(stream);
    auto endUpload = std::chrono::high_resolution_clock::now();

    uint32_t maxCompSize = getMaxCompressedSize(dataSize);
    auto compressed_dev = res.alloc<uint8_t>(stream, maxCompSize);
    auto compressedSize_dev = res.alloc<uint32_t>(stream, 1);

    auto startComp = std::chrono::high_resolution_clock::now();
    auto ansCodecConfig = 10;  // 9, 10, 11
    ansEncodeBatchStride(
        res,
        ANSCodecConfig(ansCodecConfig),
        1,
        data_dev.data(),
        dataSize,
        dataSize,
        nullptr,
        compressed_dev.data(),
        maxCompSize,
        compressedSize_dev.data(),
        stream);
    cudaStreamSynchronize(stream);
    auto endComp = std::chrono::high_resolution_clock::now();

    auto compressedSize = compressedSize_dev.copyToHost(stream)[0];
    auto decompressed_dev = res.alloc<uint8_t>(stream, dataSize);
    auto success_dev = res.alloc<uint8_t>(stream, 1);
    auto outSize_dev = res.alloc<uint32_t>(stream, 1);
    
    auto startDecomp = std::chrono::high_resolution_clock::now();
    ansDecodeBatchStride(
        res,
        ANSCodecConfig(ansCodecConfig),
        1,
        compressed_dev.data(),
        maxCompSize,
        decompressed_dev.data(),
        dataSize,
        dataSize,
        success_dev.data(),
        outSize_dev.data(),
        stream);
    cudaStreamSynchronize(stream);
    auto endDecomp = std::chrono::high_resolution_clock::now();

    auto startDownload = std::chrono::high_resolution_clock::now();
    auto decompressed = decompressed_dev.copyToHost(stream);
    cudaStreamSynchronize(stream);
    auto endDownload = std::chrono::high_resolution_clock::now();

    auto success = success_dev.copyToHost(stream)[0];

    // Calculate durations in seconds
    double uploadTime = std::chrono::duration<double>(endUpload - startUpload).count();
    double compTime = std::chrono::duration<double>(endComp - startComp).count();
    double decompTime = std::chrono::duration<double>(endDecomp - startDecomp).count();
    double downloadTime = std::chrono::duration<double>(endDownload - startDownload).count();

    // Calculate throughput (GB/s)
    double uploadThroughput = (dataSize / (1024.0 * 1024.0 * 1024.0)) / uploadTime;
    double compThroughput = (dataSize / (1024.0 * 1024.0 * 1024.0)) / compTime;
    double decompThroughput = (dataSize / (1024.0 * 1024.0 * 1024.0)) / decompTime;
    double downloadThroughput = (dataSize / (1024.0 * 1024.0 * 1024.0)) / downloadTime;

    std::cout << "\nBenchmark for " << (dataSize / (1024 * 1024)) << "MB (offset: " << offset << "):\n"
              << "Upload: " << uploadTime * 1000 << " ms (" << uploadThroughput << " GB/s)\n"
              << "Compression: " << compTime * 1000 << " ms (" << compThroughput << " GB/s)\n"
              << "Decompression: " << decompTime * 1000 << " ms (" << decompThroughput << " GB/s)\n"
              << "Download: " << downloadTime * 1000 << " ms (" << downloadThroughput << " GB/s)\n"
              << "Compression ratio: " << (float)compressedSize / dataSize << "\n"
              << "Verification: " << (inputData == decompressed ? "Pass" : "Fail") << "\n";
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>\n";
        return 1;
    }

    try {
        cudaSetDevice(0);
        const std::string filename = argv[1];
        const size_t offset = 200ULL * 1024 * 1024; // 200MB offset
        
        const std::vector<size_t> sizes = {
            1ULL * 1024 * 1024,    // 1MB
            10ULL * 1024 * 1024,   // 10MB
            100ULL * 1024 * 1024,   // 100MB
        };

        for (const auto& size : sizes) {
            runBenchmark(filename, offset, size);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
