#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include "GpuFloatCodec.h"
#include "StackDeviceMemory.h"

using namespace dietgpu;

std::vector<uint16_t> readFileWithOffset(const std::string& filename, size_t offset, size_t size) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    file.seekg(offset);
    std::vector<uint16_t> data(size / sizeof(uint16_t));  // size should be in bytes
    file.read(reinterpret_cast<char*>(data.data()), size);

    return data;
}

void runFloat16CompressionTest(const std::string& filename, size_t offset, size_t dataSize) {
    try {
        // Read data from file
        auto inputData = readFileWithOffset(filename, offset, dataSize);
        auto numElements = inputData.size();

        cudaSetDevice(0);
        auto res = makeStackMemory();
        auto stream = CudaStream::makeNonBlocking();

        // Upload data to GPU
        auto startUpload = std::chrono::high_resolution_clock::now();
        auto data_dev = res.copyAlloc(stream, inputData);
        cudaStreamSynchronize(stream);
        auto endUpload = std::chrono::high_resolution_clock::now();

        // Prepare for compression
        uint32_t maxCompSize = getMaxFloatCompressedSize(FloatType::kFloat16, numElements);
        auto compressed_dev = res.alloc<uint8_t>(stream, maxCompSize);
        auto compressedSize_dev = res.alloc<uint32_t>(stream, 1);

        // Set up compression configuration
        auto floatConfig = FloatCompressConfig(
            FloatType::kFloat16,   // Using Float16
            ANSCodecConfig(10),    // prob bits
            false,                 // 16-byte alignment
            false                  // checksum
        );

        // Compress
        auto startComp = std::chrono::high_resolution_clock::now();
        
        const void* inPtr = data_dev.data();
        void* outPtr = compressed_dev.data();
        
        floatCompress(
            res,
            floatConfig,
            1,  // batch size
            &inPtr,
            (const uint32_t*)&numElements,
            &outPtr,
            compressedSize_dev.data(),
            stream
        );
        
        cudaStreamSynchronize(stream);
        auto endComp = std::chrono::high_resolution_clock::now();

        // Get compressed size
        auto compressedSize = compressedSize_dev.copyToHost(stream)[0];

        // Prepare for decompression
        auto decompressed_dev = res.alloc<uint16_t>(stream, numElements);
        auto success_dev = res.alloc<uint8_t>(stream, 1);
        auto outSize_dev = res.alloc<uint32_t>(stream, 1);

        // Set up decompression configuration
        auto decompConfig = FloatDecompressConfig(
            FloatType::kFloat16,  // Using Float16
            ANSCodecConfig(10),
            false,  // 16-byte alignment
            false   // checksum
        );

        // Decompress
        auto startDecomp = std::chrono::high_resolution_clock::now();
        
        const void* compPtr = compressed_dev.data();
        void* decompPtr = decompressed_dev.data();
        
        floatDecompress(
            res,
            decompConfig,
            1,  // batch size
            &compPtr,
            &decompPtr,
            (const uint32_t*)&numElements,
            success_dev.data(),
            outSize_dev.data(),
            stream
        );
        
        cudaStreamSynchronize(stream);
        auto endDecomp = std::chrono::high_resolution_clock::now();

        // Download results
        auto startDownload = std::chrono::high_resolution_clock::now();
        auto decompressed = decompressed_dev.copyToHost(stream);
        auto success = success_dev.copyToHost(stream)[0];
        cudaStreamSynchronize(stream);
        auto endDownload = std::chrono::high_resolution_clock::now();

        // Calculate durations and throughput
        double uploadTime = std::chrono::duration<double>(endUpload - startUpload).count();
        double compTime = std::chrono::duration<double>(endComp - startComp).count();
        double decompTime = std::chrono::duration<double>(endDecomp - startDecomp).count();
        double downloadTime = std::chrono::duration<double>(endDownload - startDownload).count();

        double uploadThroughput = (dataSize / (1024.0 * 1024.0 * 1024.0)) / uploadTime;
        double compThroughput = (dataSize / (1024.0 * 1024.0 * 1024.0)) / compTime;
        double decompThroughput = (dataSize / (1024.0 * 1024.0 * 1024.0)) / decompTime;
        double downloadThroughput = (dataSize / (1024.0 * 1024.0 * 1024.0)) / downloadTime;

        // Verify results
        bool verified = (inputData == decompressed);

        // Print results
        std::cout << "\nFloat16 Compression Benchmark for " << (dataSize / (1024 * 1024)) << "MB (offset: " << offset << "):\n"
                  << "Upload: " << uploadTime * 1000 << " ms (" << uploadThroughput << " GB/s)\n"
                  << "Compression: " << compTime * 1000 << " ms (" << compThroughput << " GB/s)\n"
                  << "Decompression: " << decompTime * 1000 << " ms (" << decompThroughput << " GB/s)\n"
                  << "Download: " << downloadTime * 1000 << " ms (" << downloadThroughput << " GB/s)\n"
                  << "Compression ratio: " << (float)compressedSize / dataSize << "\n"
                  << "Success: " << (success ? "Yes" : "No") << "\n"
                  << "Verification: " << (verified ? "Pass" : "Fail") << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>\n";
        return 1;
    }

    try {
        cudaSetDevice(0);
        const std::string filename = argv[1];
        const size_t offset = 100ULL * 1024 * 1024; // 100MB offset

        const std::vector<size_t> sizes = {
            1ULL * 1024 * 1024,    // 1MB
            10ULL * 1024 * 1024,   // 10MB
            100ULL * 1024 * 1024,  // 100MB
        };

        for (const auto& size : sizes) {
            runFloat16CompressionTest(filename, offset, size);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
