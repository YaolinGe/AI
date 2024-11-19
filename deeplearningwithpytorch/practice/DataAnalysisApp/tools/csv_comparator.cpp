#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>

std::mutex mtx;

bool compareChunks(const std::string& filePath1, const std::string& filePath2, std::streampos start, std::streampos end) {
    std::ifstream file1(filePath1);
    std::ifstream file2(filePath2);

    if (!file1.is_open() || !file2.is_open()) {
        std::cerr << "Error opening files." << std::endl;
        return false;
    }

    file1.seekg(start);
    file2.seekg(start);

    std::string line1, line2;
    while (file1.tellg() < end && std::getline(file1, line1) && std::getline(file2, line2)) {
        if (line1 != line2) {
            return false;
        }
    }

    return true;
}

void compareCSVFiles(const std::string& filePath1, const std::string& filePath2, int numThreads, bool& result) {
    std::ifstream file1(filePath1, std::ifstream::ate | std::ifstream::binary);
    std::ifstream file2(filePath2, std::ifstream::ate | std::ifstream::binary);

    if (!file1.is_open() || !file2.is_open()) {
        std::cerr << "Error opening files." << std::endl;
        result = false;
        return;
    }

    std::streampos fileSize1 = file1.tellg();
    std::streampos fileSize2 = file2.tellg();

    if (fileSize1 != fileSize2) {
        result = false;
        return;
    }

    std::streampos chunkSize = fileSize1 / numThreads;
    std::vector<std::thread> threads;
    result = true;

    for (int i = 0; i < numThreads; ++i) {
        std::streampos start = i * chunkSize;
        std::streampos end = (i == numThreads - 1) ? fileSize1 : static_cast<std::streampos>((i + 1) * chunkSize);

        threads.emplace_back([&filePath1, &filePath2, start, end, &result]() {
            if (!compareChunks(filePath1, filePath2, start, end)) {
                std::lock_guard<std::mutex> lock(mtx);
                result = false;
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <file1.csv> <file2.csv> <numThreads>" << std::endl;
        return -1;
    }

    std::string filePath1 = argv[1];
    std::string filePath2 = argv[2];
    int numThreads = std::stoi(argv[3]);

    bool result;
    compareCSVFiles(filePath1, filePath2, numThreads, result);

    return result ? 0 : -1;
}