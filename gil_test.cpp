#include <iostream>
#include <thread>
#include <chrono>

// 空循环20亿次，使用volatile防止编译器对循环进行优化
void work() {
    volatile long long sum = 0;
    for (long long i = 0; i < 2000000000L; ++i) {
        sum += 1;
    }
}

int main() {
    using namespace std::chrono;

    // 串行跑两次
    auto start = high_resolution_clock::now();
    work(); work();
    auto end = high_resolution_clock::now();
    std::cout << "Serial Time (C++): "
              << duration_cast<milliseconds>(end - start).count() << "ms" << std::endl;

    // 并行跑两次
    start = high_resolution_clock::now();
    std::thread t1(work);
    std::thread t2(work);
    t1.join(); t2.join();
    end = high_resolution_clock::now();
    std::cout << "Parallel Time (C++): "
              << duration_cast<milliseconds>(end - start).count() << "ms" << std::endl;

    return 0;
}



