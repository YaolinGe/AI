#include<iostream>
#include<array>
#include<new>


struct Data {
    std::array<char, 1'000'000u> memoryBlock; 
}; 

void bar() {
    const Data* dataPtr = new Data; 
    std::cout << "Allocated data at " << dataPtr << std::endl; 
}

void foo() {
    for(auto i = 0u; i < 1'000'000u; ++i) {
        std::cout << "i = " << i << std::endl; 
        bar(); 
    }
}

int main() {
    foo(); 
    return 0; 
}