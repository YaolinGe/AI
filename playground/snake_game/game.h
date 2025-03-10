#ifndef SNAKE_GAME_H
#define SNAKE_GAME_H

#include <iostream>
#include <vector>
#include <ctime>    // For time functions
#include <cstdlib>  // For rand() and srand()
#include <thread>   // For sleep
#include <chrono>   // For time duration
#include <termios.h> // For terminal settings on Unix-like systems
#include <unistd.h> // For POSIX API
#include <fcntl.h>  // For file control options
#include <sys/ioctl.h> // For ioctl function

// Direction enumeration
enum Direction {
    STOP = 0,
    LEFT,
    RIGHT,
    UP,
    DOWN
};

// Position structure
struct Position {
    int x;
    int y;
};

class SnakeGame {
private:
    // Game area dimensions
    int width;
    int height;
    
    // Game state
    bool gameOver;
    
    // Snake properties
    std::vector<Position> snake;
    Direction direction;
    
    // Food position
    Position food;
    
    // Score
    int score;
    
    // Terminal settings
    struct termios oldSettings;
    
    // Methods
    void setup();
    void draw();
    void input();
    void logic();
    void generateFood();
    
    // Helper methods for cross-platform compatibility
    void clearScreen();
    bool kbhit();
    char getch();
    void sleepMs(int milliseconds);
    void setupTerminal();
    void restoreTerminal();
    
public:
    // Constructor and destructor
    SnakeGame();
    ~SnakeGame();
    
    // Main game loop
    void run();
};

#endif // SNAKE_GAME_H 