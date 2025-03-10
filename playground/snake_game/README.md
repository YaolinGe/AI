# Snake Game

A classic Snake game implemented in C++ for terminal/console.

## Description

This is a simple implementation of the classic Snake game that runs in the terminal. The player controls a snake that moves around the screen, eating food to grow longer. The game ends when the snake collides with the walls or itself.

## Features

- Terminal-based graphics
- Keyboard controls
- Score tracking
- Cross-platform compatibility (macOS, Linux)

## Controls

- `w`: Move Up
- `a`: Move Left
- `s`: Move Down
- `d`: Move Right
- `x`: Quit Game

## Requirements

- C++ compiler with C++11 support
- Terminal that supports ANSI escape codes

## How to Build and Run

### Using Make

```bash
# Navigate to the snake_game directory
cd playground/snake_game

# Build the game
make

# Run the game
make run
```

### Manual Compilation

```bash
# Navigate to the snake_game directory
cd playground/snake_game

# Compile the game
g++ -std=c++11 -Wall -Wextra -o snake main.cpp game.cpp

# Run the game
./snake
```

## Game Rules

1. Control the snake to eat the food (marked as 'F').
2. Each time the snake eats food, it grows longer and your score increases.
3. The game ends if the snake hits the wall or itself.
4. Try to achieve the highest score possible!

## Implementation Details

The game is implemented using the following components:

- `main.cpp`: Entry point of the program
- `game.h`: Header file with class and structure definitions
- `game.cpp`: Implementation of the game logic

The game uses non-blocking terminal input to capture keystrokes without requiring the Enter key to be pressed. 