#include "game.h"

// Constructor
SnakeGame::SnakeGame() {
    // Initialize game area dimensions
    width = 20;
    height = 20;
    
    // Initialize game state
    gameOver = false;
    
    // Initialize score
    score = 0;
    
    // Setup terminal for non-blocking input
    setupTerminal();
    
    // Call setup to initialize other game elements
    setup();
}

// Destructor
SnakeGame::~SnakeGame() {
    // Restore terminal settings
    restoreTerminal();
}

// Setup the game
void SnakeGame::setup() {
    // Set initial direction
    direction = STOP;
    
    // Set initial snake position (middle of the screen)
    Position head;
    head.x = width / 2;
    head.y = height / 2;
    snake.clear();
    snake.push_back(head);
    
    // Generate initial food
    generateFood();
    
    // Seed random number generator
    srand(static_cast<unsigned>(time(nullptr)));
}

// Generate food at a random position
void SnakeGame::generateFood() {
    food.x = rand() % width;
    food.y = rand() % height;
    
    // Make sure food doesn't spawn on the snake
    for (const auto& segment : snake) {
        if (segment.x == food.x && segment.y == food.y) {
            generateFood(); // Recursively try again
            return;
        }
    }
}

// Draw the game
void SnakeGame::draw() {
    // Clear the console
    clearScreen();
    
    // Draw top border
    for (int i = 0; i < width + 2; i++)
        std::cout << "#";
    std::cout << std::endl;
    
    // Draw game area
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width + 2; x++) {
            // Left border
            if (x == 0)
                std::cout << "#";
            
            // Snake head
            else if (x - 1 == snake[0].x && y == snake[0].y)
                std::cout << "O";
            
            // Food
            else if (x - 1 == food.x && y == food.y)
                std::cout << "F";
            
            // Snake body
            else {
                bool printTail = false;
                for (size_t i = 1; i < snake.size(); i++) {
                    if (x - 1 == snake[i].x && y == snake[i].y) {
                        std::cout << "o";
                        printTail = true;
                        break;
                    }
                }
                
                // Empty space
                if (!printTail)
                    std::cout << " ";
            }
            
            // Right border
            if (x == width + 1)
                std::cout << "#";
        }
        std::cout << std::endl;
    }
    
    // Draw bottom border
    for (int i = 0; i < width + 2; i++)
        std::cout << "#";
    std::cout << std::endl;
    
    // Display score
    std::cout << "Score: " << score << std::endl;
}

// Handle user input
void SnakeGame::input() {
    // Check if a key is pressed
    if (kbhit()) {
        // Get the key
        switch (getch()) {
            case 'a':
                if (direction != RIGHT) // Prevent 180-degree turns
                    direction = LEFT;
                break;
            case 'd':
                if (direction != LEFT)
                    direction = RIGHT;
                break;
            case 'w':
                if (direction != DOWN)
                    direction = UP;
                break;
            case 's':
                if (direction != UP)
                    direction = DOWN;
                break;
            case 'x':
                gameOver = true;
                break;
        }
    }
}

// Game logic
void SnakeGame::logic() {
    // Create a new head position based on the current direction
    Position newHead = snake[0];
    
    // Move the head based on direction
    switch (direction) {
        case LEFT:
            newHead.x--;
            break;
        case RIGHT:
            newHead.x++;
            break;
        case UP:
            newHead.y--;
            break;
        case DOWN:
            newHead.y++;
            break;
        default:
            break;
    }
    
    // Check for collisions with walls
    if (newHead.x < 0 || newHead.x >= width || newHead.y < 0 || newHead.y >= height) {
        gameOver = true;
        return;
    }
    
    // Check for collisions with self
    for (size_t i = 1; i < snake.size(); i++) {
        if (newHead.x == snake[i].x && newHead.y == snake[i].y) {
            gameOver = true;
            return;
        }
    }
    
    // Check if food is eaten
    if (newHead.x == food.x && newHead.y == food.y) {
        // Increase score
        score += 10;
        
        // Generate new food
        generateFood();
        
        // Add new head (snake grows)
        snake.insert(snake.begin(), newHead);
    } else {
        // Add new head
        snake.insert(snake.begin(), newHead);
        
        // Remove tail (snake moves)
        snake.pop_back();
    }
}

// Main game loop
void SnakeGame::run() {
    while (!gameOver) {
        draw();
        input();
        logic();
        
        // Control game speed
        sleepMs(100);
    }
    
    // Game over message
    clearScreen();
    std::cout << "Game Over!" << std::endl;
    std::cout << "Final Score: " << score << std::endl;
    std::cout << "Press any key to exit..." << std::endl;
    getch();
}

// Cross-platform helper methods

// Clear the screen
void SnakeGame::clearScreen() {
    // ANSI escape code to clear screen
    std::cout << "\033[2J\033[1;1H";
}

// Check if a key has been pressed
bool SnakeGame::kbhit() {
    int bytesWaiting;
    ioctl(STDIN_FILENO, FIONREAD, &bytesWaiting);
    return bytesWaiting > 0;
}

// Get a character from the terminal without waiting for Enter
char SnakeGame::getch() {
    char buf = 0;
    read(STDIN_FILENO, &buf, 1);
    return buf;
}

// Sleep for a specified number of milliseconds
void SnakeGame::sleepMs(int milliseconds) {
    std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
}

// Setup terminal for non-blocking input
void SnakeGame::setupTerminal() {
    // Get the current terminal settings
    tcgetattr(STDIN_FILENO, &oldSettings);
    
    // Create new settings
    struct termios newSettings = oldSettings;
    
    // Disable canonical mode and echo
    newSettings.c_lflag &= ~(ICANON | ECHO);
    
    // Apply the new settings
    tcsetattr(STDIN_FILENO, TCSANOW, &newSettings);
    
    // Set non-blocking mode
    int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK);
}

// Restore terminal settings
void SnakeGame::restoreTerminal() {
    // Restore the old terminal settings
    tcsetattr(STDIN_FILENO, TCSANOW, &oldSettings);
} 