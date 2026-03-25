#include <SFML/Graphics.hpp>
#include <iostream>
#include <vector>
#include <deque>
#include <cstdlib>
#include <ctime>
#include <string>
#include <omp.h>

const int WINDOW_W        = 620;
const int WINDOW_H        = 660;
const int GRID_COLS       = 30;
const int GRID_ROWS       = 30;
const int CELL            = 20;
const int OFFSET_X        = 10;
const int OFFSET_Y        = 50;
const int NUM_AI          = 3;
const float MOVE_INTERVAL = 0.15f;

enum Dir { UP, DOWN, LEFT, RIGHT };

struct Vec2 {
    int x, y;
    bool operator==(const Vec2& o) const { return x == o.x && y == o.y; }
};

struct Snake {
    std::deque<Vec2> body;
    Dir              dir;
    bool             alive;
    sf::Color        color;
    Vec2 head() const { return body.front(); }
};

Snake              player;
Snake              aiSnakes[NUM_AI];
std::vector<Vec2>  foods;
int                score    = 0;
bool               gameOver = false;
std::string        gameOverMsg = "";

// ─── Helpers ──────────────────────────────────────────────────────────────────
int rnd(int lo, int hi) { return lo + rand() % (hi - lo + 1); }

bool hitsWall(Vec2 p) {
    return p.x < 0 || p.x >= GRID_COLS || p.y < 0 || p.y >= GRID_ROWS;
}

Vec2 nextPos(Vec2 p, Dir d) {
    switch (d) {
        case UP:    return { p.x,     p.y - 1 };
        case DOWN:  return { p.x,     p.y + 1 };
        case LEFT:  return { p.x - 1, p.y     };
        case RIGHT: return { p.x + 1, p.y     };
    }
    return p;
}

Dir opposite(Dir d) {
    switch (d) {
        case UP:    return DOWN;
        case DOWN:  return UP;
        case LEFT:  return RIGHT;
        case RIGHT: return LEFT;
    }
    return d;
}

// ─── Spawn food at random free cell ───────────────────────────────────────────
void spawnFood() {
    Vec2 f;
    int tries = 0;
    do {
        f = { rnd(0, GRID_COLS - 1), rnd(0, GRID_ROWS - 1) };
        tries++;
        if (tries > 500) return;
        bool occupied = false;
        for (auto& s : player.body) if (s == f) { occupied = true; break; }
        if (occupied) continue;
        for (int i = 0; i < NUM_AI; i++) {
            for (auto& s : aiSnakes[i].body) if (s == f) { occupied = true; break; }
        }
        for (auto& existing : foods) if (existing == f) { occupied = true; break; }
        if (!occupied) break;
    } while (true);
    foods.push_back(f);
}

// ─── Init ─────────────────────────────────────────────────────────────────────
void initPlayer() {
    player.body.clear();
    player.body.push_back({ GRID_COLS / 2,     GRID_ROWS / 2     });
    player.body.push_back({ GRID_COLS / 2 - 1, GRID_ROWS / 2     });
    player.body.push_back({ GRID_COLS / 2 - 2, GRID_ROWS / 2     });
    player.dir   = RIGHT;
    player.alive = true;
    player.color = sf::Color(50, 220, 50);
}

void initAI() {
    sf::Color cols[NUM_AI] = {
        sf::Color(220, 70,  70),
        sf::Color(70,  140, 220),
        sf::Color(220, 190, 50)
    };
    int sx[NUM_AI] = { 4,  25,  4  };
    int sy[NUM_AI] = { 4,  4,   25 };
    Dir sd[NUM_AI] = { RIGHT, LEFT, RIGHT };

    for (int i = 0; i < NUM_AI; i++) {
        aiSnakes[i].body.clear();
        aiSnakes[i].body.push_back({ sx[i],     sy[i] });
        aiSnakes[i].body.push_back({ sx[i] - 1, sy[i] });
        aiSnakes[i].body.push_back({ sx[i] - 2, sy[i] });
        aiSnakes[i].dir   = sd[i];
        aiSnakes[i].alive = true;
        aiSnakes[i].color = cols[i];
    }
}

void initGame() {
    score       = 0;
    gameOver    = false;
    gameOverMsg = "";
    foods.clear();
    initPlayer();
    initAI();
    for (int i = 0; i < 5; i++) spawnFood();
}

// ─── Move one snake; returns true if ate food ─────────────────────────────────
bool moveSnake(Snake& s, bool isPlayer = false) {
    if (!s.alive) return false;

    Vec2 next = nextPos(s.head(), s.dir);

    // Wall
    if (hitsWall(next)) {
        s.alive = false;
        if (isPlayer) gameOverMsg = "You hit the wall!";
        return false;
    }

    // Food
    bool ate = false;
    for (int i = 0; i < (int)foods.size(); i++) {
        if (foods[i] == next) {
            foods.erase(foods.begin() + i);
            ate = true;
            if (isPlayer) score += 10;
            spawnFood();
            break;
        }
    }

    s.body.push_front(next);
    if (!ate) s.body.pop_back();
    return ate;
}

// ─── AI: pick direction (random but avoids walls) ────────────────────────────
void updateAIDir(Snake& s) {
    if (!s.alive) return;

    // 70% chance keep going straight if safe
    Vec2 fwd = nextPos(s.head(), s.dir);
    if (!hitsWall(fwd) && (rand() % 10) < 7) return;

    // Shuffle 4 directions
    Dir dirs[4] = { UP, DOWN, LEFT, RIGHT };
    for (int i = 3; i > 0; i--) {
        int j = rand() % (i + 1);
        Dir tmp = dirs[i]; dirs[i] = dirs[j]; dirs[j] = tmp;
    }

    Dir opp = opposite(s.dir);
    for (int i = 0; i < 4; i++) {
        if (dirs[i] == opp) continue;
        Vec2 nxt = nextPos(s.head(), dirs[i]);
        if (!hitsWall(nxt)) {
            s.dir = dirs[i];
            return;
        }
    }
}

// ─── Update all AI snakes — direction update is PARALLEL via OpenMP ──────────
void updateAISnakes() {
    // Parallel direction decisions
    #pragma omp parallel for num_threads(NUM_AI) schedule(static)
    for (int i = 0; i < NUM_AI; i++) {
        updateAIDir(aiSnakes[i]);
    }
    // Sequential movement (shared grid state)
    for (int i = 0; i < NUM_AI; i++) {
        moveSnake(aiSnakes[i], false);
    }
}

// ─── Collision: player head vs own body and AI bodies ────────────────────────
void checkPlayerCollisions() {
    if (!player.alive) return;
    Vec2 ph = player.head();

    for (int i = 1; i < (int)player.body.size(); i++) {
        if (player.body[i] == ph) {
            player.alive = false;
            gameOverMsg  = "You hit yourself!";
            return;
        }
    }
    for (int i = 0; i < NUM_AI; i++) {
        if (!aiSnakes[i].alive) continue;
        for (auto& seg : aiSnakes[i].body) {
            if (seg == ph) {
                player.alive = false;
                gameOverMsg  = "You hit an AI snake!";
                return;
            }
        }
    }
}

// ─── Collision: AI head vs player body (AI gets eliminated) ──────────────────
void checkAIvsPlayer() {
    for (int i = 0; i < NUM_AI; i++) {
        if (!aiSnakes[i].alive) continue;
        Vec2 ah = aiSnakes[i].head();
        for (auto& seg : player.body) {
            if (seg == ah) {
                aiSnakes[i].alive = false;
                score += 50;
                break;
            }
        }
    }
}

// ─── Drawing helpers ─────────────────────────────────────────────────────────
void drawCell(sf::RenderWindow& win, int gx, int gy, sf::Color col) {
    sf::RectangleShape r(sf::Vector2f(CELL - 2, CELL - 2));
    r.setPosition((float)(OFFSET_X + gx * CELL + 1),
                  (float)(OFFSET_Y + gy * CELL + 1));
    r.setFillColor(col);
    win.draw(r);
}

void drawSnake(sf::RenderWindow& win, const Snake& s) {
    if (!s.alive) return;
    bool isHead = true;
    for (auto& seg : s.body) {
        drawCell(win, seg.x, seg.y, isHead ? sf::Color::White : s.color);
        isHead = false;
    }
}

// ─── Main ─────────────────────────────────────────────────────────────────────
int main() {
    srand((unsigned)time(nullptr));

    sf::RenderWindow window(
        sf::VideoMode(WINDOW_W, WINDOW_H),
        "Multi-Snake  |  WASD = Move  |  R = Restart  |  ESC = Quit",
        sf::Style::Titlebar | sf::Style::Close
    );
    window.setFramerateLimit(60);

    // Load font
    sf::Font font;
    bool hasFont = font.loadFromFile("C:\\Windows\\Fonts\\arial.ttf");

    sf::Text scoreText, overText, hintText;
    if (hasFont) {
        scoreText.setFont(font);  scoreText.setCharacterSize(20);
        scoreText.setFillColor(sf::Color::White);
        scoreText.setPosition(8, 12);

        overText.setFont(font);   overText.setCharacterSize(30);
        overText.setFillColor(sf::Color(255, 80, 80));
        overText.setPosition(60, WINDOW_H / 2 - 50);

        hintText.setFont(font);   hintText.setCharacterSize(18);
        hintText.setFillColor(sf::Color(180, 180, 180));
        hintText.setPosition(130, WINDOW_H / 2 + 10);
    }

    initGame();
    float   timer = 0.f;
    sf::Clock clk;

    while (window.isOpen()) {
        float dt = clk.restart().asSeconds();

        // ── Events ──────────────────────────────────────────────────────────
        sf::Event ev;
        while (window.pollEvent(ev)) {
            if (ev.type == sf::Event::Closed) window.close();
            if (ev.type == sf::Event::KeyPressed) {
                if (!gameOver) {
                    if (ev.key.code == sf::Keyboard::W && player.dir != DOWN)  player.dir = UP;
                    if (ev.key.code == sf::Keyboard::S && player.dir != UP)    player.dir = DOWN;
                    if (ev.key.code == sf::Keyboard::A && player.dir != RIGHT) player.dir = LEFT;
                    if (ev.key.code == sf::Keyboard::D && player.dir != LEFT)  player.dir = RIGHT;
                }
                if (gameOver  && ev.key.code == sf::Keyboard::R) { initGame(); timer = 0.f; }
                if (ev.key.code == sf::Keyboard::Escape) window.close();
            }
        }

        // ── Update ──────────────────────────────────────────────────────────
        if (!gameOver) {
            timer += dt;
            if (timer >= MOVE_INTERVAL) {
                timer = 0.f;
                moveSnake(player, true);
                updateAISnakes();
                checkPlayerCollisions();
                checkAIvsPlayer();
                if (!player.alive) gameOver = true;
            }
        }

        // ── Draw ────────────────────────────────────────────────────────────
        window.clear(sf::Color(15, 15, 20));

        // Grid
        for (int x = 0; x < GRID_COLS; x++)
            for (int y = 0; y < GRID_ROWS; y++) {
                sf::RectangleShape cell(sf::Vector2f(CELL - 1, CELL - 1));
                cell.setPosition((float)(OFFSET_X + x * CELL),
                                 (float)(OFFSET_Y + y * CELL));
                cell.setFillColor(sf::Color(25, 25, 35));
                window.draw(cell);
            }

        // Food
        for (auto& f : foods) {
            sf::CircleShape c(CELL / 2 - 2);
            c.setFillColor(sf::Color(255, 80, 80));
            c.setPosition((float)(OFFSET_X + f.x * CELL + 2),
                          (float)(OFFSET_Y + f.y * CELL + 2));
            window.draw(c);
        }

        // Snakes
        drawSnake(window, player);
        for (int i = 0; i < NUM_AI; i++) drawSnake(window, aiSnakes[i]);

        // UI bar
        if (hasFont) {
            std::string ui = "Score: " + std::to_string(score) + "    AI alive: ";
            for (int i = 0; i < NUM_AI; i++)
                ui += aiSnakes[i].alive ? "[O] " : "[X] ";
            scoreText.setString(ui);
            window.draw(scoreText);
        }

        // Game Over overlay
        if (gameOver && hasFont) {
            sf::RectangleShape overlay(sf::Vector2f(WINDOW_W, WINDOW_H));
            overlay.setFillColor(sf::Color(0, 0, 0, 170));
            window.draw(overlay);
            overText.setString("GAME OVER  -  " + gameOverMsg);
            hintText.setString("Score: " + std::to_string(score) +
                               "    |    Press R to restart");
            window.draw(overText);
            window.draw(hintText);
        }

        window.display();
    }
    return 0;
}
