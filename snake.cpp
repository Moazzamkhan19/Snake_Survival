/*
    Multi-Snake Game — SFML + OpenMP + MPI
    ----------------------------------------
    Process 0 (MASTER): runs SFML window, player snake, draws everything
    Process 1:          handles AI Snake 0
    Process 2:          handles AI Snake 1
    Process 3:          handles AI Snake 2

    MPI Flow every frame:
      Master → sends food list + alive status + signal to workers
      Workers → update direction (OpenMP) + move → send body back
      Master → receives bodies → draws everything

    Fixes:
      - Master sends alive status TO workers so they stop if killed
      - Single initialization point for AI snakes (master sends init data)
*/

#include <SFML/Graphics.hpp>
#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <deque>
#include <cstdlib>
#include <ctime>
#include <string>

// ─── Constants ────────────────────────────────────────────────────────────────
const int WINDOW_W        = 620;
const int WINDOW_H        = 660;
const int GRID_COLS       = 30;
const int GRID_ROWS       = 30;
const int CELL            = 20;
const int OFFSET_X        = 10;
const int OFFSET_Y        = 50;
const int NUM_AI          = 6;
const float MOVE_INTERVAL = 0.15f;
const int MAX_SNAKE_LEN   = 900;

// MPI Tags
const int TAG_SIGNAL   = 1;   // master → worker: continue(1) or stop(-1)
const int TAG_FOOD     = 2;   // master → worker: food list
const int TAG_ALIVE    = 3;   // master → worker: is this snake still alive?
const int TAG_BODY     = 4;   // worker → master: snake body
const int TAG_DEAD     = 5;   // worker → master: did snake die this frame?
const int TAG_ATEFOOD  = 6;   // worker → master: did snake eat food?

// ─── Direction ────────────────────────────────────────────────────────────────
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

// ─── Globals ──────────────────────────────────────────────────────────────────
Snake             player;
Snake             aiSnakes[NUM_AI];
std::vector<Vec2> foods;
int               score       = 0;
bool              gameOver    = false;
std::string       gameOverMsg = "";

// ─── AI start positions ───────────────────────────────────────────────────────
const int AI_SX[NUM_AI] = { 4, 25, 4, 25, 14, 4 };
const int AI_SY[NUM_AI] = { 4,  4, 25, 25,  4, 14 };
const Dir AI_SD[NUM_AI] = { RIGHT, LEFT, RIGHT, LEFT, RIGHT, RIGHT };


sf::Color   AI_COLS[NUM_AI] = {
    sf::Color(220, 70,  70),
    sf::Color(70,  140, 220),
    sf::Color(220, 190, 50),
    sf::Color(180, 70,  220),
    sf::Color(220, 50,  150),
    sf::Color(150, 220, 50),
};

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

// ─── Spawn food ───────────────────────────────────────────────────────────────
void spawnFood() {
    Vec2 f;
    int tries = 0;
    do {
        f = { rnd(0, GRID_COLS - 1), rnd(0, GRID_ROWS - 1) };
        if (++tries > 500) return;
        bool occupied = false;
        for (auto& s : player.body)      if (s == f) { occupied = true; break; }
        if (occupied) continue;
        for (int i = 0; i < NUM_AI; i++)
            for (auto& s : aiSnakes[i].body) if (s == f) { occupied = true; break; }
        for (auto& e : foods)            if (e == f) { occupied = true; break; }
        if (!occupied) break;
    } while (true);
    foods.push_back(f);
}

// ─── Init ─────────────────────────────────────────────────────────────────────
void initPlayer() {
    player.body.clear();
    player.body.push_back({ GRID_COLS / 2,     GRID_ROWS / 2 });
    player.body.push_back({ GRID_COLS / 2 - 1, GRID_ROWS / 2 });
    player.body.push_back({ GRID_COLS / 2 - 2, GRID_ROWS / 2 });
    player.dir   = RIGHT;
    player.alive = true;
    player.color = sf::Color(50, 220, 50);
}

// Master initializes AI snakes with correct starting size (3 segments)
void initAI() {
    for (int i = 0; i < NUM_AI; i++) {
        aiSnakes[i].body.clear();
        aiSnakes[i].body.push_back({ AI_SX[i],     AI_SY[i] });
        aiSnakes[i].body.push_back({ AI_SX[i] - 1, AI_SY[i] });
        aiSnakes[i].body.push_back({ AI_SX[i] - 2, AI_SY[i] });
        aiSnakes[i].dir   = AI_SD[i];
        aiSnakes[i].alive = true;
        aiSnakes[i].color = AI_COLS[i];
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

// ─── Move player snake ────────────────────────────────────────────────────────
bool moveSnake(Snake& s, bool isPlayer = false) {
    if (!s.alive) return false;
    Vec2 next = nextPos(s.head(), s.dir);
    if (hitsWall(next)) {
        s.alive = false;
        if (isPlayer) gameOverMsg = "You hit the wall!";
        return false;
    }
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

// ─── AI direction ─────────────────────────────────────────────────────────────
void updateAIDir(Snake& s) {
    if (!s.alive) return;
    Vec2 fwd = nextPos(s.head(), s.dir);
    if (!hitsWall(fwd) && (rand() % 10) < 7) return;

    Dir dirs[4] = { UP, DOWN, LEFT, RIGHT };
    for (int i = 3; i > 0; i--) {
        int j = rand() % (i + 1);
        Dir tmp = dirs[i]; dirs[i] = dirs[j]; dirs[j] = tmp;
    }
    Dir opp = opposite(s.dir);
    for (int i = 0; i < 4; i++) {
        if (dirs[i] == opp) continue;
        if (!hitsWall(nextPos(s.head(), dirs[i]))) {
            s.dir = dirs[i];
            return;
        }
    }
}

// ─── MPI pack/unpack ──────────────────────────────────────────────────────────
void packSnake(const Snake& s, int* buf, int& outLen) {
    int len = (int)s.body.size();
    if (len > MAX_SNAKE_LEN) len = MAX_SNAKE_LEN;
    buf[0] = len;
    for (int i = 0; i < len; i++) {
        buf[1 + i * 2]     = s.body[i].x;
        buf[1 + i * 2 + 1] = s.body[i].y;
    }
    outLen = 1 + len * 2;
}

void unpackSnake(Snake& s, const int* buf) {
    int len = buf[0];
    s.body.clear();
    for (int i = 0; i < len; i++)
        s.body.push_back({ buf[1 + i * 2], buf[1 + i * 2 + 1] });
}

// ─── Collision ────────────────────────────────────────────────────────────────
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

void checkAIvsPlayer() {
    for (int i = 0; i < NUM_AI; i++) {
        if (!aiSnakes[i].alive) continue;
        for (auto& seg : player.body) {
            if (seg == aiSnakes[i].head()) {
                aiSnakes[i].alive = false;
                score += 50;
                break;
            }
        }
    }
}

// ─── Drawing ──────────────────────────────────────────────────────────────────
void drawCell(sf::RenderWindow& win, int gx, int gy, sf::Color col) {
    sf::RectangleShape r(sf::Vector2f(CELL - 2, CELL - 2));
    r.setPosition((float)(OFFSET_X + gx * CELL + 1),
                  (float)(OFFSET_Y + gy * CELL + 1));
    r.setFillColor(col);
    win.draw(r);
}

void drawSnake(sf::RenderWindow& win, const Snake& s) {
    if (!s.alive) return;  // ← dead snake = not drawn at all
    bool isHead = true;
    for (auto& seg : s.body) {
        drawCell(win, seg.x, seg.y, isHead ? sf::Color::White : s.color);
        isHead = false;
    }
}

// ─── WORKER PROCESS ───────────────────────────────────────────────────────────
void workerProcess(int rank) {
    int aiIdx = rank - 1;
    srand((unsigned)time(nullptr) + rank * 1234);

    // Initialize local snake with correct 3-segment start
    Snake mySnake;
    mySnake.body.push_back({ AI_SX[aiIdx],     AI_SY[aiIdx] });
    mySnake.body.push_back({ AI_SX[aiIdx] - 1, AI_SY[aiIdx] });
    mySnake.body.push_back({ AI_SX[aiIdx] - 2, AI_SY[aiIdx] });
    mySnake.dir   = AI_SD[aiIdx];
    mySnake.alive = true;

    int buf[MAX_SNAKE_LEN * 2 + 1];

    while (true) {
        // ── 1. Receive signal from master (continue or stop) ─────────────
        int signal = 0;
        MPI_Recv(&signal, 1, MPI_INT, 0, TAG_SIGNAL,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (signal == -1) break; // quit

        // ── 2. Receive alive status from master ───────────────────────────
        // Master may have killed this snake (player hit it)
        int aliveFromMaster = 1;
        MPI_Recv(&aliveFromMaster, 1, MPI_INT, 0, TAG_ALIVE,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        mySnake.alive = (aliveFromMaster == 1);

        // ── 3. Receive food list from master ──────────────────────────────
        int foodCount = 0;
        MPI_Recv(&foodCount, 1, MPI_INT, 0, TAG_FOOD,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        std::vector<int> fbuf(foodCount * 2 + 1);
        if (foodCount > 0)
            MPI_Recv(fbuf.data(), foodCount * 2, MPI_INT, 0, TAG_FOOD + 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        std::vector<Vec2> localFoods;
        for (int i = 0; i < foodCount; i++)
            localFoods.push_back({ fbuf[i * 2], fbuf[i * 2 + 1] });

        // ── 4. Update direction using OpenMP ──────────────────────────────
        #pragma omp parallel sections num_threads(1)
        {
            #pragma omp section
            { updateAIDir(mySnake); }
        }

        // ── 5. Move snake ─────────────────────────────────────────────────
        int diedThisFrame = 0;
        int ateFood       = 0;

        if (mySnake.alive) {
            Vec2 next = nextPos(mySnake.head(), mySnake.dir);
            if (hitsWall(next)) {
                mySnake.alive  = false;
                diedThisFrame  = 1;
            } else {
                for (int i = 0; i < (int)localFoods.size(); i++) {
                    if (localFoods[i] == next) {
                        ateFood = 1;
                        break;
                    }
                }
                mySnake.body.push_front(next);
                if (!ateFood) mySnake.body.pop_back();
            }
        }

        // ── 6. Send body back to master ───────────────────────────────────
        int packedLen = 0;
        packSnake(mySnake, buf, packedLen);
        MPI_Send(buf, packedLen, MPI_INT, 0, TAG_BODY, MPI_COMM_WORLD);

        // ── 7. Send dead/alive and food flags ─────────────────────────────
        int nowAlive = mySnake.alive ? 1 : 0;
        MPI_Send(&nowAlive,      1, MPI_INT, 0, TAG_DEAD,    MPI_COMM_WORLD);
        MPI_Send(&ateFood,       1, MPI_INT, 0, TAG_ATEFOOD, MPI_COMM_WORLD);
    }
}

// ─── MASTER PROCESS ───────────────────────────────────────────────────────────
void masterProcess() {
    sf::RenderWindow window(
        sf::VideoMode(WINDOW_W, WINDOW_H),
        "Multi-Snake  |  WASD=Move  R=Restart  ESC=Quit  [MPI+OpenMP]",
        sf::Style::Titlebar | sf::Style::Close
    );
    window.setFramerateLimit(60);

    sf::Font font;
    bool hasFont = font.loadFromFile("C:\\Windows\\Fonts\\arial.ttf");

    sf::Text scoreText, overText, hintText, mpiText;
    if (hasFont) {
        scoreText.setFont(font); scoreText.setCharacterSize(18);
        scoreText.setFillColor(sf::Color::White);
        scoreText.setPosition(8, 12);

        overText.setFont(font);  overText.setCharacterSize(30);
        overText.setFillColor(sf::Color(255, 80, 80));
        overText.setPosition(60, WINDOW_H / 2 - 50);

        hintText.setFont(font);  hintText.setCharacterSize(18);
        hintText.setFillColor(sf::Color(180, 180, 180));
        hintText.setPosition(100, WINDOW_H / 2 + 10);

        mpiText.setFont(font);   mpiText.setCharacterSize(13);
        mpiText.setFillColor(sf::Color(100, 200, 100));
        mpiText.setPosition(8, WINDOW_H - 22);
        mpiText.setString("MPI: 4 processes | OpenMP: parallel AI directions");
    }

    initGame();
    float     timer = 0.f;
    sf::Clock clk;
    bool      running = true;
    int       buf[MAX_SNAKE_LEN * 2 + 1];

    // Lambda to signal all workers to stop
    auto stopWorkers = [&]() {
        int stopSig = -1;
        int aliveVal = 0;
        int fc = 0;
        for (int r = 1; r <= NUM_AI; r++) {
            MPI_Send(&stopSig,  1, MPI_INT, r, TAG_SIGNAL, MPI_COMM_WORLD);
            MPI_Send(&aliveVal, 1, MPI_INT, r, TAG_ALIVE,  MPI_COMM_WORLD);
            MPI_Send(&fc,       1, MPI_INT, r, TAG_FOOD,   MPI_COMM_WORLD);
        }
    };

    while (window.isOpen()) {
        float dt = clk.restart().asSeconds();

        // ── Events ──────────────────────────────────────────────────────────
        sf::Event ev;
        while (window.pollEvent(ev)) {
            if (ev.type == sf::Event::Closed) {
                stopWorkers();
                window.close();
                running = false;
            }
            if (ev.type == sf::Event::KeyPressed) {
                if (!gameOver) {
                    if (ev.key.code == sf::Keyboard::W && player.dir != DOWN)  player.dir = UP;
                    if (ev.key.code == sf::Keyboard::S && player.dir != UP)    player.dir = DOWN;
                    if (ev.key.code == sf::Keyboard::A && player.dir != RIGHT) player.dir = LEFT;
                    if (ev.key.code == sf::Keyboard::D && player.dir != LEFT)  player.dir = RIGHT;
                }
                if (gameOver && ev.key.code == sf::Keyboard::R) {
                    initGame();
                    timer = 0.f;
                    // Re-send continue to workers after restart
                }
                if (ev.key.code == sf::Keyboard::Escape) {
                    stopWorkers();
                    window.close();
                    running = false;
                }
            }
        }

        if (!running) break;

        // ── Update ──────────────────────────────────────────────────────────
        timer += dt;
        if (timer >= MOVE_INTERVAL) {
            timer = 0.f;

            // Build food buffer
            int foodCount = (int)foods.size();
            std::vector<int> fbuf(foodCount * 2);
            for (int i = 0; i < foodCount; i++) {
                fbuf[i * 2]     = foods[i].x;
                fbuf[i * 2 + 1] = foods[i].y;
            }

            if (!gameOver) {
                // ── Send signal + alive status + food to each worker ─────────
                int contSig = 1;
                for (int r = 1; r <= NUM_AI; r++) {
                    int aiIdx    = r - 1;
                    int aliveVal = aiSnakes[aiIdx].alive ? 1 : 0;

                    MPI_Send(&contSig,  1, MPI_INT, r, TAG_SIGNAL, MPI_COMM_WORLD);
                    MPI_Send(&aliveVal, 1, MPI_INT, r, TAG_ALIVE,  MPI_COMM_WORLD);
                    MPI_Send(&foodCount,1, MPI_INT, r, TAG_FOOD,   MPI_COMM_WORLD);
                    if (foodCount > 0)
                        MPI_Send(fbuf.data(), foodCount * 2, MPI_INT, r,
                                 TAG_FOOD + 1, MPI_COMM_WORLD);
                }

                // ── Move player ──────────────────────────────────────────────
                moveSnake(player, true);

                // ── Receive updated AI data from workers ─────────────────────
                for (int r = 1; r <= NUM_AI; r++) {
                    int aiIdx = r - 1;

                    // Receive body
                    MPI_Recv(buf, MAX_SNAKE_LEN * 2 + 1, MPI_INT, r,
                             TAG_BODY, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    unpackSnake(aiSnakes[aiIdx], buf);

                    // Receive alive status from worker
                    int nowAlive = 1;
                    MPI_Recv(&nowAlive, 1, MPI_INT, r,
                             TAG_DEAD, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                    // Only mark alive if BOTH master and worker say alive
                    // This prevents blinking — master's kill is respected
                    if (!aiSnakes[aiIdx].alive) {
                        nowAlive = 0; // master already killed it, keep dead
                    }
                    aiSnakes[aiIdx].alive = (nowAlive == 1);

                    // Receive ate food
                    int ateFood = 0;
                    MPI_Recv(&ateFood, 1, MPI_INT, r,
                             TAG_ATEFOOD, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    if (ateFood && aiSnakes[aiIdx].alive) spawnFood();
                }

                // ── Collision checks ─────────────────────────────────────────
                checkPlayerCollisions();
                checkAIvsPlayer();
                if (!player.alive) gameOver = true;

            } else {
                // Game is over but workers are waiting — send them continue
                // so they don't block, but with alive=0 so they just idle
                int contSig = 1;
                for (int r = 1; r <= NUM_AI; r++) {
                    int aliveVal = 0; // tell worker its snake is dead
                    MPI_Send(&contSig,  1, MPI_INT, r, TAG_SIGNAL, MPI_COMM_WORLD);
                    MPI_Send(&aliveVal, 1, MPI_INT, r, TAG_ALIVE,  MPI_COMM_WORLD);
                    MPI_Send(&foodCount,1, MPI_INT, r, TAG_FOOD,   MPI_COMM_WORLD);
                    if (foodCount > 0)
                        MPI_Send(fbuf.data(), foodCount * 2, MPI_INT, r,
                                 TAG_FOOD + 1, MPI_COMM_WORLD);
                }
                // Drain worker responses
                for (int r = 1; r <= NUM_AI; r++) {
                    MPI_Recv(buf, MAX_SNAKE_LEN * 2 + 1, MPI_INT, r,
                             TAG_BODY, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    int tmp = 0;
                    MPI_Recv(&tmp, 1, MPI_INT, r, TAG_DEAD,    MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(&tmp, 1, MPI_INT, r, TAG_ATEFOOD, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
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

        // UI
        if (hasFont) {
            std::string ui = "Score: " + std::to_string(score) + "    AI: ";
            for (int i = 0; i < NUM_AI; i++)
                ui += aiSnakes[i].alive ? "[O] " : "[X] ";
            scoreText.setString(ui);
            window.draw(scoreText);
            window.draw(mpiText);

            if (gameOver) {
                sf::RectangleShape overlay(sf::Vector2f(WINDOW_W, WINDOW_H));
                overlay.setFillColor(sf::Color(0, 0, 0, 170));
                window.draw(overlay);
                overText.setString("GAME OVER  -  " + gameOverMsg);
                hintText.setString("Score: " + std::to_string(score) +
                                   "    |    Press R to restart");
                window.draw(overText);
                window.draw(hintText);
            }
        }

        window.display();
    }
}

// ─── Main ─────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 7) {
        if (rank == 0)
            std::cerr << "ERROR: Run with exactly 7 processes!\n"
                      << "Use: mpiexec -n 7 snake.exe\n";
        MPI_Finalize();
        return 1;
    }

    srand((unsigned)time(nullptr) + rank * 999);

    if (rank == 0)
        masterProcess();
    else
        workerProcess(rank);

    MPI_Finalize();
    return 0;
}
