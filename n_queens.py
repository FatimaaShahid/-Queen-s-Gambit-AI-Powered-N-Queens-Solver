import pygame
import sys
import random
import math
import heapq
from copy import deepcopy
from time import perf_counter as now
import matplotlib.pyplot as plt
import numpy as np

# ---------------- Config ----------------
FPS = 60
DEFAULT_N = 8
MAX_NODES = 500000  # <- Stop uninformed search after 500k expansions

# Colors (chess style)
BG_COLOR = (28, 28, 38)
WHITE_SQUARE = (240, 240, 240)
BLACK_SQUARE = (60, 60, 60)
QUEEN_COLOR = (200, 30, 30)
TEXT_COLOR = (230, 230, 230)
BUTTON_COLOR = (70, 130, 220)
BUTTON_HOVER = (95, 150, 240)
SUCCESS_COLOR = (50, 200, 100)
ERROR_COLOR = (240, 80, 80)

# ---------------- Helpers ----------------
def conflicts(board): #board = [col for row 0,col for row 1,col for row 2,...] where index is the row
    n = len(board)
    cnt = 0
    for i in range(n):
        for j in range(i+1, n):
            if board[i] == board[j] or abs(board[i]-board[j]) == abs(i-j):
                cnt += 1
    return cnt

def random_board(n):
    return [random.randrange(n) for _ in range(n)]

def copy_board(b):
    return b[:] if b is not None else None

# ---------------- Local Search ----------------
def hill_climb(initial_board, max_iters=2000):
    start = now()
    board = initial_board[:]
    n = len(board)
    iters = 0
    while iters < max_iters:
        iters += 1
        cur = conflicts(board)
        if cur == 0:
            return board, True, iters, now()-start
        best = cur
        moves = []
        for r in range(n):
            orig = board[r]
            for c in range(n):
                if c == orig: continue
                board[r] = c
                v = conflicts(board)
                if v < best:
                    best = v
                    moves = [(r,c)]
                elif v == best:
                    moves.append((r,c))
            board[r] = orig
        if not moves: #reached local minimum
            break
        r,c = random.choice(moves) #if multiple good moves, pick one at random
        board[r] = c
    return board, False, iters, now()-start

def simulated_annealing(initial_board, max_iters=5000, temp=5.0, cooling=0.995):
    start = now()
    n = len(initial_board)
    state = initial_board[:]
    cur_cost = conflicts(state)
    for i in range(max_iters):
        if cur_cost == 0:
            return state, True, i, now()-start 
        temp *= cooling #This reduces randomness over time.
        r = random.randrange(n)
        c = random.randrange(n)
        new = state[:]
        new[r] = c
        new_cost = conflicts(new)
        delta = new_cost - cur_cost
        if delta < 0 or random.random() < math.exp(-delta / max(temp, 1e-8)):
            state = new
            cur_cost = new_cost
    return state, False, max_iters, now()-start

# ---------------- Uninformed (BFS/DFS/UCS/IDS) ----------------
def bfs(n, initial=None):
    start = now()
    queue = [[]]
    nodes = 0
    while queue:
        path = queue.pop(0)
        nodes += 1
        if nodes > MAX_NODES:
            return None, False, nodes, now() - start
        row = len(path)
        if row == n:
            return path, True, nodes, now()-start
        for col in range(n):
            if all(col != path[i] and abs(col-path[i]) != row-i for i in range(row)):
                queue.append(path + [col])
    return None, False, nodes, now()-start

def dfs(n, initial=None, limit=None):
    start = now()
    stack = [[]]
    nodes = 0
    while stack:
        path = stack.pop()
        nodes += 1
        if nodes > MAX_NODES:
            return None, False, nodes, now() - start
        row = len(path)
        if row == n:
            return path, True, nodes, now()-start
        if limit is not None and row >= limit:
            continue
        for col in range(n):
            if all(col != path[i] and abs(col-path[i]) != row-i for i in range(row)):
                stack.append(path + [col])
    return None, False, nodes, now()-start

def ids(n, initial=None, max_depth=20):
    nodes_total = 0
    start = now()
    for d in range(1, max_depth+1):
        res, ok, nodes, _t = dfs(n, initial, limit=d)
        nodes_total += nodes
        if nodes > MAX_NODES:
            return None, False, nodes, now() - start
        if ok:
            return res, True, nodes_total, now()-start
    return None, False, nodes_total, now()-start

def ucs(n, initial=None):
    # For N-Queens cost is uniform; UCS reduces to BFS in this formulation.
    start = now()
    pq = [(0, [])]
    nodes = 0
    while pq:
        cost, path = heapq.heappop(pq)
        nodes += 1
        if nodes > MAX_NODES:
            return None, False, nodes, now() - start
        row = len(path)
        if row == n:
            return path, True, nodes, now()-start
        for col in range(n):
            if all(col != path[i] and abs(col-path[i]) != row-i for i in range(row)):
                heapq.heappush(pq, (cost+1, path + [col]))
    return None, False, nodes, now()-start

# ---------------- Informed (Greedy / A*) ----------------
def greedy_best_first(n, initial=None):
    start = now()
    pq = []
    # Use heuristic = conflicts of partial assignment
    heapq.heappush(pq, (0, []))
    nodes = 0
    while pq:
        h, path = heapq.heappop(pq)
        nodes += 1
        if nodes > MAX_NODES:
            return None, False, nodes, now() - start

        row = len(path)
        if row == n:
            return path, True, nodes, now()-start
        for col in range(n):
            if all(col != path[i] and abs(col-path[i]) != row-i for i in range(row)):
                newp = path + [col]
                heapq.heappush(pq, (conflicts(newp), newp))
    return None, False, nodes, now()-start

def astar(n, initial=None):
    start = now()
    pq = []
    heapq.heappush(pq, (0, [], 0))  # f=g+h, path, g = cost
    nodes = 0
    visited = set()
    while pq:
        f, path, g = heapq.heappop(pq) #This removes the state with the smallest f-value,
        nodes += 1 #node expanded
        if nodes > MAX_NODES:
            return None, False, nodes, now() - start

        row = len(path) #path = [col_of_row_0, col_of_row_1, ..., col_of_row_k] the assigned rows

        key = tuple(path)
        if key in visited:
            continue
        visited.add(key) #visisted look like {
    # (),
    # (2,),
    # (2, 4),
    # (2, 4, 1),
    # ...
# }

        if row == n:
            return path, True, nodes, now()-start
        for col in range(n):
            if all(col != path[i] and abs(col-path[i]) != row-i for i in range(row)):
                newp = path + [col]
                gg = g + 1
                hh = conflicts(newp)
                heapq.heappush(pq, (gg + hh, newp, gg))
    return None, False, nodes, now()-start

# ---------------- CSP Backtracking ----------------
def csp_backtracking(n, initial=None, max_nodes=200000):
    start = now() #the start time
    domains = {r: list(range(n)) for r in range(n)} # they will lok like {0: [0,1,2,...,n-1], 1: [0,1,2,...,n-1], ...}
    assignment = {}
    nodes = 0

    def select_var():
        unass = [v for v in range(n) if v not in assignment]
        unass.sort(key=lambda x: len(domains[x])) #minimum rem values
        return unass[0] if unass else None

    def consistent(var, val): #checks if placing that queen in the row is consistent with current assignment
        for a,v in assignment.items(): #assignmet[row] = col this si how we represent the assignment
            if v == val or abs(a-var) == abs(v-val): #same col conflict or diagonal conflict(diff in rows = diff in cols)
                return False
        return True

    def forward_check(var, val): #removes values from domains of unassigned variables that are inconsistent with var=val
        newd = deepcopy(domains)
        for v in range(n): #checking all rows
            if v in assignment or v == var: #skipping assigned rows and the current row
                continue
            newd[v] = [x for x in newd[v] if x != val and abs(v-var) != abs(x-val)] #removing all conflicting columns
            if not newd[v]:
                return None
        return newd

    def backtrack():
        nonlocal nodes, domains #allow modifying outer scope variables
        if len(assignment) == n: #if all solutions found
            return True
        var = select_var() #select new var and if not found return false
        if var is None: #no possible var
            return False
        for val in list(domains[var]):
            nodes += 1 #node expanded
            if not consistent(var, val):
                continue
            assignment[var] = val
            saved = deepcopy(domains) #This line creates a full copy of the current domains dictionary before you modify it.
            nd = forward_check(var, val)
            if nd is not None:
                domains = nd
                if backtrack(): #the recursive call
                    return True
            domains = saved
            del assignment[var]
            if nodes > MAX_NODES:
                return None, False, nodes, now() - start

        return False

    res = backtrack()
    if res:
        board = [assignment[i] for i in range(n)]
        return board, True, nodes, now()-start
    else:
        return None, False, nodes, now()-start

# ---------------- Registry ----------------
ALGORITHMS = {
    # Local (will take initial_board)
    "Hill Climb": lambda n, initial: hill_climb(initial),
    "Simulated Annealing": lambda n, initial: simulated_annealing(initial),
    # Uninformed
    "BFS": lambda n, initial: bfs(n),
    "DFS": lambda n, initial: dfs(n),
    "UCS": lambda n, initial: ucs(n),
    "IDS": lambda n, initial: ids(n),
    # Informed
    "Greedy Best-First": lambda n, initial: greedy_best_first(n),
    "A*": lambda n, initial: astar(n),
    # CSP
    "CSP Backtracking": lambda n, initial: csp_backtracking(n),
}

ALG_LIST = list(ALGORITHMS.keys())

# ---------------- GUI Utilities ----------------
class Button:
    def __init__(self, x,y,w,h,text,color=BUTTON_COLOR):
        self.rect = pygame.Rect(x,y,w,h)
        self.text = text
        self.color = color
        self.hover = False
    def draw(self, screen, font):
        col = BUTTON_HOVER if self.hover else self.color
        pygame.draw.rect(screen, col, self.rect, border_radius=6)
        txt = font.render(self.text, True, (255,255,255))
        screen.blit(txt, txt.get_rect(center=self.rect.center))
    def check_hover(self, pos):
        self.hover = self.rect.collidepoint(pos)

# ---------------- Chessboard Drawing (chess style) ----------------
def draw_chess_board(surface, board, top_left, size, show_coords=False):
    """
    Draws chess-style board with each queen and her attack path having the same color.
    """
    x0, y0 = top_left
    n = len(board)
    sq = size // n

    # Distinct bright colors (cycled if n > len)
    COLORS = [
        (220, 50, 47),    # Red
        (203, 75, 22),    # Orange
        (38, 139, 210),   # Blue
        (108, 113, 196),  # Purple
        (133, 153, 0),    # Olive Green
        (211, 54, 130),   # Magenta
        (42, 161, 152),   # Teal
        (181, 137, 0),    # Mustard
        (255, 215, 0),    # Gold
        (70, 130, 180),   # Steel Blue
    ]
    BORDER_THICKNESS = max(2, sq // 14)

    # Compute each queen's conflict tiles (path of attacks)
    queen_conflicts = [set() for _ in range(n)]
    if board is not None:
        for i in range(n):
            for j in range(i + 1, n):
                r1, c1 = i, board[i]
                r2, c2 = j, board[j]

                # Same column conflict
                if c1 == c2:
                    step = 1 if r1 < r2 else -1
                    for r in range(r1, r2 + step, step):
                        queen_conflicts[i].add((r, c1))
                        queen_conflicts[j].add((r, c2))

                # Diagonal conflict
                elif abs(r1 - r2) == abs(c1 - c2):
                    dr = 1 if r2 > r1 else -1
                    dc = 1 if c2 > c1 else -1
                    r, c = r1, c1
                    while True:
                        queen_conflicts[i].add((r, c))
                        queen_conflicts[j].add((r, c))
                        if r == r2 and c == c2:
                            break
                        r += dr
                        c += dc

    # Draw base chessboard
    for r in range(n):
        for c in range(n):
            color = WHITE_SQUARE if (r + c) % 2 == 0 else BLACK_SQUARE
            rect = pygame.Rect(x0 + c * sq, y0 + r * sq, sq, sq)
            pygame.draw.rect(surface, color, rect)

    # Draw each queen’s conflict tiles
    if board is not None:
        for qi, tiles in enumerate(queen_conflicts):
            path_color = COLORS[qi % len(COLORS)]
            for (r, c) in tiles:
                rect = pygame.Rect(x0 + c * sq, y0 + r * sq, sq, sq)
                pygame.draw.rect(surface, path_color, rect, BORDER_THICKNESS)

    # Draw queens in their own color (same as their path)
    if board is not None:
        for r in range(n):
            c = board[r]
            cx = x0 + c * sq + sq // 2
            cy = y0 + r * sq + sq // 2
            radius = max(8, sq // 3)
            q_color = COLORS[r % len(COLORS)]
            pygame.draw.circle(surface, q_color, (cx, cy), radius)
            # optional: white outline for clarity
            pygame.draw.circle(surface, (255, 255, 255), (cx, cy), radius, 1)


# ---------------- Main Visual GUI (keeps your original feel) ----------------
class QueensGUI:
    def __init__(self, N=DEFAULT_N):
        pygame.init()
        self.N = N
        # compute sizes
        self.SQ = max(28, min(64, 480 // N))
        self.BOARD_PIX = self.SQ * N
        self.W = self.BOARD_PIX + 360
        self.H = self.BOARD_PIX + 140
        # self.screen = pygame.display.set_mode((self.W, self.H))
        self.screen = pygame.display.set_mode((1600, 900), pygame.RESIZABLE)

        pygame.display.set_caption(f"{N}-Queens Solver")
        try:
            self.title_font = pygame.font.SysFont('arial', 28, bold=True)
            self.font = pygame.font.SysFont('arial', 16)
        except:
            self.title_font = pygame.font.Font(None, 32)
            self.font = pygame.font.Font(None, 18)
        self.board = random_board(N)
        self.msg = "Click a button to start."
        # buttons
        by = self.H - 80
        self.buttons = [
            Button(40, by, 120, 40, "Randomize"),
            Button(180, by, 140, 40, "Run Algorithm"),
            Button(340, by, 140, 40, "Compare"),
            Button(self.W-140, by, 100, 40, "Exit", (200,80,80))
        ]
        # algorithm dropdown-like choice for single run
        self.alg_choice = 0

    # def draw(self):
    #     self.screen.fill(BG_COLOR)
    #     title = self.title_font.render(f"{self.N}-Queens", True, TEXT_COLOR)
    #     self.screen.blit(title, (20, 12))
    #     # draw a simple algorithm selector text
    #     alg_txt = self.font.render(f"Selected algorithm: {ALG_LIST[self.alg_choice]}", True, TEXT_COLOR)
    #     self.screen.blit(alg_txt, (20, 52))
    #     # draw small left list of algorithms (click to cycle)
    #     y = 80
    #     for i, name in enumerate(ALG_LIST):
    #         color = (200,200,200) if i == self.alg_choice else (150,150,150)
    #         surf = self.font.render(name, True, color)
    #         self.screen.blit(surf, (20, y))
    #         y += 22

    #     # draw chess-style board on right
    #     top_left = (200, 40)
    #     draw_chess_board(self.screen, self.board, top_left, self.BOARD_PIX)
    #     # small info
    #     conf = conflicts(self.board)
    #     status = "Solved!" if conf == 0 else f"Conflicts: {conf}"
    #     st_col = SUCCESS_COLOR if conf==0 else TEXT_COLOR
    #     st_surf = self.font.render(status, True, st_col)
    #     self.screen.blit(st_surf, (200, 50 + self.BOARD_PIX + 8))

    #     # draw buttons
    #     for b in self.buttons:
    #         b.draw(self.screen, self.font)
    #     # message
    #     msg_surf = self.font.render(self.msg, True, TEXT_COLOR)
    #     self.screen.blit(msg_surf, (20, self.H-110))
    def draw(self):
        self.screen.fill(BG_COLOR)
        W, H = self.screen.get_size()

        # dynamic font scaling
        font_size = max(14, int(min(W, H) / 60))
        title_size = max(20, int(min(W, H) / 35))
        font = pygame.font.SysFont('arial', font_size)
        title_font = pygame.font.SysFont('arial', title_size, bold=True)

        # draw title
        title = title_font.render(f"{self.N}-Queens", True, TEXT_COLOR)
        self.screen.blit(title, (20, 12))

        # algorithm label
        alg_txt = font.render(f"Selected algorithm: {ALG_LIST[self.alg_choice]}", True, TEXT_COLOR)
        self.screen.blit(alg_txt, (20, 52))

        # algorithm list (left side)
        y = 80
        for i, name in enumerate(ALG_LIST):
            color = (200,200,200) if i == self.alg_choice else (150,150,150)
            surf = font.render(name, True, color)
            self.screen.blit(surf, (20, y))
            y += font_size + 6

        # compute board size dynamically
        # make board slightly larger and more centered
        board_size = int(min(W * 0.60, H * 0.75))

        # dynamically center board based on window width
           # give algorithm list more space
        board_x = (W - board_size) // 2 + 60    # shift a little left from full center
        board_y = 80


        draw_chess_board(self.screen, self.board, (board_x, board_y), board_size)

        # conflicts / solved text
        conf = conflicts(self.board)
        status = "Solved!" if conf == 0 else f"Conflicts: {conf}"
        st_col = SUCCESS_COLOR if conf == 0 else TEXT_COLOR
        st_surf = font.render(status, True, st_col)
        self.screen.blit(st_surf, (board_x, board_y + board_size + 10))

        # reposition buttons dynamically
        button_y = H - 70
        btn_gap = 20
        bx = 40
        for b in self.buttons:
            b.rect.y = button_y
            b.rect.x = bx
            bx += b.rect.width + btn_gap
            b.draw(self.screen, font)

        # message text
        msg_surf = font.render(self.msg, True, TEXT_COLOR)
        self.screen.blit(msg_surf, (20, H - 110))


    def handle_run_algorithm(self):
        name = ALG_LIST[self.alg_choice]
        self.msg = f"Running {name}..."
        self.draw(); pygame.display.flip()
        # For local methods, pass the current board; for others, pass None
        initial = self.board[:]
        func = ALGORITHMS[name]
        try:
            board, ok, metric, t = func(self.N, initial)
        except Exception as e:
            board, ok, metric, t = None, False, 0, 0.0
        if board:
            self.board = board
        self.msg = f"{name}: {'Success' if ok else 'Failed'} | iters/nodes={metric} | {t:.3f}s"
        return

    def run(self):
        clock = pygame.time.Clock()
        running = True
        while running:
            dt = clock.tick(FPS)/1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mx,my = event.pos
                    # click algorithm list area: cycle selection by clicking the name lines
                    if 20 <= mx <= 180:
                        # determine index by y
                        if 80 <= my <= 80 + 22*len(ALG_LIST):
                            idx = (my - 80) // 22
                            if 0 <= idx < len(ALG_LIST):
                                self.alg_choice = int(idx)
                    # buttons
                    for b in self.buttons:
                        if b.rect.collidepoint(mx,my):
                            if b.text == "Randomize":
                                self.board = random_board(self.N)
                                self.msg = "Board randomized."
                            elif b.text == "Run Algorithm":
                                self.handle_run_algorithm()
                            elif b.text == "Compare":
                                # open compare window
                                CompareGUI(self.N, starting_board=random_board(self.N)).run()
                                # after compare returns, continue
                            elif b.text == "Exit":
                                pygame.quit()
                                sys.exit()
                elif event.type == pygame.MOUSEMOTION:
                    for b in self.buttons:
                        b.check_hover(event.pos)
            self.draw()
            pygame.display.flip()
        pygame.quit()
        sys.exit()

# ---------------- Compare GUI ----------------
class CompareGUI:
    def __init__(self, n, starting_board=None):
        pygame.init()
        self.n = n
        self.start_board = starting_board if starting_board is not None else random_board(n)

        # Window size (keep reasonably large to fit content)
        self.width = max(1000, 1200)
        self.height = max(700, 720)
        # self.screen = pygame.display.set_mode((self.width, self.height))
        self.screen = pygame.display.set_mode((1600, 900), pygame.RESIZABLE)


        pygame.display.set_caption("Compare Algorithms")

        # Fonts (scale slightly with n)
        base_font = 18
        base_large = 24
        self.font = pygame.font.SysFont("arial", max(14, base_font - (n//8)))
        self.large = pygame.font.SysFont("arial", max(18, base_large - (n//8)), bold=True)

        # selection indices and results
        self.a_idx = 0
        self.b_idx = 1 if len(ALG_LIST) > 1 else 0
        self.results = None

        # buttons (positions updated dynamically in draw_ui but keep rects for hover)
        self.compare_btn = Button(self.width // 2 - 80, self.height - 72, 160, 48, "Compare")
        self.back_btn = Button(26, self.height - 72, 110, 48, "Back", (200, 80, 80))

    def run_algorithm(self, name, initial_board):
        func = ALGORITHMS[name]
        try:
            res = func(self.n, initial_board)
        except Exception:
            res = (None, False, 0, 0.0)
        if res is None:
            board, ok, metric, t = None, False, 0, 0.0
        else:
            board, ok, metric, t = res
        conf = conflicts(board) if board else self.n
        eff = (1.0 / (t * (1 + conf))) if t > 0 else (1.0 / (1e-9 * (1 + conf)))
        return {"name": name, "ok": ok, "iters": metric, "time": t, "conf": conf, "eff": eff, "board": copy_board(board)}

    # def get_layout(self):
    #     # margins and sizes computed dynamically to avoid overlap
    #     margin = 36
    #     top_space = 60
    #     list_width = max(260, int(self.width * 0.20))
    #     item_h = 26
    #     list_height = min(int(self.height * 0.28), (len(ALG_LIST) * item_h) + 8)

    #     # initial board size (small preview at top center)
    #     max_init_size = int(self.width * 0.18)
    #     init_size = min(260, max_init_size)

    #     # compute result boards area: remaining height after top lists area
    #     remaining_height = self.height - (top_space + list_height + 120)
    #     # leave bottom area for stats/buttons
    #     max_board_h = max(200, remaining_height)
    #     # board width must fit two boards plus margins
    #     max_side_by_side = (self.width - (margin * 4)) // 2
    #     board_size = min(max_board_h, max_side_by_side)
    #     board_size = max(180, board_size)

    #     # columns
    #     left_col_x = margin
    #     center_x = self.width // 2
    #     right_col_x = self.width - margin - list_width

    #     layout = {
    #         "margin": margin,
    #         "top_space": top_space,
    #         "list_width": list_width,
    #         "item_h": item_h,
    #         "list_height": list_height,
    #         "init_size": init_size,
    #         "board_size": board_size,
    #         "left_col_x": left_col_x,
    #         "center_x": center_x,
    #         "right_col_x": right_col_x,
    #     }
    #     return layout
    def get_layout(self):
        W, H = self.screen.get_size()
        margin = 40
        top_space = 70
        list_width = max(260, int(W * 0.2))
        item_h = 26
        list_height = min(int(H * 0.25), (len(ALG_LIST) * item_h) + 8)

        # --- Slightly enlarge boards while keeping alignment ---
        init_size = int(min(W * 0.55, H * 0.37))       # ↑ bigger initial board (was 0.18)
        board_size = int(min(W * 0.38, H * 0.50))      # ↑ bigger result boards (was 0.33/0.45)

        left_col_x = margin
        center_x = W // 2
        right_col_x = W - margin - list_width

        layout = {
            "margin": margin,
            "top_space": top_space,
            "list_width": list_width,
            "item_h": item_h,
            "list_height": list_height,
            "init_size": init_size,
            "board_size": board_size,
            "left_col_x": left_col_x,
            "center_x": center_x,
            "right_col_x": right_col_x,
            "width": W,
            "height": H
        }
        return layout


    def draw_ui(self):
        L = self.get_layout()
        self.screen.fill(BG_COLOR)

        # Title
        title = self.large.render("Algorithm Comparison", True, (220, 220, 220))
        self.screen.blit(title, (L["margin"], 12))

        # --- Top area ---
        top_y = L["top_space"]

        # Initial board centered and larger
        init_label = self.font.render("Initial board (randomized):", True, TEXT_COLOR)
        init_x = L["center_x"] - L["init_size"] // 2
        init_y = top_y - 50
        self.screen.blit(init_label, (init_x, init_y))
        draw_chess_board(self.screen, self.start_board, (init_x, init_y + 40), L["init_size"])

        # --- Algorithm lists ---
        left_x = L["left_col_x"]
        right_x = L["right_col_x"]
        list_y = top_y + 28

        self.screen.blit(self.font.render("Algorithm A:", True, TEXT_COLOR), (left_x, top_y))
        self.screen.blit(self.font.render("Algorithm B:", True, TEXT_COLOR), (right_x, top_y))

        for i, name in enumerate(ALG_LIST):
            # Left list
            r = pygame.Rect(left_x, list_y + i * L["item_h"], L["list_width"], L["item_h"] - 2)
            color = (80, 130, 80) if i == self.a_idx else (70, 70, 70)
            pygame.draw.rect(self.screen, color, r, border_radius=4)
            txt = self.font.render(name, True, (240, 240, 240))
            self.screen.blit(txt, (r.x + 6, r.y + 3))

            # Right list
            rr = pygame.Rect(right_x, list_y + i * L["item_h"], L["list_width"], L["item_h"] - 2)
            color = (130, 80, 80) if i == self.b_idx else (70, 70, 70)
            pygame.draw.rect(self.screen, color, rr, border_radius=4)
            txt = self.font.render(name, True, (240, 240, 240))
            self.screen.blit(txt, (rr.x + 6, rr.y + 3))

        # --- Divider ---
        divider_y = list_y + len(ALG_LIST) * L["item_h"] + 90
        pygame.draw.line(self.screen, (100, 100, 100), (L["margin"], divider_y), (L["width"] - L["margin"], divider_y), 1)

        # --- Compare boards ---
        bs = L["board_size"]
        gap = 80  # space between two result boards
        total_w = (bs * 2) + gap
        left_board_x = (L["width"] - total_w) // 2
        right_board_x = left_board_x + bs + gap
        board_y = divider_y + 30

        # Draw labels
        self.screen.blit(self.font.render(ALG_LIST[self.a_idx], True, TEXT_COLOR), (left_board_x, board_y - 24))
        self.screen.blit(self.font.render(ALG_LIST[self.b_idx], True, TEXT_COLOR), (right_board_x, board_y - 24))

        # Draw boards
        if self.results:
            left_res, right_res = self.results
            left_board = left_res["board"] if (left_res["board"] and left_res["ok"]) else self.start_board
            right_board = right_res["board"] if (right_res["board"] and right_res["ok"]) else self.start_board
        else:
            left_board = right_board = self.start_board

        draw_chess_board(self.screen, left_board, (left_board_x, board_y), bs)
        draw_chess_board(self.screen, right_board, (right_board_x, board_y), bs)

        # --- Stats under each board ---
        stats_y = board_y + bs + 12
        stats_gap = 22
        if self.results:
            for idx, res in enumerate(self.results):
                xbase = left_board_x if idx == 0 else right_board_x
                col = SUCCESS_COLOR if res["ok"] else ERROR_COLOR
                lines = [
                    f"Success: {'Yes' if res['ok'] else 'No'}",
                    f"Iterations/Nodes: {res['iters']}",
                    f"Time: {res['time']:.3f}s",
                    f"Conflicts: {res['conf']}",
                    f"Efficiency: {res['eff']:.6f}"
                ]
                for i, ln in enumerate(lines):
                    surf = self.font.render(ln, True, col)
                    self.screen.blit(surf, (xbase, stats_y + i * stats_gap))

        # --- Buttons ---
        gap = 12
        self.compare_btn.rect.topleft = (L["margin"], L["height"] - 72 - self.compare_btn.rect.height - gap)
        self.back_btn.rect.topleft = (L["margin"], L["height"] - 72)
        self.compare_btn.draw(self.screen, self.font)
        self.back_btn.draw(self.screen, self.font)

        # Hint
        hint = self.font.render("Pick two algorithms and press Compare.", True, TEXT_COLOR)
        hint_rect = hint.get_rect(center=(L["width"] // 2, L["height"] - 26))
        self.screen.blit(hint, hint_rect)

    def show_chart(self):
        if not self.results:
            return
        left, right = self.results
        labels = ["Time (s)", "Iterations", "Efficiency"]
        a_vals = [left["time"], left["iters"], left["eff"]]
        b_vals = [right["time"], right["iters"], right["eff"]]
        x = np.arange(len(labels))
        width = 0.35
        fig, ax = plt.subplots(figsize=(8, 4))
        # Optional: scale time so bars are visible
        time_scale = max(left["time"], right["time"], 0.001)
        if time_scale > 5:  # if times are large, convert to seconds directly
            pass
        else:
            # scale up time for better visual comparison (no effect on labels)
            a_vals[0] *= 3000
            b_vals[0] *= 3000
            labels[0] = "Time (ms)"

        ax.bar(x - width / 2, a_vals, width, label=left["name"])
        ax.bar(x + width / 2, b_vals, width, label=right["name"])
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.set_title("Algorithm Comparison")
        plt.tight_layout()
        plt.show()

    def run(self):
        clock = pygame.time.Clock()
        running = True
        while running:
            clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = event.pos
                    L = self.get_layout()
                    top_y = L["top_space"] + 28

                    # --- handle selections ---
                    left_x = L["left_col_x"]
                    if left_x <= mx <= left_x + L["list_width"] and top_y <= my <= top_y + len(ALG_LIST) * L["item_h"]:
                        idx = (my - top_y) // L["item_h"]
                        if 0 <= idx < len(ALG_LIST):
                            self.a_idx = int(idx)
                            if self.a_idx == self.b_idx:
                                self.b_idx = (self.a_idx + 1) % len(ALG_LIST)

                    right_x = L["right_col_x"]
                    if right_x <= mx <= right_x + L["list_width"] and top_y <= my <= top_y + len(ALG_LIST) * L["item_h"]:
                        idx = (my - top_y) // L["item_h"]
                        if 0 <= idx < len(ALG_LIST):
                            self.b_idx = int(idx)
                            if self.b_idx == self.a_idx:
                                self.a_idx = (self.b_idx + 1) % len(ALG_LIST)

                    # --- Compare clicked ---
                    if self.compare_btn.rect.collidepoint(mx, my):
                        if self.a_idx != self.b_idx:
                            a_name = ALG_LIST[self.a_idx]
                            b_name = ALG_LIST[self.b_idx]

                            # use SAME randomized board for both
                            init = copy_board(self.start_board)

                            left_res = self.run_algorithm(a_name, init)
                            right_res = self.run_algorithm(b_name, init)

                            # fallback only if algorithm completely fails
                            if left_res["board"] is None:
                                left_res["board"] = copy_board(self.start_board)
                            if right_res["board"] is None:
                                right_res["board"] = copy_board(self.start_board)

                            left_res["name"] = a_name
                            right_res["name"] = b_name
                            self.results = [left_res, right_res]
                            

                            # --- draw results first ---
                            self.draw_ui()
                            pygame.display.flip()
                            pygame.time.delay(500)  # short pause for user to see results

                            # --- then show chart ---
                            self.show_chart()

                    # --- Back clicked ---
                    if self.back_btn.rect.collidepoint(mx, my):
                        running = False

                elif event.type == pygame.MOUSEMOTION:
                    self.compare_btn.check_hover(event.pos)
                    self.back_btn.check_hover(event.pos)

            self.draw_ui()
            pygame.display.flip()
        return


# ---------------- N input GUI ----------------
def get_n_from_gui():
    pygame.init()
    font = pygame.font.SysFont(None, 40)
    small = pygame.font.SysFont(None, 26)
    screen = pygame.display.set_mode((520,320))
    pygame.display.set_caption("N-Queens - Enter N")
    input_box = pygame.Rect(160, 120, 200, 48)
    color_inactive = pygame.Color("lightskyblue3")
    color_active = pygame.Color("dodgerblue2")
    active = False
    text = "8"
    message = ""
    start_btn = pygame.Rect(210, 200, 100, 40)
    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if input_box.collidepoint(event.pos):
                    active = True
                else:
                    active = False
                if start_btn.collidepoint(event.pos):
                    if text.isdigit() and int(text) >= 4:
                        return int(text)
                    else:
                        message = "Enter integer >= 4"
            elif event.type == pygame.KEYDOWN and active:
                if event.key == pygame.K_RETURN:
                    if text.isdigit() and int(text) >= 4:
                        return int(text)
                    else:
                        message = "Enter integer >= 4"
                elif event.key == pygame.K_BACKSPACE:
                    text = text[:-1]
                else:
                    if event.unicode.isdigit():
                        text += event.unicode

        screen.fill((30,30,30))
        title = font.render("Enter N (Number of Queens):", True, (240,240,240))
        screen.blit(title, (60, 40))
        # input box
        col = color_active if active else color_inactive
        pygame.draw.rect(screen, col, input_box, 2)
        txtsurf = font.render(text, True, (255,255,255))
        screen.blit(txtsurf, (input_box.x+10, input_box.y+6))
        # start button
        pygame.draw.rect(screen, (0,150,0), start_btn)
        screen.blit(small.render("Start", True, (255,255,255)), (start_btn.x+18, start_btn.y+8))
        if message:
            screen.blit(small.render(message, True, (240,80,80)), (150, 265))

        pygame.display.flip()
        clock.tick(30)

# ---------------- Main ----------------
if __name__ == "__main__":
    n = get_n_from_gui()
    gui = QueensGUI(N=n)
    gui.run()
