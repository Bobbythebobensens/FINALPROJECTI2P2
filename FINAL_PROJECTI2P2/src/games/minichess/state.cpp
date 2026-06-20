#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>

#include "../../policy/game_history.hpp"
#include "./state.hpp"
#include "config.hpp"

/*============================================================
 * KP (King-Piece) Evaluation tables
 *
 * Always compiled. Toggled at runtime via use_kp_eval param.
 *============================================================*/

// KP material (10x scale for fine positional granularity)
static const int kp_material[7] = {0, 110, 500, 320, 290, 900, 20000};

// Material-only (simple scale)
static const int simple_material[7] = {0, 11, 50, 32, 29, 90, 2000};

// Piece-Square Tables (white perspective, mirror for black)
static const int pst[6][BOARD_H][BOARD_W] = {
    // Pawn
    {{0, 0, 0, 0, 0},
     {15, 15, 15, 15, 15},
     {4, 6, 10, 6, 4},
     {2, 4, 6, 4, 2},
     {0, 2, 2, 2, 0},
     {0, 0, 0, 0, 0}},
    // Rook
    {{2, 2, 2, 2, 2},
     {4, 4, 4, 4, 4},
     {0, 0, 2, 0, 0},
     {0, 0, 2, 0, 0},
     {0, 0, 2, 0, 0},
     {0, 0, 0, 0, 0}},
    // Knight
    {{-4, -2, 0, -2, -4},
     {-2, 2, 4, 2, -2},
     {0, 4, 6, 4, 0},
     {0, 4, 6, 4, 0},
     {-2, 2, 4, 2, -2},
     {-4, -2, 0, -2, -4}},
    // Bishop
    {{-2, 0, 0, 0, -2},
     {0, 3, 4, 3, 0},
     {0, 4, 4, 4, 0},
     {0, 4, 4, 4, 0},
     {0, 3, 4, 3, 0},
     {-2, 0, 0, 0, -2}},
    // Queen
    {{-2, 0, 2, 0, -2},
     {0, 2, 4, 2, 0},
     {0, 4, 6, 4, 0},
     {0, 4, 6, 4, 0},
     {0, 2, 4, 2, 0},
     {-2, 0, 2, 0, -2}},
    // King
    {{-8, -8, -8, -8, -8},
     {-4, -4, -4, -4, -4},
     {-4, -4, -4, -4, -4},
     {-4, -4, -4, -4, -4},
     {4, 4, 0, 4, 4},
     {6, 6, 2, 6, 6}},
};

// King tropism weights
static const int tropism_w[7] = {0, 0, 3, 3, 2, 5, 0};

static int king_tropism(int piece_type, int pr, int pc, int ekr, int ekc) {
  int dist = std::max(std::abs(pr - ekr), std::abs(pc - ekc));
  if (dist <= 2) {
    return tropism_w[piece_type] * (3 - dist);
  }
  return 0;
}

/*============================================================
 * Move tables, lookup tables & helper functions
 *============================================================*/

static const int move_table_rook_bishop[8][7][2] = {
    {{0, 1}, {0, 2}, {0, 3}, {0, 4}, {0, 5}, {0, 6}, {0, 7}},
    {{0, -1}, {0, -2}, {0, -3}, {0, -4}, {0, -5}, {0, -6}, {0, -7}},
    {{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}},
    {{-1, 0}, {-2, 0}, {-3, 0}, {-4, 0}, {-5, 0}, {-6, 0}, {-7, 0}},
    {{1, 1}, {2, 2}, {3, 3}, {4, 4}, {5, 5}, {6, 6}, {7, 7}},
    {{1, -1}, {2, -2}, {3, -3}, {4, -4}, {5, -5}, {6, -6}, {7, -7}},
    {{-1, 1}, {-2, 2}, {-3, 3}, {-4, 4}, {-5, 5}, {-6, 6}, {-7, 7}},
    {{-1, -1}, {-2, -2}, {-3, -3}, {-4, -4}, {-5, -5}, {-6, -6}, {-7, -7}},
};

static const int move_table_knight[8][2] = {
    {1, 2}, {1, -2}, {-1, 2}, {-1, -2}, {2, 1}, {2, -1}, {-2, 1}, {-2, -1},
};

static const int move_table_king[8][2] = {
    {1, 0}, {0, 1}, {-1, 0}, {0, -1}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1},
};

// King PST for endgame (white perspective, mirror for black)
static const int pst_king_endgame[BOARD_H][BOARD_W] = {
    {-8, -6, -6, -6, -8}, {-4, 2, 4, 2, -4}, {0, 4, 6, 4, 0},
    {0, 4, 6, 4, 0},      {-4, 2, 4, 2, -4}, {-8, -6, -6, -6, -8},
};

/* Bitboard helper macros and structures placed at the top for evaluation
 * performance */
#define BB_SQ(r, c) ((r) * BOARD_W + (c))
#define BB_ROW(sq) ((sq) / BOARD_W)
#define BB_COL(sq) ((sq) % BOARD_W)

static uint32_t bb_knight[30];       // knight attack mask per square
static uint32_t bb_king[30];         // king attack mask per square
static uint32_t bb_pawn_push[2][30]; // pawn push target per player/square
static uint32_t bb_pawn_cap[2][30];  // pawn capture targets per player/square
static bool bb_ready = false;

static const int bb_dr[8] = {0, 0, 1, -1, 1, 1, -1, -1};
static const int bb_dc[8] = {1, -1, 0, 0, 1, -1, 1, -1};

static void bb_init() {
  static const int kn_dr[8] = {1, 1, -1, -1, 2, 2, -2, -2};
  static const int kn_dc[8] = {2, -2, 2, -2, 1, -1, 1, -1};
  static const int ki_dr[8] = {1, 0, -1, 0, 1, 1, -1, -1};
  static const int ki_dc[8] = {0, 1, 0, -1, 1, -1, 1, -1};

  for (int r = 0; r < BOARD_H; r++) {
    for (int c = 0; c < BOARD_W; c++) {
      int sq = BB_SQ(r, c);

      // Knight
      bb_knight[sq] = 0;
      for (int d = 0; d < 8; d++) {
        int nr = r + kn_dr[d], nc = c + kn_dc[d];
        if (nr >= 0 && nr < BOARD_H && nc >= 0 && nc < BOARD_W) {
          bb_knight[sq] |= 1u << BB_SQ(nr, nc);
        }
      }

      // King
      bb_king[sq] = 0;
      for (int d = 0; d < 8; d++) {
        int nr = r + ki_dr[d], nc = c + ki_dc[d];
        if (nr >= 0 && nr < BOARD_H && nc >= 0 && nc < BOARD_W) {
          bb_king[sq] |= 1u << BB_SQ(nr, nc);
        }
      }

      // Pawn (player 0 = white, advances up = row-1)
      bb_pawn_push[0][sq] = 0;
      bb_pawn_cap[0][sq] = 0;
      if (r > 0) {
        bb_pawn_push[0][sq] = 1u << BB_SQ(r - 1, c);
        if (c > 0) {
          bb_pawn_cap[0][sq] |= 1u << BB_SQ(r - 1, c - 1);
        }
        if (c < BOARD_W - 1) {
          bb_pawn_cap[0][sq] |= 1u << BB_SQ(r - 1, c + 1);
        }
      }

      // Pawn (player 1 = black, advances down = row+1)
      bb_pawn_push[1][sq] = 0;
      bb_pawn_cap[1][sq] = 0;
      if (r < BOARD_H - 1) {
        bb_pawn_push[1][sq] = 1u << BB_SQ(r + 1, c);
        if (c > 0) {
          bb_pawn_cap[1][sq] |= 1u << BB_SQ(r + 1, c - 1);
        }
        if (c < BOARD_W - 1) {
          bb_pawn_cap[1][sq] |= 1u << BB_SQ(r + 1, c + 1);
        }
      }
    }
  }
  bb_ready = true;
}

/*============================================================
 * evaluate() — runtime-selectable eval strategy
 *============================================================*/

int State::evaluate(bool use_kp_eval, bool use_mobility,
                    const GameHistory *history) {
  (void)history; // just to suppress warning

  // [ Hackathon TODO 1-1 ]
  // if in win state, return max score(you can check base_state.hpp for max
  // score)
  if (this->game_state == WIN) {
    return P_MAX;
  }

  auto self_board = this->board.board[this->player];
  auto oppn_board = this->board.board[1 - this->player];
  int mg_self_score = 0, mg_oppn_score = 0;
  int eg_self_score = 0, eg_oppn_score = 0;

  // Search for Kings, count pawns, and determine if it's an endgame phase
  int self_kr = -1, self_kc = -1;
  int oppn_kr = -1, oppn_kc = -1;
  int self_pawns_on_file[BOARD_W] = {0};
  int oppn_pawns_on_file[BOARD_W] = {0};
  int self_queens = 0, oppn_queens = 0;
  int self_rooks = 0, oppn_rooks = 0;
  int self_knights = 0, oppn_knights = 0;
  int self_bishops = 0, oppn_bishops = 0;

  int self_pawn_count = 0;
  Point self_pawns[5];
  int oppn_pawn_count = 0;
  Point oppn_pawns[5];

  uint32_t self_occ = 0;
  uint32_t oppn_occ = 0;

  for (int r = 0; r < BOARD_H; r++) {
    for (int c = 0; c < BOARD_W; c++) {
      int p0 = self_board[r][c];
      if (p0) {
        int sq = BB_SQ(r, c);
        self_occ |= 1u << sq;
        if (p0 == 6) {
          self_kr = r;
          self_kc = c;
        } else if (p0 == 1) {
          self_pawns_on_file[c]++;
          if (self_pawn_count < 5) {
            self_pawns[self_pawn_count++] = {r, c};
          }
        } else if (p0 == 5) {
          self_queens++;
        } else if (p0 == 2) {
          self_rooks++;
        } else if (p0 == 3) {
          self_knights++;
        } else if (p0 == 4) {
          self_bishops++;
        }
      }

      int p1 = oppn_board[r][c];
      if (p1) {
        int sq = BB_SQ(r, c);
        oppn_occ |= 1u << sq;
        if (p1 == 6) {
          oppn_kr = r;
          oppn_kc = c;
        } else if (p1 == 1) {
          oppn_pawns_on_file[c]++;
          if (oppn_pawn_count < 5) {
            oppn_pawns[oppn_pawn_count++] = {r, c};
          }
        } else if (p1 == 5) {
          oppn_queens++;
        } else if (p1 == 2) {
          oppn_rooks++;
        } else if (p1 == 3) {
          oppn_knights++;
        } else if (p1 == 4) {
          oppn_bishops++;
        }
      }
    }
  }

  int current_phase =
      (self_queens + oppn_queens) * 4 + (self_rooks + oppn_rooks) * 2 +
      (self_knights + oppn_knights) * 1 + (self_bishops + oppn_bishops) * 1;
  if (current_phase > 16) {
    current_phase = 16;
  }

  if (use_kp_eval) {
    /* === KP eval: material + PST + tropism === */
    for (int r = 0; r < BOARD_H; r++) {
      for (int c = 0; c < BOARD_W; c++) {
        int self_piece = self_board[r][c];
        if (self_piece) {
          int mat = kp_material[self_piece];
          mg_self_score += mat;
          eg_self_score += mat;
          int self_pst_r = (this->player == 0) ? r : (BOARD_H - 1 - r);
          if (self_piece == 6) {
            mg_self_score += 5 * pst[5][self_pst_r][c];
            eg_self_score += 5 * pst_king_endgame[self_pst_r][c];
          } else {
            int pst_val = 5 * pst[self_piece - 1][self_pst_r][c];
            mg_self_score += pst_val;
            eg_self_score += pst_val;
          }
          if (oppn_kr >= 0) {
            int trop = 5 * king_tropism(self_piece, r, c, oppn_kr, oppn_kc);
            mg_self_score += trop;
            eg_self_score += trop;
          }
        }
        int oppn_piece = oppn_board[r][c];
        if (oppn_piece) {
          int mat = kp_material[oppn_piece];
          mg_oppn_score += mat;
          eg_oppn_score += mat;
          int oppn_pst_r = (this->player == 0) ? (BOARD_H - 1 - r) : r;
          if (oppn_piece == 6) {
            mg_oppn_score += 5 * pst[5][oppn_pst_r][c];
            eg_oppn_score += 5 * pst_king_endgame[oppn_pst_r][c];
          } else {
            int pst_val = 5 * pst[oppn_piece - 1][oppn_pst_r][c];
            mg_oppn_score += pst_val;
            eg_oppn_score += pst_val;
          }
          if (self_kr >= 0) {
            int trop = 5 * king_tropism(oppn_piece, r, c, self_kr, self_kc);
            mg_oppn_score += trop;
            eg_oppn_score += trop;
          }
        }
      }
    }
  } else {
    /* === Simple material-only eval === */
    for (int r = 0; r < BOARD_H; r++) {
      for (int c = 0; c < BOARD_W; c++) {
        int self_piece = self_board[r][c];
        if (self_piece) {
          mg_self_score += simple_material[self_piece];
          eg_self_score += simple_material[self_piece];
        }
        int oppn_piece = oppn_board[r][c];
        if (oppn_piece) {
          mg_oppn_score += simple_material[oppn_piece];
          eg_oppn_score += simple_material[oppn_piece];
        }
      }
    }
  }

  // Positional / structural evaluations (accumulated in centipawns)
  int mg_self_centipawns = 0, mg_oppn_centipawns = 0;
  int eg_self_centipawns = 0, eg_oppn_centipawns = 0;

  // Self positional evaluations
  for (int i = 0; i < self_pawn_count; i++) {
    int r = self_pawns[i].first;
    int c = self_pawns[i].second;

    // Passed pawn check
    bool passed = true;
    for (int k = 0; k < oppn_pawn_count; ++k) {
      int opp_r = oppn_pawns[k].first;
      int opp_c = oppn_pawns[k].second;
      if (std::abs(opp_c - c) <= 1) {
        if (this->player == 0 && opp_r < r) { // White target is 0
          passed = false;
          break;
        }
        if (this->player == 1 && opp_r > r) { // Black target is 5
          passed = false;
          break;
        }
      }
    }
    if (passed) {
      int dist = (this->player == 0) ? r : (BOARD_H - 1 - r);
      int bonus = 0;
      if (dist == 1)
        bonus = 150;
      else if (dist == 2)
        bonus = 80;
      else if (dist == 3)
        bonus = 40;
      else if (dist == 4)
        bonus = 20;
      mg_self_centipawns += bonus;
      eg_self_centipawns += bonus;
    }

    // Connected pawn check
    bool connected = false;
    for (int j = 0; j < self_pawn_count; ++j) {
      if (i == j)
        continue;
      if (std::abs((int)self_pawns[j].second - c) == 1 &&
          std::abs((int)self_pawns[j].first - r) <= 1) {
        connected = true;
        break;
      }
    }
    if (connected) {
      mg_self_centipawns += 15;
      eg_self_centipawns += 15;
    }

    // Advanced & safe pawn check
    int dist = (this->player == 0) ? r : (BOARD_H - 1 - r);
    if (dist <= 3) {
      int front_r = (this->player == 0) ? (r - 1) : (r + 1);
      bool blocked = false;
      if (front_r >= 0 && front_r < BOARD_H) {
        if (self_board[front_r][c] || oppn_board[front_r][c]) {
          blocked = true;
        }
      } else {
        blocked = true;
      }

      if (!blocked) {
        int oppn = 1 - this->player;
        bool attacked = false;

        // 1. Current square (r, c) under attack by opponent pawn
        int oppn_r_curr = (oppn == 0) ? (r + 1) : (r - 1);
        if (oppn_r_curr >= 0 && oppn_r_curr < BOARD_H) {
          if (c > 0 && oppn_board[oppn_r_curr][c - 1] == 1)
            attacked = true;
          if (c < BOARD_W - 1 && oppn_board[oppn_r_curr][c + 1] == 1)
            attacked = true;
        }

        // 2. Front square (front_r, c) under attack by opponent pawn
        if (!attacked && front_r >= 0 && front_r < BOARD_H) {
          int oppn_r_front = (oppn == 0) ? (front_r + 1) : (front_r - 1);
          if (oppn_r_front >= 0 && oppn_r_front < BOARD_H) {
            if (c > 0 && oppn_board[oppn_r_front][c - 1] == 1)
              attacked = true;
            if (c < BOARD_W - 1 && oppn_board[oppn_r_front][c + 1] == 1)
              attacked = true;
          }
        }

        if (!attacked) {
          int bonus = 0;
          if (dist == 3)
            bonus = 15;
          else if (dist == 2)
            bonus = 30;
          else if (dist == 1)
            bonus = 60;
          mg_self_centipawns += bonus;
          eg_self_centipawns += bonus;
        }
      }
    }

    // King proximity to friendly pawns
    if (self_kr >= 0) {
      int k_dist = std::max(std::abs(self_kr - r), std::abs(self_kc - c));
      if (k_dist == 1)
        eg_self_centipawns += 10;
      else if (k_dist == 2)
        eg_self_centipawns += 5;
    }
  }

  // Opponent positional evaluations
  int oppn_player = 1 - this->player;
  for (int i = 0; i < oppn_pawn_count; i++) {
    int r = oppn_pawns[i].first;
    int c = oppn_pawns[i].second;

    // Passed pawn check
    bool passed = true;
    for (int k = 0; k < self_pawn_count; ++k) {
      int self_r = self_pawns[k].first;
      int self_c = self_pawns[k].second;
      if (std::abs(self_c - c) <= 1) {
        if (oppn_player == 0 && self_r < r) {
          passed = false;
          break;
        }
        if (oppn_player == 1 && self_r > r) {
          passed = false;
          break;
        }
      }
    }
    if (passed) {
      int dist = (oppn_player == 0) ? r : (BOARD_H - 1 - r);
      int bonus = 0;
      if (dist == 1)
        bonus = 150;
      else if (dist == 2)
        bonus = 80;
      else if (dist == 3)
        bonus = 40;
      else if (dist == 4)
        bonus = 20;
      mg_oppn_centipawns += bonus;
      eg_oppn_centipawns += bonus;
    }

    // Connected pawn check
    bool connected = false;
    for (int j = 0; j < oppn_pawn_count; ++j) {
      if (i == j)
        continue;
      if (std::abs((int)oppn_pawns[j].second - c) == 1 &&
          std::abs((int)oppn_pawns[j].first - r) <= 1) {
        connected = true;
        break;
      }
    }
    if (connected) {
      mg_oppn_centipawns += 15;
      eg_oppn_centipawns += 15;
    }

    // Advanced & safe pawn check
    int dist = (oppn_player == 0) ? r : (BOARD_H - 1 - r);
    if (dist <= 3) {
      int front_r = (oppn_player == 0) ? (r - 1) : (r + 1);
      bool blocked = false;
      if (front_r >= 0 && front_r < BOARD_H) {
        if (self_board[front_r][c] || oppn_board[front_r][c]) {
          blocked = true;
        }
      } else {
        blocked = true;
      }

      if (!blocked) {
        int oppn = this->player;
        bool attacked = false;

        // 1. Current square (r, c) under attack by opponent pawn
        int oppn_r_curr = (oppn == 0) ? (r + 1) : (r - 1);
        if (oppn_r_curr >= 0 && oppn_r_curr < BOARD_H) {
          if (c > 0 && self_board[oppn_r_curr][c - 1] == 1)
            attacked = true;
          if (c < BOARD_W - 1 && self_board[oppn_r_curr][c + 1] == 1)
            attacked = true;
        }

        // 2. Front square (front_r, c) under attack by opponent pawn
        if (!attacked && front_r >= 0 && front_r < BOARD_H) {
          int oppn_r_front = (oppn == 0) ? (front_r + 1) : (front_r - 1);
          if (oppn_r_front >= 0 && oppn_r_front < BOARD_H) {
            if (c > 0 && self_board[oppn_r_front][c - 1] == 1)
              attacked = true;
            if (c < BOARD_W - 1 && self_board[oppn_r_front][c + 1] == 1)
              attacked = true;
          }
        }

        if (!attacked) {
          int bonus = 0;
          if (dist == 3)
            bonus = 15;
          else if (dist == 2)
            bonus = 30;
          else if (dist == 1)
            bonus = 60;
          mg_oppn_centipawns += bonus;
          eg_oppn_centipawns += bonus;
        }
      }
    }

    // King proximity to friendly pawns
    if (oppn_kr >= 0) {
      int k_dist = std::max(std::abs(oppn_kr - r), std::abs(oppn_kc - c));
      if (k_dist == 1)
        eg_oppn_centipawns += 10;
      else if (k_dist == 2)
        eg_oppn_centipawns += 5;
    }
  }

  // Bishop, Knight, Rook Mobility
  if (!bb_ready) {
    bb_init();
  }

  for (int r = 0; r < BOARD_H; r++) {
    for (int c = 0; c < BOARD_W; c++) {
      int self_piece = self_board[r][c];
      if (self_piece == 2 || self_piece == 4) { // Rook, Bishop
        int count = 0;
        int st = (self_piece == 2) ? 0 : 4;
        int end = (self_piece == 2) ? 4 : 8;
        for (int part = st; part < end; ++part) {
          int dr = bb_dr[part];
          int dc = bb_dc[part];
          int nr = r, nc = c;
          for (int k = 1; k < 8; ++k) {
            nr += dr;
            nc += dc;
            if (nr < 0 || nr >= BOARD_H || nc < 0 || nc >= BOARD_W) {
              break;
            }
            int target_sq = BB_SQ(nr, nc);
            uint32_t mask = 1u << target_sq;
            if (self_occ & mask) {
              break;
            }
            count++;
            if (oppn_occ & mask) {
              break;
            }
          }
        }
        mg_self_centipawns += count * 3;
        eg_self_centipawns += count * 3;
        if (self_piece == 2) { // Rook
          eg_self_centipawns += count * 5;
        } else { // Bishop
          eg_self_centipawns += count * 2;
        }
      } else if (self_piece == 3) { // Knight
        int sq = BB_SQ(r, c);
        int count = __builtin_popcount(bb_knight[sq] & ~self_occ);
        mg_self_centipawns += count * 3;
        eg_self_centipawns += count * 3;
        eg_self_centipawns += count * 2;

        // Knight outpost heuristic (permanent: both mg and eg)
        if ((r == 2 || r == 3) && (c == 1 || c == 2 || c == 3)) {
          bool protected_by_pawn = false;
          if (this->player == 0) { // White
            int pawn_r = r + 1;
            if (pawn_r < BOARD_H) {
              if (c > 0 && self_board[pawn_r][c - 1] == 1)
                protected_by_pawn = true;
              if (c < BOARD_W - 1 && self_board[pawn_r][c + 1] == 1)
                protected_by_pawn = true;
            }
          } else { // Black
            int pawn_r = r - 1;
            if (pawn_r >= 0) {
              if (c > 0 && self_board[pawn_r][c - 1] == 1)
                protected_by_pawn = true;
              if (c < BOARD_W - 1 && self_board[pawn_r][c + 1] == 1)
                protected_by_pawn = true;
            }
          }
          if (protected_by_pawn) {
            mg_self_centipawns += 35;
            eg_self_centipawns += 35;
          }
        }
      }

      int oppn_piece = oppn_board[r][c];
      if (oppn_piece == 2 || oppn_piece == 4) { // Rook, Bishop
        int count = 0;
        int st = (oppn_piece == 2) ? 0 : 4;
        int end = (oppn_piece == 2) ? 4 : 8;
        for (int part = st; part < end; ++part) {
          int dr = bb_dr[part];
          int dc = bb_dc[part];
          int nr = r, nc = c;
          for (int k = 1; k < 8; ++k) {
            nr += dr;
            nc += dc;
            if (nr < 0 || nr >= BOARD_H || nc < 0 || nc >= BOARD_W) {
              break;
            }
            int target_sq = BB_SQ(nr, nc);
            uint32_t mask = 1u << target_sq;
            if (oppn_occ & mask) {
              break;
            }
            count++;
            if (self_occ & mask) {
              break;
            }
          }
        }
        mg_oppn_centipawns += count * 3;
        eg_oppn_centipawns += count * 3;
        if (oppn_piece == 2) { // Rook
          eg_oppn_centipawns += count * 5;
        } else { // Bishop
          eg_oppn_centipawns += count * 2;
        }
      } else if (oppn_piece == 3) { // Knight
        int sq = BB_SQ(r, c);
        int count = __builtin_popcount(bb_knight[sq] & ~oppn_occ);
        mg_oppn_centipawns += count * 3;
        eg_oppn_centipawns += count * 3;
        eg_oppn_centipawns += count * 2;

        // Knight outpost heuristic (permanent: both mg and eg)
        if ((r == 2 || r == 3) && (c == 1 || c == 2 || c == 3)) {
          bool protected_by_pawn = false;
          if (oppn_player == 0) { // Opponent White
            int pawn_r = r + 1;
            if (pawn_r < BOARD_H) {
              if (c > 0 && oppn_board[pawn_r][c - 1] == 1)
                protected_by_pawn = true;
              if (c < BOARD_W - 1 && oppn_board[pawn_r][c + 1] == 1)
                protected_by_pawn = true;
            }
          } else { // Opponent Black
            int pawn_r = r - 1;
            if (pawn_r >= 0) {
              if (c > 0 && oppn_board[pawn_r][c - 1] == 1)
                protected_by_pawn = true;
              if (c < BOARD_W - 1 && oppn_board[pawn_r][c + 1] == 1)
                protected_by_pawn = true;
            }
          }
          if (protected_by_pawn) {
            mg_oppn_centipawns += 35;
            eg_oppn_centipawns += 35;
          }
        }
      }
    }
  }

  int mg_positional_diff = 0;
  int eg_positional_diff = 0;
  if (use_kp_eval) {
    mg_positional_diff = mg_self_centipawns - mg_oppn_centipawns;
    eg_positional_diff = eg_self_centipawns - eg_oppn_centipawns;
  } else {
    mg_positional_diff = (mg_self_centipawns - mg_oppn_centipawns) / 10;
    eg_positional_diff = (eg_self_centipawns - eg_oppn_centipawns) / 10;
  }

  int mg_score = (mg_self_score - mg_oppn_score) + mg_positional_diff;
  int eg_score = (eg_self_score - eg_oppn_score) + eg_positional_diff;

  int final_eval =
      ((mg_score * current_phase) + (eg_score * (16 - current_phase))) / 16;

  int bonus = 0;

  /* === Mobility bonus === */
  if (use_mobility) {
    int self_mobility = (int)this->legal_actions.size();
    State opp_state = *this;
    opp_state.player = 1 - this->player;
    opp_state.legal_actions.clear();
    opp_state.game_state = UNKNOWN;
    opp_state.get_legal_actions();
    int oppn_mobility = (int)opp_state.legal_actions.size();
    bonus += 2 * (self_mobility - oppn_mobility);
  }

  return final_eval + bonus;
}

/*============================================================
 * Zobrist hash for transposition table
 *============================================================*/
static uint64_t zobrist_piece[2][7][BOARD_H][BOARD_W];
static uint64_t zobrist_side;
static bool zobrist_ready = false;

static void init_zobrist() {
  uint64_t s = 0x7A35C9D1E4F02B68ULL;
  auto rand64 = [&s]() -> uint64_t {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
  };
  for (int p = 0; p < 2; p++) {
    for (int t = 0; t < 7; t++) {
      for (int r = 0; r < BOARD_H; r++) {
        for (int c = 0; c < BOARD_W; c++) {
          zobrist_piece[p][t][r][c] = rand64();
        }
      }
    }
  }
  zobrist_side = rand64();
  zobrist_ready = true;
}

uint64_t State::compute_hash_full() const {
  if (!zobrist_ready) {
    init_zobrist();
  }
  uint64_t h = 0;
  for (int p = 0; p < 2; p++) {
    for (int r = 0; r < BOARD_H; r++) {
      for (int c = 0; c < BOARD_W; c++) {
        int piece = this->board.board[p][r][c];
        if (piece) {
          h ^= zobrist_piece[p][piece][r][c];
        }
      }
    }
  }
  if (this->player) {
    h ^= zobrist_side;
  }
  return h;
}

/**
 * @brief return next state after the move
 *
 * @param move
 * @return State*
 */
State *State::next_state(const Move &move) {
  if (!zobrist_ready) {
    init_zobrist();
  }

  Board next = this->board;
  Point from = move.first, to = move.second;
  int p = this->player;
  int opp = 1 - p;

  int8_t orig_piece = next.board[p][from.first][from.second];
  int8_t moved = orig_piece;
  // promotion for pawn
  if (moved == 1 && (to.first == BOARD_H - 1 || to.first == 0)) {
    moved = 5;
  }

  /* Incremental hash update */
  uint64_t h = this->hash();
  h ^= zobrist_side; /* toggle side to move */

  /* XOR out piece from source */
  h ^= zobrist_piece[p][orig_piece][from.first][from.second];

  /* XOR out captured piece at destination */
  int8_t captured = next.board[opp][to.first][to.second];
  if (captured) {
    h ^= zobrist_piece[opp][captured][to.first][to.second];
    next.board[opp][to.first][to.second] = 0;
  }

  /* XOR in piece at destination */
  h ^= zobrist_piece[p][moved][to.first][to.second];

  next.board[p][from.first][from.second] = 0;
  next.board[p][to.first][to.second] = moved;

  State *ns = new State(next, opp);
  ns->zobrist_hash = h;
  ns->zobrist_valid = true;
  return ns;
}

// Move tables and search heuristics definitions have been moved to the top of
// the file so helper functions can access them.

/*============================================================
 * Naive move generation (array-based, branch-heavy)
 *============================================================*/
void State::get_legal_actions_naive() {
  this->game_state = NONE;
  std::vector<Move> all_actions;
  all_actions.reserve(64);
  auto self_board = this->board.board[this->player];
  auto oppn_board = this->board.board[1 - this->player];

  int now_piece, oppn_piece;
  for (int i = 0; i < BOARD_H; i += 1) {
    for (int j = 0; j < BOARD_W; j += 1) {
      if ((now_piece = self_board[i][j])) {
        switch (now_piece) {
        case 1: // pawn
          if (this->player && i < BOARD_H - 1) {
            // black
            if (!oppn_board[i + 1][j] && !self_board[i + 1][j]) {
              all_actions.push_back(Move(Point(i, j), Point(i + 1, j)));
            }
            if (j < BOARD_W - 1 &&
                (oppn_piece = oppn_board[i + 1][j + 1]) > 0) {
              all_actions.push_back(Move(Point(i, j), Point(i + 1, j + 1)));
              if (oppn_piece == 6) {
                this->game_state = WIN;
                this->legal_actions = all_actions;
                return;
              }
            }
            if (j > 0 && (oppn_piece = oppn_board[i + 1][j - 1]) > 0) {
              all_actions.push_back(Move(Point(i, j), Point(i + 1, j - 1)));
              if (oppn_piece == 6) {
                this->game_state = WIN;
                this->legal_actions = all_actions;
                return;
              }
            }
          } else if (!this->player && i > 0) {
            // white
            if (!oppn_board[i - 1][j] && !self_board[i - 1][j]) {
              all_actions.push_back(Move(Point(i, j), Point(i - 1, j)));
            }
            if (j < BOARD_W - 1 &&
                (oppn_piece = oppn_board[i - 1][j + 1]) > 0) {
              all_actions.push_back(Move(Point(i, j), Point(i - 1, j + 1)));
              if (oppn_piece == 6) {
                this->game_state = WIN;
                this->legal_actions = all_actions;
                return;
              }
            }
            if (j > 0 && (oppn_piece = oppn_board[i - 1][j - 1]) > 0) {
              all_actions.push_back(Move(Point(i, j), Point(i - 1, j - 1)));
              if (oppn_piece == 6) {
                this->game_state = WIN;
                this->legal_actions = all_actions;
                return;
              }
            }
          }
          break;

        case 2: // rook
        case 4: // bishop
        case 5: // queen
          int st, end;
          switch (now_piece) {
          case 2:
            st = 0;
            end = 4;
            break; // rook
          case 4:
            st = 4;
            end = 8;
            break; // bishop
          case 5:
            st = 0;
            end = 8;
            break; // queen
          default:
            st = 0;
            end = -1;
          }
          for (int part = st; part < end; part += 1) {
            auto move_list = move_table_rook_bishop[part];
            for (int k = 0; k < std::max(BOARD_H, BOARD_W); k += 1) {
              int p[2] = {move_list[k][0] + i, move_list[k][1] + j};

              if (p[0] >= BOARD_H || p[0] < 0 || p[1] >= BOARD_W || p[1] < 0) {
                break;
              }
              now_piece = self_board[p[0]][p[1]];
              if (now_piece) {
                break;
              }

              all_actions.push_back(Move(Point(i, j), Point(p[0], p[1])));

              oppn_piece = oppn_board[p[0]][p[1]];
              if (oppn_piece) {
                if (oppn_piece == 6) {
                  this->game_state = WIN;
                  this->legal_actions = all_actions;
                  return;
                } else {
                  break;
                }
              };
            }
          }
          break;

        case 3: // knight
          // [ Hackathon TODO 2-2 ]
          // complete knight's movement, you can refer to other pieces' movement
          for (auto move : move_table_knight) {
            int p[2] = {move[0] + i, move[1] + j};

            if (p[0] >= BOARD_H || p[0] < 0 || p[1] >= BOARD_W || p[1] < 0) {
              continue;
            }
            now_piece = self_board[p[0]][p[1]];
            if (now_piece) {
              continue;
            }

            all_actions.push_back(Move(Point(i, j), Point(p[0], p[1])));

            oppn_piece = oppn_board[p[0]][p[1]];
            if (oppn_piece == 6) {
              this->game_state = WIN;
              this->legal_actions = all_actions;
              return;
            }
          }
          break;

        case 6: // king
          for (auto move : move_table_king) {
            int p[2] = {move[0] + i, move[1] + j};

            if (p[0] >= BOARD_H || p[0] < 0 || p[1] >= BOARD_W || p[1] < 0) {
              continue;
            }
            now_piece = self_board[p[0]][p[1]];
            if (now_piece) {
              continue;
            }

            all_actions.push_back(Move(Point(i, j), Point(p[0], p[1])));

            oppn_piece = oppn_board[p[0]][p[1]];
            if (oppn_piece == 6) {
              this->game_state = WIN;
              this->legal_actions = all_actions;
              return;
            }
          }
          break;
        }
      }
    }
  }
  this->legal_actions = all_actions;
}

// Bitboard lookup tables and initialization are now defined at the top of the
// file.

void State::get_legal_actions_bitboard() {
  if (!bb_ready) {
    bb_init();
  }

  this->game_state = NONE;
  this->legal_actions.clear();
  this->legal_actions.reserve(64);

  int self = this->player;
  int oppn = 1 - self;

  // Build occupancy bitmasks and piece-type lookup
  uint32_t self_occ = 0, oppn_occ = 0;
  int self_pt[30] = {}; // piece type at each square (self)
  int oppn_pt[30] = {}; // piece type at each square (opponent)

  for (int r = 0; r < BOARD_H; r++) {
    for (int c = 0; c < BOARD_W; c++) {
      int sq = BB_SQ(r, c);
      if (this->board.board[self][r][c]) {
        self_occ |= 1u << sq;
        self_pt[sq] = this->board.board[self][r][c];
      }
      if (this->board.board[oppn][r][c]) {
        oppn_occ |= 1u << sq;
        oppn_pt[sq] = this->board.board[oppn][r][c];
      }
    }
  }

  uint32_t all_occ = self_occ | oppn_occ;

  // Iterate own pieces via bit scan
  uint32_t pieces = self_occ;
  while (pieces) {
    int sq = __builtin_ctz(pieces);
    pieces &= pieces - 1;
    int r = BB_ROW(sq), c = BB_COL(sq);
    int piece = self_pt[sq];
    uint32_t targets = 0;

    switch (piece) {
    case 1: { // Pawn
      uint32_t push = bb_pawn_push[self][sq] & ~all_occ;
      uint32_t cap = bb_pawn_cap[self][sq] & oppn_occ;
      // Check for king capture in captures
      uint32_t cap_scan = cap;
      while (cap_scan) {
        int to = __builtin_ctz(cap_scan);
        cap_scan &= cap_scan - 1;
        if (oppn_pt[to] == 6) {
          this->game_state = WIN;
          this->legal_actions.push_back(
              Move(Point(r, c), Point(BB_ROW(to), BB_COL(to))));
          return;
        }
      }
      targets = push | cap;
      break;
    }

    case 3: { // Knight
      targets = bb_knight[sq] & ~self_occ;
      uint32_t opp_targets = targets & oppn_occ;
      while (opp_targets) {
        int to = __builtin_ctz(opp_targets);
        opp_targets &= opp_targets - 1;
        if (oppn_pt[to] == 6) {
          this->game_state = WIN;
          this->legal_actions.push_back(
              Move(Point(r, c), Point(BB_ROW(to), BB_COL(to))));
          return;
        }
      }
      break;
    }

    case 6: { // King
      targets = bb_king[sq] & ~self_occ;
      uint32_t opp_targets = targets & oppn_occ;
      while (opp_targets) {
        int to = __builtin_ctz(opp_targets);
        opp_targets &= opp_targets - 1;
        if (oppn_pt[to] == 6) {
          this->game_state = WIN;
          this->legal_actions.push_back(
              Move(Point(r, c), Point(BB_ROW(to), BB_COL(to))));
          return;
        }
      }
      break;
    }

    case 2:   // Rook
    case 4:   // Bishop
    case 5: { // Queen
      int d_start = (piece == 4) ? 4 : 0;
      int d_end = (piece == 2) ? 4 : 8;
      for (int d = d_start; d < d_end; d++) {
        int cr = r + bb_dr[d], cc = c + bb_dc[d];
        while (cr >= 0 && cr < BOARD_H && cc >= 0 && cc < BOARD_W) {
          int to = BB_SQ(cr, cc);
          uint32_t to_bit = 1u << to;
          if (self_occ & to_bit) {
            break; // own piece blocks
          }

          if ((oppn_occ & to_bit) && oppn_pt[to] == 6) {
            this->game_state = WIN;
            this->legal_actions.push_back(Move(Point(r, c), Point(cr, cc)));
            return;
          }

          targets |= to_bit;
          if (oppn_occ & to_bit) {
            break; // captured, stop sliding
          }
          cr += bb_dr[d];
          cc += bb_dc[d];
        }
      }
      break;
    }
    }

    // Convert target bitmask to Move objects
    while (targets) {
      int to = __builtin_ctz(targets);
      targets &= targets - 1;
      this->legal_actions.push_back(
          Move(Point(r, c), Point(BB_ROW(to), BB_COL(to))));
    }
  }
}

/*============================================================
 * Dispatcher
 *============================================================*/
void State::get_legal_actions() {
#ifdef USE_BITBOARD
  get_legal_actions_bitboard();
#else
  get_legal_actions_naive();
#endif
}

const char piece_table[2][7][5] = {{" ", "♙", "♖", "♘", "♗", "♕", "♔"},
                                   {" ", "♟", "♜", "♞", "♝", "♛", "♚"}};
/**
 * @brief encode the output for command line output
 *
 * @return std::string
 */
std::string State::encode_output() const {
  std::stringstream ss;
  int now_piece;
  for (int i = 0; i < BOARD_H; i += 1) {
    for (int j = 0; j < BOARD_W; j += 1) {
      if ((now_piece = this->board.board[0][i][j])) {
        ss << std::string(piece_table[0][now_piece]);
      } else if ((now_piece = this->board.board[1][i][j])) {
        ss << std::string(piece_table[1][now_piece]);
      } else {
        ss << " ";
      }
      ss << " ";
    }
    ss << "\n";
  }
  return ss.str();
}

/**
 * @brief encode the state to the format for player
 *
 * @return std::string
 */
std::string State::encode_state() {
  std::stringstream ss;
  ss << this->player;
  ss << "\n";
  for (int pl = 0; pl < 2; pl += 1) {
    for (int i = 0; i < BOARD_H; i += 1) {
      for (int j = 0; j < BOARD_W; j += 1) {
        ss << int(this->board.board[pl][i][j]);
        ss << " ";
      }
      ss << "\n";
    }
    ss << "\n";
  }
  return ss.str();
}

BaseState *State::create_null_state() const {
  State *s = new State(this->board, 1 - this->player);
  s->get_legal_actions();
  return s;
}

/* === Board serialization === */
static const char *piece_chars = ".PRNBQK";
static const char *piece_chars_lower = ".prnbqk";

std::string State::encode_board() const {
  std::string s;
  for (int r = 0; r < BOARD_H; r++) {
    if (r > 0) {
      s += '/';
    }
    for (int c = 0; c < BOARD_W; c++) {
      int w = board.board[0][r][c];
      int b = board.board[1][r][c];
      if (w > 0 && w <= 6) {
        s += piece_chars[w];
      } else if (b > 0 && b <= 6) {
        s += piece_chars_lower[b];
      } else {
        s += '.';
      }
    }
  }
  return s;
}

void State::decode_board(const std::string &s, int side_to_move) {
  player = side_to_move;
  game_state = UNKNOWN;
  zobrist_valid = false;
  board = Board{};
  int r = 0, c = 0;
  for (char ch : s) {
    if (ch == '/') {
      r++;
      c = 0;
      continue;
    }
    if (r >= BOARD_H || c >= BOARD_W) {
      break;
    }
    if (ch >= 'A' && ch <= 'Z') {
      for (int p = 1; p <= 6; p++) {
        if (piece_chars[p] == ch) {
          board.board[0][r][c] = p;
          break;
        }
      }
    } else if (ch >= 'a' && ch <= 'z') {
      for (int p = 1; p <= 6; p++) {
        if (piece_chars_lower[p] == ch) {
          board.board[1][r][c] = p;
          break;
        }
      }
    }
    c++;
  }
  get_legal_actions();
}

/* (Zobrist tables moved above next_state) */

/*============================================================
 * Cell display for protocol (d command)
 *============================================================*/
std::string State::cell_display(int row, int col) const {
  int w = static_cast<int>(board.board[0][row][col]);
  int b = static_cast<int>(board.board[1][row][col]);
  if (w) {
    const char *names = ".PRNBQK";
    return std::string(" ") + names[w] + " ";
  } else if (b) {
    const char *names = ".prnbqk";
    return std::string(" ") + names[b] + " ";
  } else {
    return " . ";
  }
}

/* === Repetition: chess 3-fold rule === */
bool State::check_repetition(const GameHistory &history, int &out_score) const {
  if (history.count(hash()) >= 3) {
    out_score = 0; /* draw */
    return true;
  }
  return false;
}
