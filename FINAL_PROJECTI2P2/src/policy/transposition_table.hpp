#pragma once
#include <cstdint>
#include <vector>
#include <cstring>
#include "base_state.hpp"

enum TTFlag : uint8_t { TT_NONE = 0, TT_EXACT, TT_LOWER, TT_UPPER };

struct TTEntry {
    uint64_t key = 0;
    int depth = -1;
    int score = 0;
    Move best_move = {};
    TTFlag flag = TT_NONE;
};

class TranspositionTable {
public:
    static constexpr size_t SIZE = 1 << 20; /* ~1M entries */

    TranspositionTable() : entries(SIZE) {}

    void clear(){
        for(auto& e : entries){
            e = TTEntry{};
        }
    }

    bool probe(
        uint64_t key,
        int depth,
        int alpha,
        int beta,
        int& out_score,
        Move& out_move
    ) const {
        const TTEntry& e = entries[key & (SIZE - 1)];
        if(e.key != key || e.flag == TT_NONE){
            return false;
        }
        out_move = e.best_move;
        if(e.depth < depth){
            return false;
        }
        if(e.flag == TT_EXACT){
            out_score = e.score;
            return true;
        }
        if(e.flag == TT_LOWER && e.score >= beta){
            out_score = e.score;
            return true;
        }
        if(e.flag == TT_UPPER && e.score <= alpha){
            out_score = e.score;
            return true;
        }
        return false;
    }

    void store(
        uint64_t key,
        int depth,
        int score,
        TTFlag flag,
        const Move& move
    ){
        TTEntry& e = entries[key & (SIZE - 1)];
        if(e.key != 0 && e.key != key && e.depth > depth){
            return;
        }
        if(e.key != key){
            e.best_move = {};
        }
        e.key = key;
        e.depth = depth;
        e.score = score;
        e.flag = flag;
        if(move.first.first != 0 || move.first.second != 0
           || move.second.first != 0 || move.second.second != 0){
            e.best_move = move;
        }
    }

private:
    std::vector<TTEntry> entries;
};
