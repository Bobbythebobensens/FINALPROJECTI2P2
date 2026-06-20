#include <utility>
#include <algorithm>
#include <vector>
#include <cstring>
#include <chrono>
#include "state.hpp"
#include "config.hpp"
#include "minimax.hpp"
#include "transposition_table.hpp"


static constexpr int MAX_PLY = 64;
static constexpr int MAX_QDEPTH = 8;

static TranspositionTable g_tt;
static Move g_killers[MAX_PLY][2];
static int g_history[BOARD_H][BOARD_W][BOARD_H][BOARD_W];

static void clear_search_heuristics(){
    for(int p = 0; p < MAX_PLY; p++){
        g_killers[p][0] = {};
        g_killers[p][1] = {};
    }
    std::memset(g_history, 0, sizeof(g_history));
}


static bool is_capture(const State* state, const Move& m){
    int opp = 1 - state->player;
    return state->piece_at(opp, (int)m.second.first, (int)m.second.second) != 0;
}


static int mvv_lva(const State* state, const Move& m){
    int opp = 1 - state->player;
    int victim = state->piece_at(opp, (int)m.second.first, (int)m.second.second);
    if(victim == 0){
        return 0;
    }
    int attacker = state->piece_at(
        state->player, (int)m.first.first, (int)m.first.second
    );
    return 1000 * PIECE_VALUES[victim] - PIECE_VALUES[attacker];
}


static int move_order_score(
    const State* state,
    const Move& m,
    const Move& tt_move,
    int ply
){
    if(m == tt_move){
        return 2000000;
    }
    int cap = mvv_lva(state, m);
    if(cap > 0){
        return 1000000 + cap;
    }
    if(ply >= 0 && ply < MAX_PLY){
        if(m == g_killers[ply][0]){
            return 900000;
        }
        if(m == g_killers[ply][1]){
            return 800000;
        }
    }
    return g_history[(int)m.first.first][(int)m.first.second]
                    [(int)m.second.first][(int)m.second.second];
}


static void order_moves(
    const State* state,
    std::vector<Move>& moves,
    const Move& tt_move,
    int ply
){
    std::stable_sort(
        moves.begin(),
        moves.end(),
        [&](const Move& a, const Move& b){
            return move_order_score(state, a, tt_move, ply)
                > move_order_score(state, b, tt_move, ply);
        }
    );
}


static void update_killers(int ply, const Move& m, bool capture){
    if(capture || ply < 0 || ply >= MAX_PLY){
        return;
    }
    if(g_killers[ply][0] == m){
        return;
    }
    g_killers[ply][1] = g_killers[ply][0];
    g_killers[ply][0] = m;
}


static void update_history(const Move& m, int depth){
    auto& h = g_history[(int)m.first.first][(int)m.first.second]
                       [(int)m.second.first][(int)m.second.second];
    h += depth * depth;
    if(h > 1000000){
        h = 1000000;
    }
}


static bool ensure_legal(State* state){
    if(state->legal_actions.empty() && state->game_state == UNKNOWN){
        state->get_legal_actions();
    }
    return true;
}


int MiniMax::quiesce(
    State* state,
    int alpha,
    int beta,
    GameHistory& history,
    int ply,
    SearchContext& ctx,
    const MMParams& p,
    int qdepth
){
    ctx.nodes++;
    if(ply > ctx.seldepth){
        ctx.seldepth = ply;
    }
    if ((ctx.nodes & 2047) == 0 && ctx.time_limit_ms > 0) {
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - ctx.start_time).count();
        if (elapsed >= ctx.time_limit_ms) {
            ctx.stop = true;
        }
    }
    if(ctx.stop){
        return 0;
    }

    ensure_legal(state);

    if(state->game_state == WIN){
        return P_MAX - ply;
    }
    if(state->game_state == DRAW){
        return 0;
    }

    int rep_score;
    if(state->check_repetition(history, rep_score)){
        return rep_score;
    }

    if(qdepth >= MAX_QDEPTH){
        return state->evaluate(p.use_kp_eval, false, &history);
    }

    // Check if the current player's king is in check.
    bool in_check = false;
    {
        State temp(state->board, 1 - state->player);
        temp.get_legal_actions();
        if(temp.game_state == WIN){
            in_check = true;
        }
    }

    if(!in_check){
        int stand_pat = state->evaluate(p.use_kp_eval, false, &history);
        if(stand_pat >= beta){
            return beta;
        }
        if(stand_pat > alpha){
            alpha = stand_pat;
        }

        /* Delta pruning: only if not in check. */
        int delta = p.use_kp_eval ? 400 : 40;
        if(stand_pat + delta < alpha){
            return alpha;
        }
    }

    Move tt_move = {};
    int tt_score = 0;
    uint64_t key = state->hash();
    g_tt.probe(key, 0, alpha, beta, tt_score, tt_move);

    std::vector<Move> moves_to_search;
    if(in_check){
        moves_to_search = state->legal_actions;
    }else{
        moves_to_search.reserve(state->legal_actions.size());
        for(const Move& m : state->legal_actions){
            if(is_capture(state, m)){
                moves_to_search.push_back(m);
            }
        }
    }
    order_moves(state, moves_to_search, tt_move, ply);

    for(const Move& action : moves_to_search){
        State* next = static_cast<State*>(state->next_state(action));
        next->get_legal_actions();

        int score;
        if(next->game_state == WIN){
            score = -(P_MAX - (ply + 1));
        }else{
            score = -quiesce(
                next, -beta, -alpha, history, ply + 1, ctx, p, qdepth + 1
            );
        }
        if(ctx.stop){
            delete next;
            return 0;
        }
        delete next;

        if(score >= beta){
            update_history(action, 1);
            return beta;
        }
        if(score > alpha){
            alpha = score;
        }
    }

    return alpha;
}


int MiniMax::eval_ctx(
    State* state,
    int depth,
    int alpha,
    int beta,
    GameHistory& history,
    int ply,
    SearchContext& ctx,
    const MMParams& p,
    std::vector<Move>* pv_out
){
    ctx.nodes++;
    if(ply > ctx.seldepth){
        ctx.seldepth = ply;
    }
    if ((ctx.nodes & 2047) == 0 && ctx.time_limit_ms > 0) {
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - ctx.start_time).count();
        if (elapsed >= ctx.time_limit_ms) {
            ctx.stop = true;
        }
    }
    if(ctx.stop){
        return 0;
    }

    ensure_legal(state);

    if(state->game_state == WIN){
        return P_MAX - ply;
    }
    if(state->game_state == DRAW){
        return 0;
    }

    int rep_score;
    if(state->check_repetition(history, rep_score)){
        return rep_score;
    }

    uint64_t key = state->hash();
    Move tt_move = {};
    int tt_score = 0;
    if(g_tt.probe(key, depth, alpha, beta, tt_score, tt_move)){
        return tt_score;
    }

    history.push(key);

    if(depth <= 0){
        int score = quiesce(state, alpha, beta, history, ply, ctx, p, 0);
        history.pop(key);
        return score;
    }

    /* Null-move pruning: passing is almost never good in king-capture chess. */
    if(
        depth >= 4
        && state->legal_actions.size() > 1
        && beta < P_MAX - 100
    ){
        State* null_state = static_cast<State*>(state->create_null_state());
        if(null_state != nullptr && null_state->game_state != WIN){
            int null_score = -eval_ctx(
                null_state,
                depth - 3,
                -beta,
                -beta + 1,
                history,
                ply + 1,
                ctx,
                p,
                nullptr
            );
            delete null_state;
            if(ctx.stop){
                history.pop(key);
                return 0;
            }
            if(null_score >= beta){
                history.pop(key);
                return beta;
            }
        }else{
            delete null_state;
        }
    }

    int best_score = M_MAX;
    Move best_move = {};
    std::vector<Move> best_pv;

    std::vector<Move> moves = state->legal_actions;
    order_moves(state, moves, tt_move, ply);

    int orig_alpha = alpha;
    bool first_move = true;

    for(const Move& action : moves){
        State* next = static_cast<State*>(state->next_state(action));
        next->get_legal_actions();

        bool same = next->same_player_as_parent();
        bool capture = is_capture(state, action);

        int score;
        std::vector<Move> child_pv;
        if(next->game_state == WIN){
            score = -(P_MAX - (ply + 1));
        }else if(same){
            score = eval_ctx(
                next,
                depth,
                alpha,
                beta,
                history,
                ply + 1,
                ctx,
                p,
                &child_pv
            );
        }else{
            if(first_move){
                score = -eval_ctx(
                    next,
                    depth - 1,
                    -beta,
                    -alpha,
                    history,
                    ply + 1,
                    ctx,
                    p,
                    &child_pv
                );
            }else{
                score = -eval_ctx(
                    next,
                    depth - 1,
                    -alpha - 1,
                    -alpha,
                    history,
                    ply + 1,
                    ctx,
                    p,
                    &child_pv
                );
                if(score > alpha && score < beta){
                    score = -eval_ctx(
                        next,
                        depth - 1,
                        -beta,
                        -alpha,
                        history,
                        ply + 1,
                        ctx,
                        p,
                        &child_pv
                    );
                }
            }
        }
        delete next;

        if(ctx.stop){
            history.pop(key);
            return 0;
        }

        if(score > best_score){
            best_score = score;
            best_move = action;
            best_pv = child_pv;
            best_pv.insert(best_pv.begin(), action);
        }
        if(best_score > alpha){
            alpha = best_score;
        }
        if(alpha >= beta){
            update_killers(ply, action, capture);
            update_history(action, depth);
            break;
        }
        first_move = false;
    }

    TTFlag flag = TT_EXACT;
    if(best_score <= orig_alpha){
        flag = TT_UPPER;
    }else if(best_score >= beta){
        flag = TT_LOWER;
    }
    g_tt.store(key, depth, best_score, flag, best_move);

    history.pop(key);

    if(pv_out != nullptr){
        *pv_out = best_pv;
    }
    return best_score;
}


SearchResult MiniMax::search(
    State* state,
    int depth,
    GameHistory& history,
    SearchContext& ctx
){
    ctx.reset();
    MMParams p = MMParams::from_map(ctx.params);
    SearchResult result;
    result.depth = depth;

    if(!state->legal_actions.size()){
        state->get_legal_actions();
    }
    if(!state->legal_actions.empty()){
        result.best_move = state->legal_actions[0];
    }

    struct RootMoveInfo {
        Move move;
        int score;
        std::vector<Move> pv;
    };
    std::vector<RootMoveInfo> searched_moves;
    searched_moves.reserve(state->legal_actions.size());

    int best_score = M_MAX - 10;
    int alpha = M_MAX;
    int beta = P_MAX;
    int move_index = 0;
    int total_moves = (int)state->legal_actions.size();
    std::vector<Move> root_pv;

    std::vector<Move> moves = state->legal_actions;
    Move tt_move = {};
    int tt_score = 0;
    g_tt.probe(state->hash(), depth, alpha, beta, tt_score, tt_move);
    order_moves(state, moves, tt_move, 0);

    for(const Move& action : moves){
        State* next = static_cast<State*>(state->next_state(action));
        next->get_legal_actions();

        bool same = next->same_player_as_parent();

        int score;
        std::vector<Move> child_pv;
        if(next->game_state == WIN){
            score = -(P_MAX - 1);
            child_pv = {};
        }else if(same){
            score = eval_ctx(
                next,
                depth,
                alpha,
                beta,
                history,
                1,
                ctx,
                p,
                &child_pv
            );
        }else{
            if(move_index == 0){
                score = -eval_ctx(
                    next,
                    depth - 1,
                    -beta,
                    -alpha,
                    history,
                    1,
                    ctx,
                    p,
                    &child_pv
                );
            }else{
                score = -eval_ctx(
                    next,
                    depth - 1,
                    -alpha - 1,
                    -alpha,
                    history,
                    1,
                    ctx,
                    p,
                    &child_pv
                );
                if(score > alpha && score < beta){
                    score = -eval_ctx(
                        next,
                        depth - 1,
                        -beta,
                        -alpha,
                        history,
                        1,
                        ctx,
                        p,
                        &child_pv
                    );
                }
            }
        }
        delete next;

        if(ctx.stop){
            break;
        }

        searched_moves.push_back({action, score, child_pv});

        if(score > best_score){
            best_score = score;
            result.best_move = action;
            root_pv = child_pv;
            root_pv.insert(root_pv.begin(), action);

            if(p.report_partial && ctx.on_root_update){
                ctx.on_root_update({
                    result.best_move,
                    best_score,
                    depth,
                    move_index + 1,
                    total_moves
                });
            }
        }
        if(best_score > alpha){
            alpha = best_score;
        }
        move_index++;
    }

    // Avoid repetition draws by picking the best non-repeating move if possible
    if (!searched_moves.empty()) {
        std::stable_sort(searched_moves.begin(), searched_moves.end(),
            [](const RootMoveInfo& a, const RootMoveInfo& b){
                return a.score > b.score;
            });

        bool found = false;
        for (const auto& rm : searched_moves) {
            State* next_state_ptr = static_cast<State*>(state->next_state(rm.move));
            if (history.count(next_state_ptr->hash()) == 0) {
                result.best_move = rm.move;
                result.score = rm.score;
                result.pv = rm.pv;
                result.pv.insert(result.pv.begin(), rm.move);
                found = true;
                delete next_state_ptr;
                break;
            }
            delete next_state_ptr;
        }

        if (!found) {
            // All moves repeat; fallback to the first best move
            result.best_move = searched_moves[0].move;
            result.score = searched_moves[0].score;
            result.pv = searched_moves[0].pv;
            result.pv.insert(result.pv.begin(), searched_moves[0].move);
        }
    } else {
        result.score = best_score;
        result.pv = root_pv.empty() ? std::vector<Move>{result.best_move} : root_pv;
    }

    result.seldepth = ctx.seldepth;
    result.nodes = ctx.nodes;
    if (result.pv.empty()) {
        result.pv = {result.best_move};
    }
    return result;
}


void MiniMax::clear_tt(){
    g_tt.clear();
    clear_search_heuristics();
}


void MiniMax::begin_search(){
    clear_search_heuristics();
}


ParamMap MiniMax::default_params(){
    return {
        {"UseKPEval", "true"},
        {"UseEvalMobility", "true"},
        {"ReportPartial", "true"},
    };
}

std::vector<ParamDef> MiniMax::param_defs(){
    return {
        {"UseKPEval", ParamDef::CHECK, "true"},
        {"UseEvalMobility", ParamDef::CHECK, "true"},
        {"ReportPartial", ParamDef::CHECK, "true"},
    };
}
