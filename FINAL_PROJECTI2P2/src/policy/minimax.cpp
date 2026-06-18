#include <utility>
#include "state.hpp"
#include "minimax.hpp"


int MiniMax::eval_ctx(
    State *state,
    int depth,
    int alpha,
    int beta,
    GameHistory& history,
    int ply,
    SearchContext& ctx,
    const MMParams& p
){
    ctx.nodes++;
    if(ply > ctx.seldepth){
        ctx.seldepth = ply;
    }
    if(ctx.stop){
        return 0;
    }

    if(state->legal_actions.empty() && state->game_state == UNKNOWN){
        state->get_legal_actions();
    }

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
    history.push(state->hash());

    if(depth <= 0){
        int score = state->evaluate(
            p.use_kp_eval, p.use_eval_mobility, &history
        ); 
        history.pop(state->hash());
        return score;
    }

    int best_score = M_MAX;

    for(auto& action : state->legal_actions){
        State* next = static_cast<State*>(state->next_state(action));
        next->get_legal_actions();

        bool same = next->same_player_as_parent();

        int score;
        if (same) {
            score = eval_ctx(
                next,
                depth,
                alpha,
                beta,
                history,
                ply + 1,
                ctx,
                p
            );
        } else {
            score = -eval_ctx(
                next,
                depth - 1,
                -beta,
                -alpha,
                history,
                ply + 1,
                ctx,
                p
            );
        }

        delete next;

        if(score > best_score){
            best_score = score;
        }
        if(best_score > alpha){
            alpha = best_score;
        }
        if(alpha >= beta){
            break;
        }
    }

    history.pop(state->hash());
    return best_score;
}


SearchResult MiniMax::search(
    State *state,
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

    int best_score = M_MAX - 10;
    int alpha = M_MAX;
    int beta = P_MAX;
    int move_index = 0;
    int total_moves = (int)state->legal_actions.size();

    for(auto& action : state->legal_actions){
        State* next = static_cast<State*>(state->next_state(action));
        next->get_legal_actions();

        bool same = next->same_player_as_parent();
        
        int score;
        if (same) {
            score = eval_ctx(
                next,
                depth,
                alpha,
                beta,
                history,
                1,
                ctx,
                p
            );
        } else {
            score = -eval_ctx(
                next,
                depth - 1,
                -beta,
                -alpha,
                history,
                1,
                ctx,
                p
            );
        }
        delete next;

        if(score > best_score){
            best_score = score;
            result.best_move = action;

            if(p.report_partial && ctx.on_root_update){
               ctx.on_root_update({result.best_move, best_score, depth, move_index + 1, total_moves});
            }
        }
        if(best_score > alpha){
            alpha = best_score;
        }
        move_index++;
    }

    result.score = best_score;
    result.seldepth = ctx.seldepth;
    result.nodes = ctx.nodes;
    result.pv = {result.best_move};
    return result;
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
