#pragma once
#include <vector>
#include "search_types.hpp"
#include "game_history.hpp"

struct MMParams {
    bool use_kp_eval = true;
    bool use_eval_mobility = true;
    bool report_partial = true;

    static MMParams from_map(const ParamMap& m){
        MMParams p;
        p.use_kp_eval       = param_bool(m, "UseKPEval", true);
        p.use_eval_mobility = param_bool(m, "UseEvalMobility", true);
        p.report_partial    = param_bool(m, "ReportPartial", true);
        return p;
    }
};

class MiniMax{
public:
    static int quiesce(
        State* state,
        int alpha,
        int beta,
        GameHistory& history,
        int ply,
        SearchContext& ctx,
        const MMParams& p,
        int qdepth = 0
    );

    static int eval_ctx(
        State* state,
        int depth,
        int alpha,
        int beta,
        GameHistory& history,
        int ply,
        SearchContext& ctx,
        const MMParams& p,
        std::vector<Move>* pv_out = nullptr
    );

    static SearchResult search(
        State* state,
        int depth,
        GameHistory& history,
        SearchContext& ctx
    );

    static void clear_tt();
    static void begin_search();

    static ParamMap default_params();
    static std::vector<ParamDef> param_defs();
};
