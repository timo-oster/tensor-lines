#ifndef CPP_TENSOR_SUJUDI_HAIMES_EVALUATOR_HH
#define CPP_TENSOR_SUJUDI_HAIMES_EVALUATOR_HH

#include "EvaluatorUtils.hh"

#include <array>

namespace pev
{
class TensorSujudiHaimesEvaluator
{
    using Self = TensorSujudiHaimesEvaluator;
    template <typename T, std::size_t... Degrees>
    using TPBT = TensorProductBezierTriangle<T, double, Degrees...>;

public:

    struct Options
    {
        double tolerance = 1e-6;

        friend bool operator==(const Options& o1, const Options& o2)
        {
            return o1.tolerance == o2.tolerance;
        }

        friend bool operator!=(const Options& o1, const Options& o2)
        {
            return !(o1 == o2);
        }
    };

    TensorSujudiHaimesEvaluator() = default;

    TensorSujudiHaimesEvaluator(const DoubleTri& tri,
                                const TensorInterp& t,
                                const std::array<TensorInterp, 3>& dt,
                                const Options& opts);

    TensorSujudiHaimesEvaluator(
            const DoubleTri& tri,
            const std::array<TPBT<double, 1, 3>, 6>& target_funcs,
            // const TPBT<double, 0, 2>& dir_length,
            // const TPBT<double, 2, 2>& tr_length,
            // const TPBT<double, 2, 0>& trace_sq,
            bool last_split_dir,
            uint64_t split_level,
            const Options& opts)
            : _tri(tri),
              _target_funcs(target_funcs),
              // _dir_length(dir_length),
              // _tr_length(tr_length),
              // _trace_sq(trace_sq),
              _last_split_dir(last_split_dir),
              _split_level(split_level),
              _opts(opts)
    {
    }

    /**
     * Get the triangles in position and direction space represented by the
     * evaluator
     */
    const DoubleTri& tris() const
    {
        return _tri;
    }

    /**
     * Split into four new evaluators
     */
    std::array<Self, 4> split() const;

    /**
     * Get the current subdivision level
     */
    std::size_t splitLevel() const
    {
        return _split_level;
    }

    /**
     * @brief Evaluate the current state and check if it should be split.
     * @details Evaluates the target functions and returns Result::Accept if the
     *          solution is within tolerances of tolerance; returns
     *          Result::Discard if no solution can be found by subdividing
     *          further, and returns Result::Split if further subdivision is
     *          necessary to find a solution.
     */
    Result eval();

    /**
     * Get the current worst case residual error
     */
    double error() const;

    /**
     * Get an estimate of the condition of the problem.
     *
     * Should return 0 if the solution is certain to be a point and
     * approach infinity as the solution becomes more like a line structure
     */
    double condition() const;

    friend double distance(const Self& t1, const Self& t2);

    friend bool operator==(const Self& t1, const Self& t2);

    friend bool operator!=(const Self& t1, const Self& t2);

private:
    DoubleTri _tri = DoubleTri{};

    // todo: Split up into 2 * 3 and reduce degree of T*r. Measure speedup.
    std::array<TPBT<double, 1, 3>, 6> _target_funcs =
            std::array<TPBT<double, 1, 3>, 6>{};

    // TPBT<double, 0, 2> _dir_length = TPBT<double, 0, 2>{};

    // TPBT<double, 2, 2> _tr_length = TPBT<double, 2, 2>{};

    // TPBT<double, 2, 0> _trace_sq = TPBT<double, 2, 0>{};

    bool _last_split_dir = false;
    uint64_t _split_level = 0;

    Options _opts = Options{};

    template <std::size_t D>
    std::array<Self, 4> split() const
    {
        static_assert(D >= 0 && D < 2,
                      "Split space must be 0 (position) or 1 (direction)");

        auto part = [&](std::size_t i) {
            return TensorSujudiHaimesEvaluator(
                    _tri.split<D>(i),
                    {_target_funcs[0].split<D>(i),
                     _target_funcs[1].split<D>(i),
                     _target_funcs[2].split<D>(i),
                     _target_funcs[3].split<D>(i),
                     _target_funcs[4].split<D>(i),
                     _target_funcs[5].split<D>(i)},
                    // _dir_length.split<D>(i),
                    // _tr_length.split<D>(i),
                    // _trace_sq.split<D>(i),
                    D == 1,
                    _split_level + 1,
                    _opts);
        };
        return {part(0), part(1), part(2), part(3)};
    }
};

static_assert(is_evaluator<TensorSujudiHaimesEvaluator>::value,
              "TensorSujudiHaimesEvaluator is not a valid evaluator!");
}

#endif
