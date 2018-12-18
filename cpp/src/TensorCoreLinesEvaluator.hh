#ifndef CPP_TENSOR_CORE_LINES_EVALUATOR_HH
#define CPP_TENSOR_CORE_LINES_EVALUATOR_HH

#include "EvaluatorUtils.hh"

#include <array>

namespace tl
{
class TensorCoreLinesEvaluator
{
    using Self = TensorCoreLinesEvaluator;
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

    TensorCoreLinesEvaluator() = default;

    TensorCoreLinesEvaluator(const DoubleTri& tri,
                             const TensorInterp& t,
                             const std::array<TensorInterp, 3>& dt,
                             const Options& opts);

    TensorCoreLinesEvaluator(
            const DoubleTri& tri,
            const std::array<TPBT<double, 1, 2>, 3>& target_funcs_t,
            const std::array<TPBT<double, 0, 3>, 3>& target_funcs_dt,
            bool last_split_dir,
            uint64_t split_level,
            const Options& opts)
            : _tri(tri),
              _target_funcs_t(target_funcs_t),
              _target_funcs_dt(target_funcs_dt),
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

    friend bool operator==(const Self& t1, const Self& t2);

    friend bool operator!=(const Self& t1, const Self& t2);

private:
    DoubleTri _tri = DoubleTri{};

    std::array<TPBT<double, 1, 2>, 3> _target_funcs_t =
            std::array<TPBT<double, 1, 2>, 3>{};

    std::array<TPBT<double, 0, 3>, 3> _target_funcs_dt =
            std::array<TPBT<double, 0, 3>, 3>{};

    bool _last_split_dir = false;
    uint64_t _split_level = 0;

    Options _opts = Options{};

    template <std::size_t D>
    std::array<Self, 4> split() const
    {
        static_assert(D >= 0 && D < 2,
                      "Split space must be 0 (position) or 1 (direction)");

        auto part = [&](std::size_t i) {
            return TensorCoreLinesEvaluator(
                    _tri.split<D>(i),
                    {_target_funcs_t[0].split<D>(i),
                     _target_funcs_t[1].split<D>(i),
                     _target_funcs_t[2].split<D>(i)},
                    {_target_funcs_dt[0].split<D>(i),
                     _target_funcs_dt[1].split<D>(i),
                     _target_funcs_dt[2].split<D>(i)},
                    D == 1,
                    _split_level + 1,
                    _opts);
        };
        return {part(0), part(1), part(2), part(3)};
    }
};

double distance(const TensorCoreLinesEvaluator& t1,
                const TensorCoreLinesEvaluator& t2);

static_assert(is_evaluator<TensorCoreLinesEvaluator>::value,
              "TensorCoreLinesEvaluator is not a valid evaluator!");
}

#endif
