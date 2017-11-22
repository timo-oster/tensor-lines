#ifndef CPP_PARALLEL_EIGENVECTORS_EVALUATOR_HH
#define CPP_PARALLEL_EIGENVECTORS_EVALUATOR_HH

#include "EvaluatorUtils.hh"

#include <array>

namespace pev
{
class ParallelEigenvectorsEvaluator
{
    using Self = ParallelEigenvectorsEvaluator;
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

    ParallelEigenvectorsEvaluator() = default;

    ParallelEigenvectorsEvaluator(const DoubleTri& tri,
                                  const TensorInterp& s,
                                  const TensorInterp& t,
                                  const Options& opts);

    ParallelEigenvectorsEvaluator(
            const DoubleTri& tri,
            const std::array<TPBT<double, 1, 2>, 6>& target_funcs,
            bool last_split_dir,
            uint64_t split_level,
            const Options& opts)
            : _tri(tri),
              _target_funcs(target_funcs),
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
     *          solution is within tolerances of spatial_epsilon,
     *          direction_epsilon and ev_epsilon; returns Result::Discard if no
     *          solution can be found by subdividing further, and returns
     *          Result::Split if further subdivision is necessary to find a
     *          solution.
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

    std::array<TPBT<double, 1, 2>, 6> _target_funcs =
            std::array<TPBT<double, 1, 2>, 6>{};

    bool _last_split_dir = false;
    uint64_t _split_level = 0;

    Options _opts = Options{};

    template <std::size_t D>
    std::array<Self, 4> split() const
    {
        static_assert(D >= 0 && D < 2,
                      "Split space must be 0 (position) or 1 (direction)");

        auto part = [&](std::size_t i) {
            return ParallelEigenvectorsEvaluator(
                    _tri.split<D>(i),
                    {_target_funcs[0].split<D>(i),
                     _target_funcs[1].split<D>(i),
                     _target_funcs[2].split<D>(i),
                     _target_funcs[3].split<D>(i),
                     _target_funcs[4].split<D>(i),
                     _target_funcs[5].split<D>(i)},
                    D == 1,
                    _split_level + 1,
                    _opts);
        };

        return {part(0), part(1), part(2), part(3)};
    }
};

static_assert(is_evaluator<ParallelEigenvectorsEvaluator>::value,
              "ParallelEigenvectorsEvaluator is not a valid evaluator!");
}

#endif
