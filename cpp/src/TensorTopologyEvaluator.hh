#ifndef CPP_TENSOR_TOPOLOGY_EVALUATOR_HH
#define CPP_TENSOR_TOPOLOGY_EVALUATOR_HH

#include "EvaluatorUtils.hh"

#include <array>

namespace tl
{
class TensorTopologyEvaluator
{
    using Self = TensorTopologyEvaluator;
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

    TensorTopologyEvaluator() = default;

    TensorTopologyEvaluator(const DoubleTri& tri,
                            const TensorInterp& t,
                            const Options& opts);

    TensorTopologyEvaluator(
            const DoubleTri& tri,
            const std::array<TPBT<double, 3, 0>, 7>& target_funcs,
            uint64_t split_level,
            const Options& opts)
            : _tri(tri),
              _target_funcs(target_funcs),
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

    std::array<TPBT<double, 3, 0>, 7> _target_funcs =
            std::array<TPBT<double, 3, 0>, 7>{};

    uint64_t _split_level = 0;

    Options _opts = Options{};
};

static_assert(is_evaluator<TensorTopologyEvaluator>::value,
              "TensorTopologyEvaluator is not a valid evaluator!");
}

#endif
