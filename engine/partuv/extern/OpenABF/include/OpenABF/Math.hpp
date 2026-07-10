#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace OpenABF
{

/** @brief Pi, templated for floating-point type */
template <class T>
constexpr T PI = T(3.1415926535897932385L);

/** @brief Inf, templated for floating-point type */
template <class T>
constexpr T INF = std::numeric_limits<T>::infinity();

/** @brief Vector dot product (inner product) */
template <typename T1, typename T2>
auto dot(const T1& a, const T2& b)
{
    using Ret = decltype(*std::begin(a));
    return std::inner_product(
        std::begin(a), std::end(a), std::begin(b), Ret(0));
}
/** @brief Vector cross product */
template <typename T1, typename T2>
auto cross(const T1& a, const T2& b) -> T1
{
    T1 c;
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];
    return c;
}

/** @brief Norm type enumeration */
enum class Norm {
    L1,  /**< \f$L_1\f$ norm */
    L2,  /**< \f$L_2\f$ norm */
    LInf /**< \f$L_{Inf}\f$ norm */
};

/** @brief Compute vector norm */
template <class Vector>
auto norm(const Vector& v, Norm norm = Norm::L2)
{
    using Ret = decltype(*std::begin(v));
    switch (norm) {
        case Norm::L1: {
            return std::accumulate(
                std::begin(v), std::end(v), Ret(0),
                [](auto a, auto b) { return a + std::abs(b); });
        }
        case Norm::L2: {
            auto sum = std::accumulate(
                std::begin(v), std::end(v), Ret(0),
                [](auto a, auto b) { return a + (b * b); });
            return std::sqrt(sum);
        }
        case Norm::LInf: {
            return std::abs(*std::max_element(
                std::begin(v), std::end(v),
                [](auto a, auto b) { return std::abs(a) < std::abs(b); }));
        }
    }
    throw std::invalid_argument("Invalid norm option");
}

/** @brief Normalize a vector (i.e. compute a unit vector) */
template <class Vector>
auto normalize(Vector v)
{
    return v / norm(v, Norm::L2);
}

/** @brief Compute the interior angle between two vectors */
template <class Vector1, class Vector2>
auto interior_angle(const Vector1& a, const Vector2& b)
{
    double denom = norm(a, Norm::L2) * norm(b, Norm::L2);
    const double epsilon = 1e-9;
    if (fabs(denom) < epsilon) {
        return 0.0;
    }

    double cosine = dot(a, b) / denom;
    
    // Clamp to [-1, 1] to avoid NaNs
    cosine = std::max(-1.0, std::min(1.0, cosine));
    return std::acos(cosine);
}

/** @brief Convert degrees to radians */
template <
    typename T = float,
    typename T2,
    std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
constexpr auto to_radians(T2 deg) -> T
{
    return deg * PI<T> / T(180);
}

/** @brief Convert radians to degrees */
template <
    typename T = float,
    typename T2,
    std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
constexpr auto to_degrees(T2 rad) -> T
{
    return rad * T(180) / PI<T>;
}

}  // namespace OpenABF