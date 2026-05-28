#pragma once

#include <array>
#include <iostream>

#include "OpenABF/Math.hpp"

namespace OpenABF
{
namespace detail
{
#if __cplusplus > 201103L
/** @brief Helper type to perform parameter pack folding in C++11/14 */
struct ExpandType {
    /** Constructor */
    template <typename... T>
    explicit ExpandType(T&&...)
    {
    }
};
#endif
}  // namespace detail

/**
 * @brief N-dimensional vector class
 *
 * Essentially a wrapper around std::array that makes it more convenient for
 * vector math purposes.
 *
 * @tparam T Element type
 * @tparam Dims Number of elements
 */
template <
    typename T,
    std::size_t Dims,
    std::enable_if_t<std::is_arithmetic<T>::value, bool> = true>
class Vec
{
    /** Underlying element storage */
    using Container = std::array<T, Dims>;

public:
    /** Element type */
    using value_type = T;
    /** Vector size type */
    using size_type = typename Container::size_type;
    /** Difference type */
    using difference_type = typename Container::difference_type;
    /** Reference type */
    using reference = value_type&;
    /** Const reference type */
    using const_reference = const value_type&;
    /** Pointer type */
    using pointer = value_type*;
    /** Const pointer type */
    using const_pointer = const value_type*;
    /** Iterator type */
    using iterator = typename Container::iterator;
    /** Const iterator type */
    using const_iterator = typename Container::const_iterator;
    /** Reverse iterator type */
    using reverse_iterator = typename Container::reverse_iterator;
    /** Const reverse iterator type */
    using const_reverse_iterator = typename Container::const_reverse_iterator;

    /** @brief Default constructor */
    Vec() { val_.fill(0); }

    /**
     * @brief Construct with element values
     *
     * The number of arguments provided must match Dims.
     */
    template <typename... Args>
    explicit Vec(Args... args)
    {
        static_assert(sizeof...(args) == Dims, "Incorrect number of arguments");
        std::size_t i{0};
#if __cplusplus >= 201703L
        // C++17 folding
        ((val_[i++] = args), ...);
#elif __cplusplus > 201103L
        detail::ExpandType{0, ((val_[i++] = args), 0)...};
#else
        static_assert(false, "C++ standard >= C++14 not detected");
#endif
    }

    /** @brief Copy constructor */
    template <typename Vector>
    explicit Vec(const Vector& vec)
    {
        std::copy(val_.begin(), val_.end(), std::begin(vec));
    }

    /** @brief Bounds-checked element access */
    constexpr reference at(size_type pos) { return val_.at(pos); }
    /** @brief Bounds-checked element access */
    constexpr const_reference at(size_type pos) const { return val_.at(pos); }
    /** @brief Element access */
    constexpr reference operator[](size_type i) { return val_[i]; }
    /** @brief Element access */
    constexpr const_reference operator[](size_type i) const { return val_[i]; }

    /** @brief First element */
    constexpr reference front() { return val_.front(); }
    /** @brief First element */
    constexpr const_reference front() const { return val_.front(); }
    /** @brief Last element */
    constexpr reference back() { return val_.back(); }
    /** @brief Last element */
    constexpr const_reference back() const { return val_.back(); }

    /** @brief Get a pointer to the first element of the raw data */
    constexpr pointer data() { return val_.data(); }
    /** @brief Get a pointer to the first element of the raw data */
    constexpr const_pointer data() const { return val_.data(); }

    /** @brief Get an iterator to the first element of the vector */
    constexpr iterator begin() noexcept { return val_.begin(); }
    /** @brief Get an iterator to the first element of the vector */
    constexpr const_iterator begin() const noexcept { return val_.begin(); }
    /** @brief Get an iterator to the first element of the vector */
    constexpr const_iterator cbegin() const noexcept { return val_.cbegin(); }

    /** @brief Get an iterator to one past the last element in the vector */
    constexpr iterator end() noexcept { return val_.end(); }
    /** @brief Get an iterator to one past the last element in the vector */
    constexpr const_iterator end() const noexcept { return val_.end(); }
    /** @brief Get an iterator to one past the last element in the vector */
    constexpr const_iterator cend() const noexcept { return val_.cend(); }

    /** @brief Get an iterator to the first element of the reverse vector */
    constexpr iterator rbegin() noexcept { return val_.rbegin(); }
    /** @brief Get an iterator to the first element of the vector */
    constexpr const_iterator rbegin() const noexcept { return val_.rbegin(); }
    /** @brief Get an iterator to the first element of the vector */
    constexpr const_iterator crbegin() const noexcept { return val_.crbegin(); }

    /**
     * @brief Get an iterator to one past the last element in the reverse vector
     */
    constexpr iterator rend() noexcept { return val_.rend(); }
    /**
     * @brief Get an iterator to one past the last element in the reverse vector
     */
    constexpr const_iterator rend() const noexcept { return val_.rend(); }
    /**
     * @brief Get an iterator to one past the last element in the reverse vector
     */
    constexpr const_iterator crend() const noexcept { return val_.crend(); }

    /** @brief Return whether the vector is empty (uninitialized) */
    constexpr bool empty() const noexcept { return val_.empty(); }
    /** @brief Return the number of elements in the vector */
    constexpr size_type size() const noexcept { return val_.size(); }

    /** @brief Fill the vector with a value */
    constexpr void fill(const T& value) { val_.fill(value); }
    /** @brief Swap this vector with another vector */
    constexpr void swap(Vec& other) noexcept { val_.swap(other.val_); }

    /** @brief Equality comparison operator */
    bool operator==(const Vec& rhs) const { return val_ == rhs.val_; }
    /** @brief Inequality comparison operator */
    bool operator!=(const Vec& rhs) const { return val_ != rhs.val_; }

    /** @brief Assignment operator */
    template <class Vector>
    Vec& operator=(const Vector& b)
    {
        std::size_t idx{0};
        for (auto& v : val_) {
            v = b[idx++];
        }
        return *this;
    }

    /** @brief Assignment operator for std::initializer_list */
    template <typename T2>
    Vec& operator=(const std::initializer_list<T2>& b)
    {
        auto it = b.begin();
        for (auto& v : val_) {
            v = *it;
            it++;
        }
        return *this;
    }

    /** @brief Addition assignment operator */
    template <class Vector>
    Vec& operator+=(const Vector& b)
    {
        std::size_t idx{0};
        for (auto& v : val_) {
            v += b[idx++];
        }
        return *this;
    }

    /** @brief Addition operator */
    template <class Vector>
    friend Vec operator+(Vec lhs, const Vector& rhs)
    {
        lhs += rhs;
        return lhs;
    }

    /** @brief Subtraction assignment operator */
    template <class Vector>
    Vec& operator-=(const Vector& b)
    {
        std::size_t idx{0};
        for (auto& v : val_) {
            v -= b[idx++];
        }
        return *this;
    }

    /** @brief Subtraction operator */
    template <class Vector>
    friend Vec operator-(Vec lhs, const Vector& rhs)
    {
        lhs -= rhs;
        return lhs;
    }

    /** @brief Multiplication assignment operator */
    template <
        typename T2,
        std::enable_if_t<std::is_arithmetic<T2>::value, bool> = true>
    Vec& operator*=(const T2& b)
    {
        for (auto& v : val_) {
            v *= b;
        }
        return *this;
    }

    /** @brief Multiplication operator */
    template <class Vector>
    friend Vec operator*(Vec lhs, const Vector& rhs)
    {
        lhs *= rhs;
        return lhs;
    }

    /** @brief Division assignment operator */
    template <
        typename T2,
        std::enable_if_t<std::is_arithmetic<T2>::value, bool> = true>
    Vec& operator/=(const T2& b)
    {
        for (auto& v : val_) {
            v /= b;
        }
        return *this;
    }

    /** @brief Division operator */
    template <class Vector>
    friend Vec operator/(Vec lhs, const Vector& rhs)
    {
        lhs /= rhs;
        return lhs;
    }

    /** @brief Compute the vector dot product (i.e. inner product) */
    template <class Vector>
    T dot(const Vector& v)
    {
        return OpenABF::dot(val_, v);
    }

    /** @brief Compute the vector cross product */
    template <class Vector, std::size_t D = Dims>
    std::enable_if_t<D == 3, Vec> cross(const Vector& v)
    {
        return OpenABF::cross(*this, v);
    }

    /** @brief Compute the vector magnitude */
    T magnitude() const { return OpenABF::norm(*this, Norm::L2); }

    /** @brief Return the unit vector of this vector */
    Vec unit() const { return OpenABF::normalize(*this); }

private:
    /** Values */
    Container val_{};
};

/** @brief 3D, 32-bit float vector */
using Vec3f = Vec<float, 3>;
/** @brief 3D, 64-bit float vector */
using Vec3d = Vec<double, 3>;

}  // namespace OpenABF

/** Debug: Print a vector to a std::ostream */
template <typename T, std::size_t Dims>
std::ostream& operator<<(std::ostream& os, const OpenABF::Vec<T, Dims>& vec)
{
    os << "[";
    std::size_t i{0};
    for (const auto& v : vec) {
        if (i++ > 0) {
            os << ", ";
        }
        os << v;
    }
    os << "]";
    return os;
}