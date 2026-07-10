#pragma once

#include <stdexcept>
#include <string>

namespace OpenABF
{

/** @brief Solver exception */
class SolverException : public std::runtime_error
{
public:
    /** @brief Constructor with message */
    explicit SolverException(const char* msg) : std::runtime_error(msg) {}
    /** @brief Constructor with message */
    explicit SolverException(std::string msg) : std::runtime_error(msg) {}
};

/** @brief Solver exception */
class MeshException : public std::runtime_error
{
public:
    /** @brief Constructor with message */
    explicit MeshException(const char* msg) : std::runtime_error(msg) {}
    /** @brief Constructor with message */
    explicit MeshException(std::string msg) : std::runtime_error(msg) {}
};

}  // namespace OpenABF