#pragma once

#include <string>

namespace uvgami {

// obj plus <stem>_seams sidecar in, per-island tutte + slim flatten,
// area-normalized shelf pack, obj with vt out. packOnly skips the solve and
// repacks the input map. polygons stay polygons: the solve fan-triangulates
// internally but output corners match the input faces one to one.
int runFlatten(const std::string &inputPath, const std::string &outputDir,
               int maxIterations, bool packOnly);

}  // namespace uvgami
