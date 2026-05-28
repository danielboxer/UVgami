#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <omp.h>

#include "pipeline.h"   // now declares the new signature
#include "Component.h"
#include "Config.h"

namespace py = pybind11;
static inline int to_int_or_neg1(const py::handle& h) {
    if (h.is_none()) return -1;
    return py::cast<int>(h);
}

// tree: { key: {"left": L, "right": R}, ... }
static std::vector<NodeRecord> nodes_from_py_dict(const py::dict& tree) {
    std::vector<int> keys;
    keys.reserve(tree.size());
    for (auto&& kv : tree) {
        keys.push_back(py::cast<int>(kv.first));
    }
    std::sort(keys.begin(), keys.end());

    std::vector<NodeRecord> nodes;
    nodes.reserve(keys.size());
    for (int k : keys) {
        py::dict rec = py::cast<py::dict>(tree[py::int_(k)]);
        NodeRecord n{};
        n.id    = k;
        n.left  = to_int_or_neg1(rec["left"]);
        n.right = to_int_or_neg1(rec["right"]);
        nodes.push_back(n);
    }
    return nodes;
}
// ---------------------------------------------------------------------------
// 1.  Wrapper for the “file‑based” call (optional – keep it if you still
//     have a file‑loading overload of pipeline(); otherwise delete it).
// ---------------------------------------------------------------------------
static std::pair<UVParts, std::vector<UVParts>>
pipeline_file_py(const std::string& tree_filename,
                 const std::string& mesh_filename,
                 const std::string&                       configPath,
                 double            threshold)
{
    ConfigManager::instance().loadFromFile(configPath);
    std::pair<UVParts, std::vector<UVParts>> out;

        std::vector<UVParts> parts;
        UVParts final_part = pipeline(tree_filename, mesh_filename, threshold, parts);
        out = {final_part, std::move(parts)};
    return out;
}

// ---------------------------------------------------------------------------
// 2.  Matrix‑based wrapper – no globals, no helpers, just forward.
// ---------------------------------------------------------------------------
static std::pair<UVParts, std::vector<UVParts>>
pipeline_numpy_py(const Eigen::Ref<const Eigen::MatrixXd>& V,
                  const Eigen::Ref<const Eigen::MatrixXi>& F,
                  const py::dict&                          tree_dict,
                  const std::string&                       configPath,
                  double                                   threshold,
                  bool                                     pack_final_mesh)
{
    std::pair<UVParts, std::vector<UVParts>> out;
    std::vector<NodeRecord> nodes = nodes_from_py_dict(tree_dict);
    std::vector<UVParts> parts;
    UVParts final_part = pipeline(V, F, nodes, configPath, threshold, pack_final_mesh, parts);
    out = {final_part, std::move(parts)};
    return out;
}

int omp_threads_used() {
    int used = 0;
    #pragma omp parallel
    {
        #pragma omp single
        used = omp_get_num_threads();
    }
    return used;
}

// ---------------------------------------------------------------------------
// 3.  Module definition.
// ---------------------------------------------------------------------------
PYBIND11_MODULE(_core, m)
{
    m.doc() = "UV‑unwrapping pipeline bindings";

    py::class_<Component>(m, "Component")
        .def(py::init<>())
        .def_readwrite("V",          &Component::V)
        .def_readwrite("F",          &Component::F)
        .def_readwrite("UV",         &Component::UV)
        .def_readwrite("distortion", &Component::distortion)
        .def_readwrite("index",      &Component::index)
        .def_readwrite("faces",      &Component::faces)
        .def("save_mesh",            &Component::save_mesh)
        .def("__add__",              &Component::operator+);

    // ------------ UVParts ---------------------------------------------------
    py::class_<UVParts>(m, "UVParts")
        .def(py::init<>())
        .def(py::init<const std::vector<Component>&>())
        .def(py::init<int>())
        .def_readwrite("components",     &UVParts::components)
        .def_readwrite("distortion",     &UVParts::distortion)
        .def_readwrite("num_components", &UVParts::num_components)
        .def_property_readonly(
            "hierarchy_json",
            [](const UVParts& p) { return p.hierarchy.to_json().dump(); }
        )
        .def("getUV",        &UVParts::getUV)
        .def("getNumFaces",  &UVParts::getNumFaces)
        .def("to_components",&UVParts::to_components)
        .def("__add__",      &UVParts::operator+)
        .def("__eq__",       &UVParts::operator==)
        .def("__ne__",       &UVParts::operator!=);

    // ------------ Pipeline --------------------------------------------------
    // Keep the file‑based front end only if you still have that overload.
    m.def("pipeline",
          &pipeline_file_py,
          py::arg("tree_filename"),
          py::arg("mesh_filename"),
          py::arg("configPath"),
          py::arg("threshold") = 1.25,
          py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
Run the UV‑unwrapping pipeline that loads the mesh from disk.
Returns (final_part, individual_parts).
)pbdoc");

    m.def("pipeline_numpy",
          &pipeline_numpy_py,
          py::arg("V"),
          py::arg("F"),
          py::arg("tree_dict"),
          py::arg("configPath"),
          py::arg("threshold") = 1.25,
          py::arg("pack_final_mesh") = false,
          R"pbdoc(
Run the UV‑unwrapping pipeline directly on numpy arrays V (Nx3) and F (Mx3).
Returns (final_part, individual_parts).
)pbdoc");

    m.def("omp_threads_used", &omp_threads_used, "Return OpenMP team size for a dummy region");
}
