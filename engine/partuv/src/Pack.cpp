#include "Pack.h"
#include <Eigen/Dense>
#include <Eigen/Core>
#include <igl/doublearea.h>

#include <vector>


// #include "uvpCore.hpp"
// #include <Eigen/Core>
#include "Component.h"
#include <unordered_map>
// // #include <uvpCore.hpp>
// #include "linmath.h"
// // #include "Component.h"


// void IglUvWrapper::add_component(
//     const Component& input_comp,
//     int group_id,
//     bool normalize_uv
// )
// {
//         Component comp = input_comp;
//         if (normalize_uv) normalize_uv_by_3d_area(comp);

//         int curr_face_num = m_FaceArray.size();
//         int curr_vert_num = m_VertArray.size();


//         const Eigen::MatrixXd &V = comp.V;
//         const Eigen::MatrixXi &F = comp.F.array() + curr_vert_num;
//         const Eigen::MatrixXd &UV = comp.UV;

//         // std::cout << "m_UV.rows() " << m_UV.rows() << std::endl;
//         Eigen::Index oldRows = m_UV.rows();
//         m_UV.conservativeResize(oldRows + UV.rows(), m_UV.cols());
//         m_UV.block(oldRows, 0, UV.rows(), UV.cols()) = UV;

//         // std::cout << "m_UV.rows() " << m_UV.rows() << std::endl;


//         if (UV.rows() != V.rows() || UV.cols() != 2)
//             throw std::runtime_error("UV must be (#V × 2)");
            



//         // add curr_vert_num to F 
//         Eigen::MatrixXi F_new = F.array() + curr_vert_num;
        


//         std::unordered_map<UvVertT, int, UvVertHashT, UvVertEqualT> vertMap;
//         const int nFaces   = F.rows();
//         const int vertsPer = F.cols();      // 3, 4, or n‑gon


//         m_FaceArray.reserve(nFaces + curr_face_num);
//         m_VertArray.reserve(curr_vert_num + V.rows());

        
//         for (int f = 0; f < nFaces; ++f)
//         {
            
//             m_FaceArray.emplace_back(f);

//             m_FaceArray.back().m_IntParams[static_cast<SizeT>(UVP_ISLAND_INT_PARAMS::GROUP)] = 0    ;
//             auto &dst = m_FaceArray.back().m_Verts;
//             dst.reserve(vertsPer);

//             for (int c = 0; c < vertsPer; ++c)
//             {
//                 const int vid = F(f, c);

//                 UvVertT uvv;
//                 uvv.m_ControlId   = vid;
//                 uvv.m_UvCoords[0] = static_cast<float>(m_UV(vid, 0));
//                 uvv.m_UvCoords[1] = static_cast<float>(m_UV(vid, 1));

//                 auto it = vertMap.find(uvv);
//                 int idx;
//                 if (it == vertMap.end())
//                 {
//                     idx = static_cast<int>(m_VertArray.size());
//                     vertMap.emplace(uvv, idx);
//                     m_VertArray.emplace_back(uvv);
//                 }
//                 else
//                     idx = it->second;

//                 dst.pushBack(idx);
//                 ++m_PolyVertexCount;
//             }
//         }

//         m_uvData.m_FaceCount  = static_cast<int>(m_FaceArray.size());
//         m_uvData.m_pFaceArray = m_FaceArray.data();
//         m_uvData.m_VertCount  = static_cast<int>(m_VertArray.size());
//         m_uvData.m_pVertArray = m_VertArray.data();
// }


// void IglUvWrapper::applyPackResult(const UvpIslandsMessageT      *islandsMsg,
//                          const UvpPackSolutionMessageT *solutionMsg)
// {
//     std::vector<Eigen::Vector2d> newUV(m_VertArray.size());
//     for (size_t i = 0; i < m_VertArray.size(); ++i)
//         newUV[i] = { m_VertArray[i].m_UvCoords[0], m_VertArray[i].m_UvCoords[1] };

//     const auto &islands = islandsMsg->m_Islands;
//     mat4x4 M;
//     for (const auto &islSol : solutionMsg->m_IslandSolutions)
//     {
//         islandSolutionToMatrix(islSol, M);
//         for (int fId : islands[islSol.m_IslandIdx])
//         {
//             for (int vIdx : m_FaceArray[fId].m_Verts)
//             {
//                 const auto &orig = m_VertArray[vIdx];
//                 vec4 p = { orig.m_UvCoords[0], orig.m_UvCoords[1], 0.f, 1.f }, q;
//                 mat4x4_mul_vec4(q, M, p);
//                 newUV[vIdx] = { q[0] / q[3], q[1] / q[3] };
//             }
//         }
//     }

//     // std::cout << "newUV.size() " << newUV.size() << std::endl;
//     // for (size_t i = 0; i < newUV.size(); ++i)
//     // {
//     //     std::cout << "newUV[" << i << "] " << newUV[i] << std::endl;
//     // }
//     // scatter back – one UV per control vertex
//     for (size_t i = 0; i < m_VertArray.size(); ++i)
//     {
//         int ctrl = m_VertArray[i].m_ControlId;
//         m_UV(ctrl,0) = newUV[i][0];
//         m_UV(ctrl,1) = newUV[i][1];
//     }
//     // UV.resize(newUV.size(), 2);
//     // for (size_t i = 0; i < newUV.size(); ++i)
//     // {
//     //     UV(i, 0) = newUV[i][0];
//     //     UV(i, 1) = newUV[i][1];
//     // }
// }


// Eigen::MatrixXd & IglUvWrapper::runPacking(){
//     UvpOperationInputT uvpInput;
//     // uvpInput.m_PackingMode = UVP_PACKING_MODE::GROUPS_TOGETHER;
//     // uvpInput.m_GroupingMethod = UVP_GROUPING_METHOD::EXTERNAL;
    
//     uvpInput.m_Opcode = UVP_OPCODE::PACK;
//     uvpInput.m_UvData = m_uvData;
//     uvpInput.m_pDeviceId   = "cpu";
//     // uvpInput.m_NormalizeIslands = true;

//     // uvpInput.m_RealtimeSolution = true;
//     // uvpInput.m_Benchmark = true;
//     // uvpInput.m_Opcode = UVP_OPCODE::PACK;
    
//     // uvpInput.m_FixedScale = true;
    
//     uvpInput.m_HeuristicSearchTime = 5;      
//     uvpInput.m_HeuristicMaxWaitTime = 5;
    
//     UvpOpExecutorT opExecutor(false);
//     opExecutor.execute(uvpInput);
//     applyPackResult(
//         static_cast<const UvpIslandsMessageT*>(opExecutor.getLastMessage(UvpMessageT::MESSAGE_CODE::ISLANDS)),
//         static_cast<const UvpPackSolutionMessageT*>(opExecutor.getLastMessage(UvpMessageT::MESSAGE_CODE::PACK_SOLUTION))
//     );
//     return m_UV;
// }


void normalize_uv_by_3d_area(Component& comp,
                              const double eps)
{
    // Eigen::MatrixXd V = comp.V;
    // Eigen::MatrixXi F = comp.F;
    // Eigen::MatrixXd UV = comp.UV;

    Eigen::VectorXd A3, A2;
    
    // Compute double area for each face in 3D and in UV domain
    igl::doublearea(comp.V, comp.F, A3);   // A3(i) = 2 * area of face i in 3D
    igl::doublearea(comp.UV, comp.F, A2);  // A2(i) = 2 * area of face i in 2D

    // Compute total (actual) areas by summing the double areas and dividing by 2
    const double total_area_3D = std::abs(A3.sum()) * 0.5;
    const double total_area_2D = std::abs(A2.sum()) * 0.5;
    
    if (total_area_3D < eps || total_area_2D < eps) return;            // degenerate – leave untouched
    const double s = std::sqrt(total_area_3D / total_area_2D);
    comp.UV *= s;     
}


// void opExecutorMessageHandler(void* m_pMessageHandlerData, UvpMessageT *pMsg)
// {
//     // This handler is called every time the packer sends a message to the application.
//     // Simply pass the message to the underlaying executor object.
//     reinterpret_cast<UvpOpExecutorT*>(m_pMessageHandlerData)->handleMessage(pMsg);
// }

