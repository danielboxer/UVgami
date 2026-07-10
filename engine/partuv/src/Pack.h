#include <Eigen/Dense>
#include <Eigen/Core>

#include <vector>


// #include "uvpCore.hpp"
// #include <Eigen/Core>
#include "Component.h"
#include <unordered_map>

// #include "linmath.h"
// // #include "Component.h"


// using namespace uvpcore;

// typedef std::array<UvpMessageT*, static_cast<int>(UvpMessageT::MESSAGE_CODE::VALUE_COUNT)> UvpMessageArrayT;
// void opExecutorMessageHandler(void* m_pMessageHandlerData, UvpMessageT *pMsg);
// // Wrapper class simplifying execution of UVP operations.
// class UvpOpExecutorT
// {
// private:
//     friend void opExecutorMessageHandler(void* m_pMessageHandlerData, UvpMessageT *pMsg);

//     std::list<UvpMessageT*> m_ReceivedMessages;
//     UvpMessageArrayT m_LastMessagePerCode;

//     bool m_DebugMode;

//     void destroyMessages()
//     {       
//         // The application becomes the owner of UVP messages after receiving it,
//         // so we have to make sure they are eventually deallocated by calling
//         // the destory method on them (do not use the delete operator).
//         for (UvpMessageT *pMsg : m_ReceivedMessages)
//         {
//             pMsg->destroy();
//         }
//         m_ReceivedMessages.clear();
//     }

//     void reset()
//     {
//         destroyMessages();
//         m_LastMessagePerCode = { nullptr };
//     }

//     void handleMessage(UvpMessageT *pMsg)
//     {
//         // This method is called every time the packer sends a message to the application.
//         // We need to handle the message properly.

//         if (pMsg->m_Code == UvpMessageT::MESSAGE_CODE::PROGRESS_REPORT)
//         {
//             UvpProgressReportMessageT *pReportProgressMsg = static_cast<UvpProgressReportMessageT*>(pMsg);

//             std::cout << "[UVP PROGRESS REPORT] Phase: " << static_cast<int>(pReportProgressMsg->m_PackingPhase);
//             for (int i = 0; i < pReportProgressMsg->m_ProgressSize; i++)
//             {
//                 std::cout << ", Progress[" << i << "]: " << pReportProgressMsg->m_ProgressArray[i];
//             }
//             std::cout << "\n";
//         }
//         else if (pMsg->m_Code == UvpMessageT::MESSAGE_CODE::BENCHMARK)
//         {
//             UvpBenchmarkMessageT *pBenchmarkMsg = static_cast<UvpBenchmarkMessageT*>(pMsg);

//             std::cout << "[UVP BENCHMARK] Device name: " << pBenchmarkMsg->m_DeviceName.c_str() << ", Total packing time (ms): " <<
//                 pBenchmarkMsg->m_TotalPackTimeMs << ", Average packing time (ms): " << pBenchmarkMsg->m_AvgPackTimeMs << "\n";
//         }

//         m_LastMessagePerCode[static_cast<int>(pMsg->m_Code)] = pMsg;
//         m_ReceivedMessages.push_back(pMsg);
//     }


// public:
//     UvpOpExecutorT(bool debugMode) :
//         m_DebugMode(debugMode)
//     {}

//     ~UvpOpExecutorT()
//     {
//         destroyMessages();
//     }

//     UVP_ERRORCODE execute(UvpOperationInputT &uvpInput)
//     {
//         reset();

//         uvpInput.m_pMessageHandler = opExecutorMessageHandler;
//         uvpInput.m_pMessageHandlerData = this;

//         if (m_DebugMode)
//         {

//             const char *pValidationResult = uvpInput.validate();

//             if (pValidationResult)
//             {
//                 throw std::runtime_error("Operation input validation failed: " + std::string(pValidationResult));
//             }
//         }

//         UvpOperationT uvpOp(uvpInput);
//         // Execute the actual operation - this call will block the current thread
//         // until the operation is finished.
//         UVP_ERRORCODE retCode = uvpOp.entry();

//         return retCode;
//     }

//     UvpMessageT* getLastMessage(UvpMessageT::MESSAGE_CODE code)
//     {
//         return m_LastMessagePerCode[static_cast<int>(code)];
//     }
// };



// /* ──────────────────────────────────────────────────────────────────────────── */
// /* helper copied verbatim from sample                                          */
// inline void islandSolutionToMatrix(const UvpIslandPackSolutionT& sol, mat4x4& M)
// {
//     mat4x4_identity(M);
//     mat4x4_translate_in_place(M,  sol.m_PostScaleOffset[0], sol.m_PostScaleOffset[1], 0.0f);
//     mat4x4_scale_aniso      (M,  M, 1.0f/sol.m_Scale, 1.0f/sol.m_Scale, 1.0f);
//     mat4x4_translate_in_place(M,  sol.m_Offset[0],      sol.m_Offset[1],      0.0f);
//     mat4x4_translate_in_place(M,  sol.m_Pivot[0],       sol.m_Pivot[1],       0.0f);
//     mat4x4_rotate_Z         (M,  M, sol.m_Angle);
//     mat4x4_translate_in_place(M, -sol.m_Pivot[0],      -sol.m_Pivot[1],       0.0f);
//     mat4x4_scale_aniso      (M,  M, sol.m_PreScale,     sol.m_PreScale,       1.0f);
// }



// // -----------------------------------------------------------------------------
// class IglUvWrapper
// {
// public:
//     IglUvWrapper() : m_UV(Eigen::MatrixXd::Zero(0, 2)) {}


//     void add_component(
//         const Component& input_comp,
//         int group_id,
//         bool normalize_uv = true
//     );

//     /// Apply pack result and write back into m_UV (same shape as input UV).
//     void applyPackResult(const UvpIslandsMessageT      *islandsMsg,
//                          const UvpPackSolutionMessageT *solutionMsg);
//                         //  Eigen::MatrixXd &UV)

//     Eigen::MatrixXd &runPacking();

//     const Eigen::MatrixXd &packedUV() const { return m_UV; }

// private:
//     std::vector<UvFaceT>     m_FaceArray;
//     Eigen::MatrixXd          m_UV;
//     std::vector<UvVertT>     m_VertArray;
//     int                      m_PolyVertexCount = 0;
//     UvDataT                  m_uvData;
// };

void normalize_uv_by_3d_area(Component& comp,
                              const double eps = 1e-12);


