#include <cfloat>
#include <string>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <thread>

#include "uvgami.h"
#include "Flatten.hpp"
#include "IglUtils.hpp"
#include "Optimizer.hpp"
#include "SymDirichletEnergy.hpp"

#include <igl/cut_to_disk.h>
#include <igl/cut_mesh.h>
#include <igl/readOFF.h>
#include <igl/boundary_loop.h>
#include <igl/map_vertices_to_circle.h>
#include <igl/harmonic.h>
#include <igl/arap.h>
#include <igl/avg_edge_length.h>
#include <igl/euler_characteristic.h>
#include <igl/edge_lengths.h>
#include <igl/is_vertex_manifold.h>
#include <igl/is_edge_manifold.h>
#include <igl/facet_components.h>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>

#define TCLAP_NAMESTARTSTRING "-"
#include "tclap/CmdLine.h"

Eigen::MatrixXd V, UV, N;
Eigen::MatrixXi F, FUV, FN;

// optimization
std::vector<const uvgami::TriMesh *> triSoup;
int vertAmt_input;
uvgami::TriMesh triSoup_backup;
uvgami::Optimizer *optimizer;
std::vector<uvgami::Energy *> energyTerms;
std::vector<double> energyParams;

bool rand1PInitCut = false;
// pinned runs can chase an unreachable distortion bound forever, capped below
bool pinnedMode = false;
// relax the kept map without any split/merge, stops at the first stationary point
bool noCutMode = false;
// greedily merge islands along shared mesh edges, no splits ever
bool stitchMode = false;
int stationaryCount = 0;
double lambda_init = 0.999;
bool optimization_on = false;
int iterNum = 0;
int converged = 0;
double fracThres = 0.0;
bool topoLineSearch = true;
int initCutOption = 0;
bool outerLoopFinished = false;
double upperBound = 4.1;
const double convTol_upperBound = 1.0e-3;

std::vector<std::pair<double, double>> energyChanges_bSplit,
    energyChanges_iSplit, energyChanges_merge;
std::vector<std::vector<int>> paths_bSplit, paths_iSplit, paths_merge;
std::vector<Eigen::MatrixXd> newVertPoses_bSplit, newVertPoses_iSplit,
    newVertPoses_merge;

int opType_queried = -1;
std::vector<int> path_queried;
Eigen::MatrixXd newVertPos_queried;
bool reQuery = false;
double filterExp_in = 0.6;
int inSplitTotalAmt;

// std::ofstream logFile;
std::string outputFolderPath;
std::string meshName;

const int channel_initial = 0;
const int channel_result = 1;
bool canSaveMesh = false;
bool mute = true;
std::atomic<bool> forceQuit = false;
std::atomic<bool> forceQuitSave = false;
std::atomic<bool> snapshot = false;
int maxSeamWeight = 100;
int maxFaceWeight = 10;

const char *pathSeparator() {
#ifdef _WIN32
    return "\\";
#else
    return "/";
#endif
}
uvgami::ChronoTimer mainTimer("unwrap time");

void stdin_listener() {
    std::string line;
    do {
        std::cin >> line;
        if (line == "stop") {
            forceQuit = true;
            forceQuitSave = true;
        } else if (line == "cancel") {
            forceQuit = true;
            forceQuitSave = false;
        } else if (line == "snapshot") {
            snapshot = true;
        }
    } while (!line.empty());
}

void proceedOptimization(int proceedNum) {
    for (int proceedI = 0; (proceedI < proceedNum) && converged == 0;
         proceedI++) {
        converged = optimizer->solve(1);
        iterNum = optimizer->getIterNum();
    }
}

// per-iteration reporting: emits the progress line the addon parses and serves
// the snapshot command
void reportProgress(void) {
    Eigen::VectorXd distortionPerElem;
    energyTerms[0]->getEnergyValPerElem(*triSoup[channel_result],
                                        distortionPerElem, true);
    uvgami::IglUtils::reportDistortion(distortionPerElem, 4.0, 8.5);

    if (snapshot) {
        triSoup[channel_result]->saveAsMesh(F, true);
        snapshot = false;
    }
}

bool postDrawFunc(void) {
    if (iterNum == 0) {
        optimization_on = !optimization_on;
        if (optimization_on && converged)
            optimization_on = false;
    }
    if (forceQuit) {
        canSaveMesh = forceQuitSave;
        outerLoopFinished = true;
    }
    if (canSaveMesh) {
        // save mesh
        if (outerLoopFinished) {
            if (!triSoup[channel_result]->saveAsMesh(outputFolderPath, F, true))
                std::cerr << "Unable to save mesh" << std::endl;
            mainTimer.finish();
        }
        canSaveMesh = false;
    }

    if (outerLoopFinished)
        return true;

    return false;
}

int computeOptPicked(
    const std::vector<std::pair<double, double>> &energyChanges0,
    const std::vector<std::pair<double, double>> &energyChanges1,
    double lambda) {
    assert(!energyChanges0.empty());
    assert(!energyChanges1.empty());
    assert((lambda >= 0.0) && (lambda <= 1.0));

    double minEChange0 = DBL_MAX;
    for (int ecI = 0; ecI < energyChanges0.size(); ecI++) {
        if ((energyChanges0[ecI].first == DBL_MAX) ||
            (energyChanges0[ecI].second == DBL_MAX))
            continue;
        double EwChange = energyChanges0[ecI].first * (1.0 - lambda) +
                          energyChanges0[ecI].second * lambda;
        if (EwChange < minEChange0)
            minEChange0 = EwChange;
    }
    double minEChange1 = DBL_MAX;
    for (int ecI = 0; ecI < energyChanges1.size(); ecI++) {
        if ((energyChanges1[ecI].first == DBL_MAX) ||
            (energyChanges1[ecI].second == DBL_MAX))
            continue;
        double EwChange = energyChanges1[ecI].first * (1.0 - lambda) +
                          energyChanges1[ecI].second * lambda;
        if (EwChange < minEChange1)
            minEChange1 = EwChange;
    }

    assert((minEChange0 != DBL_MAX) || (minEChange1 != DBL_MAX));
    return (minEChange0 > minEChange1);
}

int computeBestCand(const std::vector<std::pair<double, double>> &energyChanges,
                    double lambda, double &bestEChange) {
    assert((lambda >= 0.0) && (lambda <= 1.0));

    bestEChange = DBL_MAX;
    int id_minEChange = -1;
    for (int ecI = 0; ecI < energyChanges.size(); ecI++) {
        if ((energyChanges[ecI].first == DBL_MAX) ||
            (energyChanges[ecI].second == DBL_MAX))
            continue;
        double EwChange = energyChanges[ecI].first * (1.0 - lambda) +
                          energyChanges[ecI].second * lambda;
        if (EwChange < bestEChange) {
            bestEChange = EwChange;
            id_minEChange = ecI;
        }
    }

    return id_minEChange;
}

// with a pinned border there may be no boundary-split candidates at all, and
// the lambda loops below would compare against DBL_MAX sentinels forever
bool hasValidCand(const std::vector<std::pair<double, double>> &energyChanges) {
    for (const auto &candI : energyChanges) {
        if ((candI.first != DBL_MAX) && (candI.second != DBL_MAX))
            return true;
    }
    return false;
}

bool checkCand(const std::vector<std::pair<double, double>> &energyChanges) {
    for (const auto &candI : energyChanges) {
        if ((candI.first < 0.0) || (candI.second < 0.0))
            return true;
    }
    double minEChange = DBL_MAX;
    for (const auto &candI : energyChanges) {
        if (candI.first < minEChange)
            minEChange = candI.first;
        if (candI.second < minEChange)
            minEChange = candI.second;
    }
    // DISABLE std::cout << "candidates not valid, minEChange: " << minEChange
    // << std::endl;

    return false;
}

double updateLambda(double measure_bound, double lambda_SD = energyParams[0],
                    double kappa = 1.0, double kappa2 = 1.0) {
    lambda_SD =
        (std::max)(0.0, kappa * (measure_bound -
                                 (upperBound - convTol_upperBound / 2.0)) +
                            kappa2 * lambda_SD / (1.0 - lambda_SD));
    return lambda_SD / (1.0 + lambda_SD);
}

bool updateLambda_stationaryV(bool cancelMomentum = true,
                              bool checkConvergence = false) {
    Eigen::MatrixXd edgeLengths;
    igl::edge_lengths(triSoup[channel_result]->V_rest,
                      triSoup[channel_result]->F, edgeLengths);
    const double eps_E_se = 1.0e-3 * edgeLengths.minCoeff() /
                            triSoup[channel_result]->virtualRadius;

    // measurement and energy value computation
    const double E_SD = optimizer->getLastEnergyVal(true) / energyParams[0];
    double E_se;
    triSoup[channel_result]->computeSeamSparsity(E_se);
    E_se /= triSoup[channel_result]->virtualRadius;
    double stretch_l2, stretch_inf, stretch_shear, compress_inf;
    triSoup[channel_result]->computeStandardStretch(
        stretch_l2, stretch_inf, stretch_shear, compress_inf);
    double measure_bound = E_SD;
    const double eps_lambda =
        (std::min)(1.0e-3,
                   std::abs(updateLambda(measure_bound) - energyParams[0]));

    // TODO?: stop when first violates bounds from feasible, don't go to best
    // feasible. check after each merge whether distortion is violated
    //  oscillation detection
    static int iterNum_bestFeasible = -1;
    static uvgami::TriMesh triSoup_bestFeasible;
    static double E_se_bestFeasible = DBL_MAX;
    static int lastStationaryIterNum =
        0; // still necessary because boundary and interior query are with same
           // iterNum
    static std::map<double, std::vector<std::pair<double, double>>>
        configs_stationaryV;
    if (iterNum != lastStationaryIterNum) {
        // not a roll back config
        const double lambda = 1.0 - energyParams[0];
        bool oscillate = false;
        const auto low = configs_stationaryV.lower_bound(E_se);
        if (low == configs_stationaryV.end()) {
            // all less than E_se
            if (!configs_stationaryV.empty()) {
                // use largest element
                if (std::abs(configs_stationaryV.rbegin()->first - E_se) <
                    eps_E_se) {
                    for (const auto &lambdaI :
                         configs_stationaryV.rbegin()->second) {
                        if ((std::abs(lambdaI.first - lambda) < eps_lambda) &&
                            (std::abs(lambdaI.second - E_SD) < eps_E_se)) {
                            oscillate = true;
                            // DISABLE logFile <<
                            // configs_stationaryV.rbegin()->first << ", " <<
                            // lambdaI.second << std::endl; DISABLE logFile <<
                            // E_se << ", " << lambda << ", " << E_SD <<
                            // std::endl;
                            break;
                        }
                    }
                }
            }
        } else if (low == configs_stationaryV.begin()) {
            // all not less than E_se
            if (std::abs(low->first - E_se) < eps_E_se) {
                for (const auto &lambdaI : low->second) {
                    if ((std::abs(lambdaI.first - lambda) < eps_lambda) &&
                        (std::abs(lambdaI.second - E_SD) < eps_E_se)) {
                        oscillate = true;
                        // DISABLE logFile << low->first << ", " <<
                        // lambdaI.first << ", " << lambdaI.second << std::endl;
                        // DISABLE logFile << E_se << ", " << lambda << ", " <<
                        // E_SD << std::endl;
                        break;
                    }
                }
            }
        } else {
            const auto prev = std::prev(low);
            if (std::abs(low->first - E_se) < eps_E_se) {
                for (const auto &lambdaI : low->second) {
                    if ((std::abs(lambdaI.first - lambda) < eps_lambda) &&
                        (std::abs(lambdaI.second - E_SD) < eps_E_se)) {
                        oscillate = true;
                        // DISABLE logFile << low->first << ", " <<
                        // lambdaI.first << ", " << lambdaI.second << std::endl;
                        // DISABLE logFile << E_se << ", " << lambda << ", " <<
                        // E_SD << std::endl;
                        break;
                    }
                }
            }
            if ((!oscillate) && (std::abs(prev->first - E_se) < eps_E_se)) {
                for (const auto &lambdaI : prev->second) {
                    if ((std::abs(lambdaI.first - lambda) < eps_lambda) &&
                        (std::abs(lambdaI.second - E_SD) < eps_E_se)) {
                        oscillate = true;
                        // DISABLE logFile << prev->first << ", " <<
                        // lambdaI.first << ", " << lambdaI.second << std::endl;
                        // DISABLE logFile << E_se << ", " << lambda << ", " <<
                        // E_SD << std::endl;
                        break;
                    }
                }
            }
        }
        // record best feasible UV map
        if ((measure_bound <= upperBound) && (E_se < E_se_bestFeasible)) {
            iterNum_bestFeasible = iterNum;
            triSoup_bestFeasible = *triSoup[channel_result];
            E_se_bestFeasible = E_se;
        }
        if (oscillate && (iterNum_bestFeasible >= 0)) {
            // arrive at the best feasible config again
            // DISABLE logFile << "oscillation detected at measure = " <<
            // measure_bound << ", b = " << upperBound <<
            //    "lambda = " << energyParams[0] << std::endl;
            // DISABLE logFile << lastStationaryIterNum << ", " << iterNum <<
            // std::endl;
            if (iterNum_bestFeasible != iterNum) {
                optimizer->setConfig(triSoup_bestFeasible, iterNum,
                                     optimizer->getTopoIter());
                // DISABLE logFile << "rolled back to best feasible in iter " <<
                // iterNum_bestFeasible << std::endl;
            }
            return false;
        } else if (oscillate) {
            // no feasible config yet, so there is nothing to roll back to,
            // but revisiting the same stationary state means the split/merge
            // pair is cycling and spinning further cannot reach the bound.
            // give it a few chances to escape, then keep the current map
            static int oscillated_infeasible = 0;
            if (++oscillated_infeasible >= 3)
                return false;
        } else {
            configs_stationaryV[E_se].emplace_back(
                std::pair<double, double>(lambda, E_SD));
        }
    }
    lastStationaryIterNum = iterNum;
    // convergence check
    if (checkConvergence) {
        if (measure_bound <= upperBound) {
            // save info at first feasible stationaryVT for comparison
            static bool saved = false;
            if (!saved) {
                //                logFile << "saving firstFeasibleS..." <<
                //                std::endl; saveScreenshot(outputFolderPath +
                //                "firstFeasibleS.png", 0.5, false, true);
                //                //TODO: saved is before roll back...
                //                triSoup[channel_result]->saveAsMesh(outputFolderPath
                //                + "firstFeasibleS_mesh.obj", F);
                saved = true;
                //              logFile << "firstFeasibleS saved" << std::endl;
            }
            if (measure_bound >= upperBound - convTol_upperBound) {
                // DISABLE logFile << "all converged at measure = " <<
                // measure_bound << ", b = " << upperBound <<
                //    " lambda = " << energyParams[0] << std::endl;
                if (iterNum_bestFeasible != iterNum) {
                    assert(iterNum_bestFeasible >= 0);
                    optimizer->setConfig(triSoup_bestFeasible, iterNum,
                                         optimizer->getTopoIter());
                    // DISABLE logFile << "rolled back to best feasible in iter
                    // " << iterNum_bestFeasible << std::endl;
                }
                return false;
            }
        }
    }

    // lambda update (dual update)
    energyParams[0] = updateLambda(measure_bound);
    // TODO: needs to be careful on lambda update space

    // critical lambda scheme
    if (checkConvergence) {
        // update lambda until feasible update on T might be triggered
        if (measure_bound > upperBound) {
            // need to cut further, increase energyParams[0]
            // DISABLE logFile << "curUpdated = " << energyParams[0] << ",
            // increase" << std::endl;
            if ((!energyChanges_merge.empty()) &&
                hasValidCand(energyChanges_bSplit) &&
                (computeOptPicked(energyChanges_bSplit, energyChanges_merge,
                                  1.0 - energyParams[0]) == 1)) {
                // still picking merge. the dual update saturates in double
                // precision (x/(1+x) sticks at 1), and with a pinned border
                // the pick can stay merge at every lambda, so break at the
                // fixed point instead of spinning
                double lambda_last = -1.0;
                do {
                    energyParams[0] = updateLambda(measure_bound);
                    if (energyParams[0] == lambda_last)
                        break;
                    lambda_last = energyParams[0];
                } while (
                    (computeOptPicked(energyChanges_bSplit, energyChanges_merge,
                                      1.0 - energyParams[0]) == 1));
                // DISABLE logFile << "iterativelyUpdated = " << energyParams[0]
                // << ", increase for switch" << std::endl;
            }

            if (!checkCand(energyChanges_iSplit) &&
                !checkCand(energyChanges_bSplit)) {
                // if filtering too strong
                reQuery = true;
                // DISABLE logFile << "enlarge filtering!" << std::endl;
            } else {
                double eDec_b, eDec_i;
                assert(!(energyChanges_bSplit.empty() &&
                         energyChanges_iSplit.empty()));
                int id_pickingBSplit = computeBestCand(
                    energyChanges_bSplit, 1.0 - energyParams[0], eDec_b);
                int id_pickingISplit = computeBestCand(
                    energyChanges_iSplit, 1.0 - energyParams[0], eDec_i);
                // break at the dual update's fixed point, pins can leave no
                // split profitable at any lambda
                double lambda_last = -1.0;
                while ((eDec_b > 0.0) && (eDec_i > 0.0)) {
                    if (energyParams[0] == lambda_last)
                        break;
                    lambda_last = energyParams[0];
                    energyParams[0] = updateLambda(measure_bound);
                    id_pickingBSplit = computeBestCand(
                        energyChanges_bSplit, 1.0 - energyParams[0], eDec_b);
                    id_pickingISplit = computeBestCand(
                        energyChanges_iSplit, 1.0 - energyParams[0], eDec_i);
                }
                if (id_pickingBSplit < 0 && id_pickingISplit < 0) {
                    // no pickable split at all, widen the filter instead
                    reQuery = true;
                } else if ((id_pickingISplit < 0) || (eDec_b <= 0.0) ||
                           ((id_pickingBSplit >= 0) && (eDec_b <= eDec_i))) {
                    opType_queried = 0;
                    path_queried = paths_bSplit[id_pickingBSplit];
                    newVertPos_queried = newVertPoses_bSplit[id_pickingBSplit];
                } else {
                    opType_queried = 1;
                    path_queried = paths_iSplit[id_pickingISplit];
                    newVertPos_queried = newVertPoses_iSplit[id_pickingISplit];
                }
                // DISABLE logFile << "iterativelyUpdated = " << energyParams[0]
                // << ", increased, current eDec = " <<
                //   eDec_b << ", " << eDec_i << "; id: " << id_pickingBSplit <<
                //   ", " << id_pickingISplit << std::endl;
            }
        } else {
            bool noOp = true;
            for (const auto ecI : energyChanges_merge) {
                if (ecI.first != DBL_MAX) {
                    noOp = false;
                    break;
                }
            }
            if (noOp) {
                // DISABLE logFile << "No merge operation available, end
                // process!" << std::endl;
                energyParams[0] = 1.0 - eps_lambda;
                optimizer->updateEnergyData(true, false, false);
                if (iterNum_bestFeasible != iterNum)
                    optimizer->setConfig(triSoup_bestFeasible, iterNum,
                                         optimizer->getTopoIter());
                return false;
            }
            // DISABLE logFile << "curUpdated = " << energyParams[0] << ",
            // decrease" << std::endl;
            //!!! also account for iSplit for this switch?
            if (hasValidCand(energyChanges_bSplit) &&
                computeOptPicked(energyChanges_bSplit, energyChanges_merge,
                                 1.0 - energyParams[0]) == 0) {
                // still picking split, break at the dual update's fixed point
                // (see the merge loop above)
                double lambda_last = -1.0;
                do {
                    energyParams[0] = updateLambda(measure_bound);
                    if (energyParams[0] == lambda_last)
                        break;
                    lambda_last = energyParams[0];
                } while (computeOptPicked(energyChanges_bSplit,
                                          energyChanges_merge,
                                          1.0 - energyParams[0]) == 0);

                // DISABLE logFile << "iterativelyUpdated = " << energyParams[0]
                // << ", decrease for switch" << std::endl;
            }

            double eDec_m;
            assert(!energyChanges_merge.empty());
            int id_pickingMerge = computeBestCand(
                energyChanges_merge, 1.0 - energyParams[0], eDec_m);
            // break at the dual update's fixed point (see the merge loop in
            // the increase branch)
            double lambda_last = -1.0;
            while (eDec_m > 0.0) {
                if (energyParams[0] == lambda_last)
                    break;
                lambda_last = energyParams[0];
                energyParams[0] = updateLambda(measure_bound);
                id_pickingMerge = computeBestCand(
                    energyChanges_merge, 1.0 - energyParams[0], eDec_m);
            }
            if (id_pickingMerge < 0) {
                // a merge can be listed but unpickable (partial DBL_MAX
                // sentinel), treat it like the noOp case above
                energyParams[0] = 1.0 - eps_lambda;
                optimizer->updateEnergyData(true, false, false);
                if (iterNum_bestFeasible != iterNum)
                    optimizer->setConfig(triSoup_bestFeasible, iterNum,
                                         optimizer->getTopoIter());
                return false;
            }
            opType_queried = 2;
            path_queried = paths_merge[id_pickingMerge];
            newVertPos_queried = newVertPoses_merge[id_pickingMerge];

            // DISABLE logFile << "iterativelyUpdated = " << energyParams[0] <<
            // ", decreased, current eDec = " << eDec_m << std::endl;
        }
    }
    // lambda value sanity check
    if (energyParams[0] > 1.0 - eps_lambda)
        energyParams[0] = 1.0 - eps_lambda;
    if (energyParams[0] < eps_lambda)
        energyParams[0] = eps_lambda;

    optimizer->updateEnergyData(true, false, false);

    // DISABLE logFile << "measure = " << measure_bound << ", b = " <<
    // upperBound << ", updated lambda = " << energyParams[0] << std::endl;
    return true;
}

void converge_preDrawFunc(void) {
    reportProgress();
    optimization_on = false;
    // std::cout << "optimization converged, in " << secPast << "s." <<
    // std::endl;
    outerLoopFinished = true;
}

// solve() only reports convergence on the gradient tolerance or a line
// search stall at a healthy step size. degenerate geometry can decrease the
// energy by denormal amounts at tiny steps forever, hitting neither, so a
// long flat run counts as stationary
// measured: a dead solve decreases by exactly zero for thousands of
// iterations, a healthy one never below ~4e-8, so 1e-12 clears both
const int SOLVE_STALL_ITERATION_CAP = 100;
const double SOLVE_STALL_RELATIVE_TOLERANCE = 1.0e-12;

bool preDrawFunc(void) {
    if (optimization_on) {
        static int stalledSolveIterations = 0;
        static double stalledSolveEnergy = -1.0;
        while (!converged) {
            proceedOptimization(1);
            // check per iteration, not per phase: a stop or viewer request
            // during a long solve must not wait for convergence
            if (forceQuit)
                // postDrawFunc saves the current map and exits
                return false;
            if (snapshot)
                reportProgress();
            const double energy = optimizer->getLastEnergyVal(true);
            if (std::abs(energy - stalledSolveEnergy) <=
                SOLVE_STALL_RELATIVE_TOLERANCE * std::abs(stalledSolveEnergy)) {
                if (++stalledSolveIterations >= SOLVE_STALL_ITERATION_CAP)
                    converged = 1;
            } else {
                stalledSolveIterations = 0;
                stalledSolveEnergy = energy;
            }
        }
        stalledSolveIterations = 0;
        stalledSolveEnergy = -1.0;
        reportProgress();

        // give postDraw option to save mesh
        canSaveMesh = true;

        double stretch_l2, stretch_inf, stretch_shear, compress_inf;
        triSoup[channel_result]->computeStandardStretch(
            stretch_l2, stretch_inf, stretch_shear, compress_inf);
        double measure_bound =
            optimizer->getLastEnergyVal(true) / energyParams[0];
        if (converged == 2) {
            converged = 0;
            return false;
        }
        // if necessary, turn on scaffolding for random one point initial cut
        if (!optimizer->isScaffolding() && rand1PInitCut)
            optimizer->setScaffolding(true);

        // everything past this point queries cuts, a nocut run is done at the
        // first stationary point of the kept map
        if (noCutMode) {
            converge_preDrawFunc();
            return false;
        }

        // a stitch run merges two islands at a time, re-converging in between
        // so the zip and relaxation settle before the next placement. no
        // cuts are ever queried, done when nothing fits and no blocked
        // front loosened up
        if (stitchMode) {
            bool changed = optimizer->zipStitched();
            if (optimizer->stitchIslands())
                changed = true;
            if (changed)
                converged = 0;
            else
                converge_preDrawFunc();
            return false;
        }

        // a pinned run is done at the first feasible stationary state. the
        // full search only tightens distortion up to the bound by merging
        // cuts back, and a pinned border leaves no productive merge, so it
        // spins on an unchanged map until the no-progress cap
        if (pinnedMode && measure_bound <= upperBound) {
            converge_preDrawFunc();
            return false;
        }

        double E_se;
        triSoup[channel_result]->computeSeamSparsity(E_se);
        E_se /= triSoup[channel_result]->virtualRadius;

        if (pinnedMode && ++stationaryCount > 500) {
            converge_preDrawFunc();
            return false;
        }

        // a queued split the line search rejects leaves the map untouched,
        // the solve re-converges to the same stationary state and the same
        // op gets picked again, with lambda creeping one dual step per round
        // (each step is exactly eps_lambda, so oscillation detection can
        // never see a revisit). unchanged seam energy and vertex count mean
        // nothing is moving, stop with the map we have. distortion stays out
        // of the check: the lambda renormalization wobbles it ~1e-6 relative
        // on a frozen map, and a real op always moves one of the other two
        static int noProgressCount = 0;
        static double E_se_last = -1.0;
        static Eigen::Index V_last = -1;
        if (std::abs(E_se - E_se_last) <= 1.0e-9 * std::abs(E_se_last) &&
            triSoup[channel_result]->V_rest.rows() == V_last) {
            // lambda creep is ~1e-6 per frozen round, far too small to flip
            // a pick the pins already blocked, so pinned runs get 3 rounds
            if (++noProgressCount >= (pinnedMode ? 3 : 50)) {
                converge_preDrawFunc();
                return false;
            }
        } else {
            noProgressCount = 0;
            E_se_last = E_se;
            V_last = triSoup[channel_result]->V_rest.rows();
        }

        // continue to split boundary
        if (!updateLambda_stationaryV()) {
            // oscillation detected
            converge_preDrawFunc();
        } else {
            // DISABLE logFile << "boundary op V " <<
            // triSoup[channel_result]->V_rest.rows() << std::endl;
            if (optimizer->createFracture(fracThres, false, topoLineSearch)) {
                converged = 0;
            } else {
                // if no boundary op, try interior split if split is the current
                // best boundary op
                if ((measure_bound > upperBound) &&
                    optimizer->createFracture(fracThres, false, topoLineSearch,
                                              true)) {
                    // DISABLE logFile << "interior split " <<
                    // triSoup[channel_result]->V_rest.rows() << std::endl;
                    converged = 0;
                } else {
                    if (!updateLambda_stationaryV(false, true)) {
                        // all converged
                        converge_preDrawFunc();
                    } else {
                        // split or merge after lambda update
                        if (reQuery) {
                            bool found = false;
                            do {
                                // log(0) and log(1) would make this step 0 or
                                // inf, a tiny pinned patch has 0 or 1 interior
                                // candidates, so saturate outright
                                if (inSplitTotalAmt >= 2) {
                                    filterExp_in +=
                                        std::log(2.0) /
                                        std::log(inSplitTotalAmt);
                                    filterExp_in =
                                        (std::min)(1.0, filterExp_in);
                                } else {
                                    filterExp_in = 1.0;
                                }
                                found = optimizer->createFracture(
                                    fracThres, false, topoLineSearch, true);
                            } while (!found && filterExp_in < 1.0);
                            reQuery = false;
                            // TODO: set filtering param back?
                            if (!found) {
                                // a pinned border can leave nothing left to
                                // split, stop at the best map found
                                converge_preDrawFunc();
                                return false;
                            }
                        } else {
                            optimizer->createFracture(
                                opType_queried, path_queried,
                                newVertPos_queried, topoLineSearch);
                        }
                        opType_queried = -1;
                        converged = 0;
                    }
                }
            }
        }
    }
    return false;
}

static std::vector<float> split(const std::string &str, char sep) {
    std::vector<float> tokens;

    float i;
    std::stringstream ss(str);
    while (ss >> i) {
        tokens.push_back(i);
        if (ss.peek() == sep) {
            ss.ignore();
        }
    }

    return tokens;
}

// reads a one-line "index,weight,..." sidecar into out, skipping out-of-range
// indices. returns false when the file doesn't exist
static bool loadWeightSidecar(const std::string &filePath,
                              Eigen::VectorXd &out) {
    std::ifstream file(filePath);
    if (!file.is_open())
        return false;
    std::string line;
    getline(file, line);
    std::vector<float> tokens = split(line, ',');
    for (uint32_t i = 0; i + 1 < tokens.size(); i += 2) {
        const int selected = (int)tokens[i];
        if (selected >= 0 && selected < out.size())
            out[selected] = tokens[i + 1];
    }
    return true;
}

// a chart is disk-topology when its euler characteristic is 1. imported UV
// charts that aren't disks have to be cut before they can be flattened
static std::vector<bool> chartDiskFlags(const Eigen::MatrixXi &F,
                                        int n_components,
                                        const Eigen::VectorXi &C) {
    std::vector<std::set<int>> verts(n_components);
    std::vector<std::set<std::pair<int, int>>> edges(n_components);
    std::vector<int> faces(n_components, 0);
    for (int triI = 0; triI < F.rows(); ++triI) {
        int c = C[triI];
        ++faces[c];
        for (int i = 0; i < 3; ++i) {
            int a = F(triI, i), b = F(triI, (i + 1) % 3);
            verts[c].insert(a);
            edges[c].insert(std::pair<int, int>(std::min(a, b), std::max(a, b)));
        }
    }
    std::vector<bool> isDisk(n_components);
    for (int c = 0; c < n_components; ++c) {
        isDisk[c] = static_cast<int>(verts[c].size()) -
                        static_cast<int>(edges[c].size()) + faces[c] ==
                    1;
    }
    return isDisk;
}

// scale-free symmetric Dirichlet of an imported map, face weights included.
// the input arrives packed, so its uniform scale is arbitrary and the raw
// energy there is meaningless: E(s) = s^2*grow + shrink/s^2 has a closed
// form minimum, measure at it
static double importedMapMeasure(const uvgami::TriMesh &mesh) {
    double grow = 0.0, shrink = 0.0, total = 0.0;
    for (int triI = 0; triI < mesh.F.rows(); ++triI) {
        const Eigen::RowVector3i &tri = mesh.F.row(triI);
        const Eigen::RowVector3d e1 =
            mesh.V_rest.row(tri[1]) - mesh.V_rest.row(tri[0]);
        const Eigen::RowVector3d e2 =
            mesh.V_rest.row(tri[2]) - mesh.V_rest.row(tri[0]);
        const double l1 = e1.norm();
        const double area = e1.cross(e2).norm() / 2;
        if (l1 <= 0.0 || area <= 0.0)
            return DBL_MAX;
        const double x2 = e1.dot(e2) / l1;
        const double y2 = 2 * area / l1;
        const Eigen::RowVector2d u1 = mesh.V.row(tri[1]) - mesh.V.row(tri[0]);
        const Eigen::RowVector2d u2 = mesh.V.row(tri[2]) - mesh.V.row(tri[0]);
        const double a = u1[0] / l1;
        const double b = (u2[0] - x2 * a) / y2;
        const double c = u1[1] / l1;
        const double d = (u2[1] - x2 * c) / y2;
        const double det = a * d - b * c;
        if (det <= 0.0)
            return DBL_MAX;
        const double frob2 = a * a + b * b + c * c + d * d;
        const double weighted = area * mesh.faceWeight[triI];
        grow += weighted * frob2;
        shrink += weighted * frob2 / (det * det);
        total += area;
    }
    if (total <= 0.0)
        return DBL_MAX;
    return 2 * std::sqrt(grow * shrink) / total;
}

// an escaped exception fast-fails with no message otherwise, name it on
// stderr so a field crash is diagnosable from the addon's log
static void reportTerminate() {
    if (auto e = std::current_exception()) {
        try {
            std::rethrow_exception(e);
        } catch (const std::exception &ex) {
            std::cerr << "fatal: " << ex.what() << std::endl;
        } catch (...) {
            std::cerr << "fatal: non-std exception" << std::endl;
        }
    } else {
        std::cerr << "fatal: terminate without exception" << std::endl;
    }
    std::_Exit(90);
}

int main(int argc, char *argv[]) {
    std::set_terminate(reportTerminate);
    std::string meshFileName;
    lambda_init = 0.999;
    std::filesystem::path inputFolderPath;
    bool hasUV = false;
    bool ignoreUV = false;
    bool flattenMode = false;
    bool packOnlyMode = false;
    int flattenIters = 10;

    try {
        TCLAP::CmdLine cmd("uvgami command line", ' ', "1.1.2");
        TCLAP::ValueArg<std::string> inputArg("i", "input", "Input mesh", true,
                                              "", "string", cmd);
        TCLAP::ValueArg<std::string> outputArg(
            "o", "output", "Output directory", false, "", "string", cmd);
        TCLAP::ValueArg<double> lambdaInitArg("L", "lambda_init",
                                              "Lambda initial value", false, 0,
                                              "double", cmd);
        TCLAP::ValueArg<double> upperBoundArg("u", "upper_bound", "Upper bound",
                                              false, 0, "double", cmd);
        TCLAP::ValueArg<uint32_t> maxSeamWeightArg("s", "max_seam_weight",
                                                   "Maximum seam weight", false,
                                                   0, "uint32_t", cmd);
        TCLAP::ValueArg<uint32_t> maxFaceWeightArg(
            "w", "max_face_weight", "Maximum face importance weight", false, 0,
            "uint32_t", cmd);
        TCLAP::SwitchArg ignoreUVArg("g", "ignore_uv", "Ignore UV map", cmd);
        TCLAP::SwitchArg flattenArg(
            "", "flatten",
            "Flatten and pack along the _seams sidecar, no optimization", cmd);
        TCLAP::ValueArg<int> flattenItersArg(
            "", "flatten_iters", "SLIM iterations in flatten mode", false, 10,
            "int", cmd);
        TCLAP::SwitchArg packOnlyArg(
            "", "pack_only", "Repack the input UV map without solving", cmd);
        cmd.parse(argc, argv);

        flattenMode = flattenArg.getValue() || packOnlyArg.getValue();
        packOnlyMode = packOnlyArg.getValue();
        if (flattenItersArg.getValue() > 0)
            flattenIters = flattenItersArg.getValue();

        if (maxSeamWeightArg.isSet())
            maxSeamWeight = maxSeamWeightArg.getValue();
        if (maxFaceWeightArg.isSet())
            maxFaceWeight = maxFaceWeightArg.getValue();
        if (ignoreUVArg.isSet())
            ignoreUV = ignoreUVArg.getValue();
        meshFileName = inputArg.getValue();
        inputFolderPath = std::filesystem::path(meshFileName).parent_path();
        if (outputArg.isSet())
            outputFolderPath = outputArg.getValue();
        else
            outputFolderPath =
                std::string(inputFolderPath.parent_path().u8string()) +
                pathSeparator() + "output" + pathSeparator();
        if (lambdaInitArg.isSet()) {
            lambda_init = lambdaInitArg.getValue();
            if (lambda_init < 0.0 || lambda_init >= 1.0)
                lambda_init = 0.999;
        }
        if (upperBoundArg.isSet())
            upperBound = upperBoundArg.getValue();
    } catch (TCLAP::ArgException &e) // catch any exceptions
    {
        std::cerr << "error: " << e.error() << " for arg " << e.argId()
                  << std::endl;
        return 1;
    }
    // create output folder
    if (!std::filesystem::exists(outputFolderPath) &&
        !std::filesystem::create_directory(outputFolderPath)) {
        printf("Failed to create output directory %s\n",
               outputFolderPath.c_str());
        return -1;
    }
    if (flattenMode)
        return uvgami::runFlatten(meshFileName, outputFolderPath, flattenIters,
                                  packOnlyMode);

    // Load mesh
    std::string meshFilePath = meshFileName;
    meshFileName =
        meshFileName.substr(meshFileName.find_last_of(pathSeparator()) + 1);
    meshName = meshFileName.substr(0, meshFileName.find_last_of('.'));
    const std::string suffix =
        meshFilePath.substr(meshFilePath.find_last_of('.'));
    bool loadSucceed = false;
    if (suffix == ".off") {
        loadSucceed = igl::readOFF(meshFilePath, V, F);
    } else if (suffix == ".obj") {
        loadSucceed = igl::readOBJ(meshFilePath, V, UV, N, F, FUV, FN);
    } else {
        std::cout << "unkown mesh file format" << std::endl;
        return UVGAMI_RC_UNKNOWN_MESH_FORMAT;
    }
    if (!loadSucceed) {
        std::cerr << "failed to load mesh" << std::endl;
        return UVGAMI_RC_FAILED_TO_LOAD_MESH;
    }
    // nan or absurd coordinates reach the overlap grid and the solver as
    // hangs with no exit code, reject them at load
    if (V.rows() != 0 &&
        (!V.allFinite() || V.cwiseAbs().maxCoeff() > 1e15 ||
         (UV.rows() != 0 &&
          (!UV.allFinite() || UV.cwiseAbs().maxCoeff() > 1e15)))) {
        std::cerr << "input has non-finite or extreme coordinates" << std::endl;
        return UVGAMI_RC_INVALID_COORDS;
    }
    // computeFeatures throws on a zero-area rest triangle, refuse at load
    // with the same area it computes
    for (int triI = 0; triI < F.rows(); triI++) {
        const Eigen::Vector3d e1 = V.row(F(triI, 1)) - V.row(F(triI, 0));
        const Eigen::Vector3d e2 = V.row(F(triI, 2)) - V.row(F(triI, 0));
        if (0.5 * e1.cross(e2).norm() == 0.0) {
            std::cerr << "input has zero-area faces" << std::endl;
            return UVGAMI_RC_ZERO_AREA_FACES;
        }
    }
    //    //DEBUG
    //    uvgami::TriMesh squareMesh(uvgami::P_SQUARE, 1.0, 0.1, false);
    //    V = squareMesh.V_rest;
    //    F = squareMesh.F;

    hasUV = !ignoreUV && (UV.rows() != 0);
    if (!hasUV) {
        // a bowtie vertex joins two face fans at a point. split it into one
        // vertex per fan so the mesh passes the manifold check below, the
        // position is unchanged and the fans were only touching anyway
        {
            std::vector<std::vector<int>> vertTris(V.rows());
            for (int triI = 0; triI < F.rows(); ++triI) {
                for (int i = 0; i < 3; ++i) {
                    vertTris[F(triI, i)].emplace_back(triI);
                }
            }
            int bowtieAmt = 0;
            const int vAmt = static_cast<int>(vertTris.size());
            for (int vI = 0; vI < vAmt; ++vI) {
                const std::vector<int> &tris = vertTris[vI];
                if (tris.size() < 2) {
                    continue;
                }
                // fans are the connected groups of incident triangles,
                // joined when two share an edge through vI
                std::map<int, std::vector<int>> edgeTris;
                for (const auto triI : tris) {
                    for (int i = 0; i < 3; ++i) {
                        if (F(triI, i) != vI) {
                            edgeTris[F(triI, i)].emplace_back(triI);
                        }
                    }
                }
                std::set<int> left(tris.begin(), tris.end());
                bool firstFan = true;
                while (!left.empty()) {
                    std::vector<int> fan({*left.begin()});
                    left.erase(fan[0]);
                    for (size_t fanI = 0; fanI < fan.size(); ++fanI) {
                        for (int i = 0; i < 3; ++i) {
                            const int u = F(fan[fanI], i);
                            if (u == vI) {
                                continue;
                            }
                            for (const auto nbTriI : edgeTris[u]) {
                                if (left.erase(nbTriI)) {
                                    fan.emplace_back(nbTriI);
                                }
                            }
                        }
                    }
                    if (firstFan) {
                        firstFan = false;
                        continue;
                    }
                    const int nV = static_cast<int>(V.rows());
                    V.conservativeResize(nV + 1, 3);
                    V.row(nV) = V.row(vI);
                    for (const auto triI : fan) {
                        for (int i = 0; i < 3; ++i) {
                            if (F(triI, i) == vI) {
                                F(triI, i) = nV;
                            }
                        }
                    }
                    ++bowtieAmt;
                }
            }
            if (bowtieAmt) {
                std::cerr << "split " << bowtieAmt
                          << " non-manifold vertices into per-fan copies"
                          << std::endl;
            }
        }
        vertAmt_input = V.rows();
        Eigen::VectorXi B;
        bool isManifoldVertices = igl::is_vertex_manifold(F, B);
        if (!isManifoldVertices) {
            std::cerr << "input mesh contains non-manifold vertices"
                      << std::endl;
            return UVGAMI_RC_NON_MANIFOLD_VERTICES;
        }
        bool isManifoldEdges = igl::is_edge_manifold(F);
        if (!isManifoldEdges) {
            std::cerr << "input mesh contains non-manifold edges" << std::endl;
            return UVGAMI_RC_NON_MANIFOLD_EDGES;
        }
        // igl's manifold checks ignore winding, and a repeated directed
        // edge corrupts the edge2Tri adjacency built on unique directions
        std::set<std::pair<int, int>> directedEdges;
        for (int triI = 0; triI < F.rows(); ++triI) {
            for (int i = 0; i < 3; ++i) {
                if (!directedEdges.emplace(F(triI, i), F(triI, (i + 1) % 3))
                         .second) {
                    std::cerr << "input mesh has inconsistently oriented faces"
                              << std::endl;
                    return UVGAMI_RC_FLIPPED_FACES;
                }
            }
        }
    }

    // with input UV the components are the UV charts, so the cutting below
    // works on either
    Eigen::VectorXi C;
    igl::facet_components(hasUV ? FUV : F, C);
    int n_components = C.maxCoeff() + 1;

    uvgami::TriMesh temp =
        hasUV ? uvgami::TriMesh(V, F, UV, FUV, false)
              : uvgami::TriMesh(V, F, Eigen::MatrixXd(), Eigen::MatrixXi(),
                                false);

    // stitch mode: a <mesh>_stitch sidecar asks for greedy island merging on
    // the kept map, a redone layout has no island placement worth stitching.
    // detected before the keep decision because stitch runs relax the disk
    // requirement below
    std::string stitchFileName = std::string(inputFolderPath.u8string()) +
                                 pathSeparator() + meshName + "_stitch";
    stitchMode = std::ifstream(stitchFileName).is_open();

    // pinned uvs: <mesh>_fixed lists comma-separated 0-based uv vertex
    // indices to hold in place while the rest reshapes and cuts. only valid
    // when the input map is kept, a redone layout has nothing to pin to,
    // and the check comes before the cut-to-disk fallback so pinned runs
    // fail with this reason instead of a cutting error. read before the keep
    // decision because nocut relaxes the disk requirement like stitch does
    std::set<int> fixedVerts;
    std::string fixedFileName = std::string(inputFolderPath.u8string()) +
                                pathSeparator() + meshName + "_fixed";
    std::ifstream fixedFile(fixedFileName);
    if (fixedFile.is_open()) {
        std::string line;
        getline(fixedFile, line);
        for (float token : split(line, ','))
            fixedVerts.insert((int)token);
        // optional second line "nocut" keeps the map's topology untouched.
        // an empty pin line is valid: a whole-island relax holds nothing
        if (getline(fixedFile, line)) {
            while (!line.empty() &&
                   (line.back() == '\r' || line.back() == '\n'))
                line.pop_back();
            noCutMode = line == "nocut";
        }
        fixedFile.close();
    }

    // an input UV chart is kept when it is already a valid flattening: no
    // flipped or overlapping triangles, and disk topology. the rest are cut to
    // disks below and re-laid out, which keeps the input seams and adds only
    // what the topology needs. the test is per chart, so one bad chart no
    // longer costs every other chart its layout
    std::vector<bool> keepChart(n_components, false);
    int keptCharts = 0;
    bool keepInputUV = false;
    if (hasUV) {
        std::vector<std::vector<int>> chartTris(n_components);
        for (int triI = 0; triI < temp.F.rows(); ++triI) {
            chartTris[C[triI]].emplace_back(triI);
        }

        std::vector<bool> isDisk = chartDiskFlags(temp.F, n_components, C);

        std::vector<std::vector<int>> bnd_all;
        igl::boundary_loop(temp.F, bnd_all);
        std::set<int> crossingVerts;
        uvgami::IglUtils::checkUVBoundaryOverlap(temp.V, bnd_all,
                                                 &crossingVerts);
        // a crossing condemns both charts it touches, so two islands laid on
        // top of each other are both re-cut
        std::vector<bool> overlaps(n_components, false);
        for (int triI = 0; triI < temp.F.rows(); ++triI) {
            for (int i = 0; i < 3; ++i) {
                if (crossingVerts.count(temp.F(triI, i))) {
                    overlaps[C[triI]] = true;
                }
            }
        }

        // a pinched vertex belongs to two charts at once (the fans meet at a
        // point, so nothing joins them into one component). pinning it for one
        // chart while re-cutting the other would pull it two ways, so when the
        // map is not kept whole, neither of those charts is kept
        std::vector<int> vertChart(temp.V.rows(), -1);
        std::vector<bool> pinched(n_components, false);
        for (int triI = 0; triI < temp.F.rows(); ++triI) {
            for (int i = 0; i < 3; ++i) {
                int &owner = vertChart[temp.F(triI, i)];
                if (owner == -1) {
                    owner = C[triI];
                } else if (owner != C[triI]) {
                    pinched[owner] = true;
                    pinched[C[triI]] = true;
                }
            }
        }

        std::vector<bool> inverted(n_components);
        for (int c = 0; c < n_components; ++c) {
            inverted[c] = !temp.checkInversion(true, chartTris[c]);
        }
        bool anyInversion = false, allDisks = true;
        for (int c = 0; c < n_components; ++c) {
            anyInversion = anyInversion || inverted[c];
            allDisks = allDisks && isDisk[c];
        }
        // a stitch run can keep charts with holes: an interior split the
        // engine never merged back leaves a slit, and the machinery is
        // hole-safe. a pinched boundary is not, the scaffold's corner air
        // loop cannot represent it, so those still re-cut. a nocut run never
        // queries topology either, so it rides the same exception
        bool stitchKeepable = stitchMode || noCutMode;
        if (stitchKeepable && !allDisks) {
            std::map<std::pair<int, int>, int> edgeCount;
            for (int triI = 0; triI < temp.F.rows(); ++triI) {
                for (int i = 0; i < 3; ++i) {
                    int a = temp.F(triI, i), b = temp.F(triI, (i + 1) % 3);
                    edgeCount[{std::min(a, b), std::max(a, b)}]++;
                }
            }
            std::vector<int> bndDeg(temp.V.rows(), 0);
            for (const auto &ec : edgeCount) {
                if (ec.second == 1) {
                    bndDeg[ec.first.first]++;
                    bndDeg[ec.first.second]++;
                }
            }
            std::vector<bool> pinchFree(n_components, true);
            for (int triI = 0; triI < temp.F.rows(); ++triI) {
                for (int i = 0; i < 3; ++i) {
                    if (bndDeg[temp.F(triI, i)] > 2)
                        pinchFree[C[triI]] = false;
                }
            }
            for (int c = 0; c < n_components; ++c)
                stitchKeepable = stitchKeepable && pinchFree[c];
        }

        // whole-map decision first, so a map that was kept before is still
        // kept byte for byte
        keepInputUV = (allDisks || stitchKeepable) && !anyInversion &&
                      crossingVerts.empty();

        int badInverted = 0, badOverlapping = 0, badNonDisk = 0;
        if (keepInputUV) {
            keepChart.assign(n_components, true);
            keptCharts = n_components;
        } else {
            for (int c = 0; c < n_components; ++c) {
                keepChart[c] =
                    isDisk[c] && !overlaps[c] && !inverted[c] && !pinched[c];
                keptCharts += keepChart[c];
                if (inverted[c]) {
                    ++badInverted;
                } else if (overlaps[c]) {
                    ++badOverlapping;
                } else if (!isDisk[c]) {
                    ++badNonDisk;
                }
            }
        }

        if (!keepInputUV && keptCharts == 0) {
            std::cout << (anyInversion             ? "local injectivity violated"
                          : !crossingVerts.empty() ? "self-intersecting UV islands"
                                                   : "charts are not disk-topology")
                      << " in input UV map, cutting to disk-topology and "
                         "applying Tutte's embedding..."
                      << std::endl;
        } else if (!keepInputUV) {
            std::cout << "kept " << keptCharts << " of " << n_components
                      << " input UV charts, re-cutting " << badInverted
                      << " inverted, " << badOverlapping
                      << " self-intersecting, " << badNonDisk
                      << " not disk-topology" << std::endl;
        }
    }

    if (!fixedVerts.empty() && !keepInputUV) {
        std::cerr << "pinned vertices need the input UV map kept" << std::endl;
        return UVGAMI_RC_PINNED_UV_NOT_KEPT;
    }

    if (stitchMode && !keepInputUV) {
        std::cerr << "stitching needs the input UV map kept" << std::endl;
        return UVGAMI_RC_STITCH_UV_NOT_KEPT;
    }
    if (noCutMode && !keepInputUV) {
        std::cerr << "nocut needs the input UV map kept" << std::endl;
        return UVGAMI_RC_NOCUT_UV_NOT_KEPT;
    }
    if (!fixedVerts.empty()) {
        // the distortion energy is scale-sensitive and pins block the global
        // rescale the solver would otherwise start with, it would inflate the
        // interior against the held border instead. match the uv scale to the
        // rest shape up front, the output is normalized and realigned through
        // the pins so this scale never leaks out
        double uvArea = 0.0, restArea = 0.0;
        for (int triI = 0; triI < temp.F.rows(); ++triI) {
            const Eigen::RowVector3i &tri = temp.F.row(triI);
            const Eigen::RowVector2d e1 = temp.V.row(tri[1]) - temp.V.row(tri[0]);
            const Eigen::RowVector2d e2 = temp.V.row(tri[2]) - temp.V.row(tri[0]);
            uvArea += std::abs(e1[0] * e2[1] - e1[1] * e2[0]) / 2;
            const Eigen::RowVector3d p0 = temp.V_rest.row(tri[0]);
            const Eigen::RowVector3d p1 = temp.V_rest.row(tri[1]);
            const Eigen::RowVector3d p2 = temp.V_rest.row(tri[2]);
            restArea += (p1 - p0).cross(p2 - p0).norm() / 2;
        }
        if (uvArea > 0.0 && restArea > 0.0)
            temp.V *= std::sqrt(restArea / uvArea);
    }

    uvgami::TriMesh *keptInputMesh = nullptr;
    if (keepInputUV) {
        keptInputMesh = new uvgami::TriMesh(temp);
        triSoup.emplace_back(keptInputMesh);
    } else {
        // in each pass, make one cut on each component if needed, until all
        // becoming disk-topology
        std::vector<Eigen::MatrixXi> F_component(n_components);
        std::vector<std::set<int>> V_ind_component(n_components);
        for (int triI = 0; triI < temp.F.rows(); ++triI) {
            F_component[C[triI]].conservativeResize(
                F_component[C[triI]].rows() + 1, 3);
            F_component[C[triI]].bottomRows(1) = temp.F.row(triI);
            for (int i = 0; i < 3; ++i) {
                V_ind_component[C[triI]].insert(temp.F(triI, i));
            }
        }
        while (true) {
            std::vector<int> components_to_cut;
            for (int componentI = 0; componentI < n_components; ++componentI) {
                int EC = igl::euler_characteristic(temp.V,
                                                   F_component[componentI]) -
                         temp.V.rows() + V_ind_component[componentI].size();
                if (EC < 1) {
                    // treat as higher-genus surfaces using cut_to_disk()
                    components_to_cut.emplace_back(-componentI - 1);
                } else if (EC == 2) {
                    // closed genus-0 surface
                    components_to_cut.emplace_back(componentI);
                } else if (EC != 1) {
                    std::cerr << "unsupported single-connected component"
                              << std::endl;
                    return UVGAMI_RC_UNSUPPORTED_TOPOLOGY;
                }
            }

            if (components_to_cut.empty()) {
                break;
            }

            try {
                for (auto componentI : components_to_cut) {
                    if (componentI < 0) {
                        // cut high genus
                        componentI = -componentI - 1;

                        // meshes with boundary are supported; boundary edges
                        // will be included as cuts
                        std::vector<std::vector<int>> cuts;
                        igl::cut_to_disk(F_component[componentI], cuts);

                        // only cut one seam each time to avoid seam vertex id
                        // inconsistency
                        int cuts_made = 0;
                        for (auto &seamI : cuts) {
                            if (seamI.front() == seamI.back()) {
                                // cutPath() does not support closed-loop cuts,
                                // split it into two cuts
                                cuts_made += temp.cutPath(
                                    std::vector<int>({seamI[seamI.size() - 3],
                                                      seamI[seamI.size() - 2],
                                                      seamI[seamI.size() - 1]}),
                                    true);
                                temp.initSeams = temp.cohE;
                                seamI.resize(seamI.size() - 2);
                            }
                            cuts_made += temp.cutPath(seamI, true);
                            temp.initSeams = temp.cohE;
                            if (cuts_made) {
                                break;
                            }
                        }

                        if (!cuts_made) {
                            std::cerr << "no cuts made when cutting input "
                                         "geometry to disk-topology"
                                      << std::endl;
                            return UVGAMI_RC_CUT_FAILED;
                        }
                    } else {
                        // cut the topological sphere into a topological disk;
                        // seed at the component's smallest vertex index so a
                        // single-component mesh cuts at vertex 0
                        int seedVI = *V_ind_component[componentI].begin();
                        switch (initCutOption) {
                        case 0:
                            temp.onePointCut(seedVI);
                            rand1PInitCut = (n_components == 1);
                            break;
                        case 1:
                            temp.farthestPointCut(seedVI);
                            break;
                        default:
                            assert(0);
                            break;
                        }
                    }
                }
            } catch (const std::exception &e) {
                std::cerr << "initial cut failed: " << e.what()
                          << std::endl;
                return UVGAMI_RC_CUT_FAILED;
            }

            // data update on each component for identifying a new cut
            F_component.resize(0);
            F_component.resize(n_components);
            V_ind_component.resize(0);
            V_ind_component.resize(n_components);
            for (int triI = 0; triI < temp.F.rows(); ++triI) {
                F_component[C[triI]].conservativeResize(
                    F_component[C[triI]].rows() + 1, 3);
                F_component[C[triI]].bottomRows(1) = temp.F.row(triI);
                for (int i = 0; i < 3; ++i) {
                    V_ind_component[C[triI]].insert(temp.F(triI, i));
                }
            }
        }

        // cutting through a boundary vertex can orphan the original row,
        // every face reassigned to its duplicate. an orphan poisons what
        // follows: its zero laplacian row makes the tutte solve singular,
        // and the scaffold seeds a triangle hole at its position, so drop
        // orphan rows and remap before flattening
        {
            std::vector<int> vMap(temp.V.rows(), -1);
            for (int triI = 0; triI < temp.F.rows(); ++triI) {
                for (int i = 0; i < 3; ++i) {
                    vMap[temp.F(triI, i)] = 0;
                }
            }
            int vNew = 0;
            for (int vI = 0; vI < temp.V.rows(); ++vI) {
                if (vMap[vI] == 0) {
                    vMap[vI] = vNew++;
                }
            }
            if (vNew < temp.V.rows()) {
                for (int vI = 0; vI < temp.V.rows(); ++vI) {
                    if (vMap[vI] >= 0 && vMap[vI] != vI) {
                        temp.V_rest.row(vMap[vI]) = temp.V_rest.row(vI);
                        temp.V.row(vMap[vI]) = temp.V.row(vI);
                        temp.vertWeight[vMap[vI]] = temp.vertWeight[vI];
                    }
                }
                temp.V_rest.conservativeResize(vNew, 3);
                temp.V.conservativeResize(vNew, 2);
                temp.vertWeight.conservativeResize(vNew);
                for (int triI = 0; triI < temp.F.rows(); ++triI) {
                    for (int i = 0; i < 3; ++i) {
                        temp.F(triI, i) = vMap[temp.F(triI, i)];
                    }
                }
                F_component.assign(n_components, Eigen::MatrixXi());
                V_ind_component.assign(n_components, std::set<int>());
                for (int triI = 0; triI < temp.F.rows(); ++triI) {
                    F_component[C[triI]].conservativeResize(
                        F_component[C[triI]].rows() + 1, 3);
                    F_component[C[triI]].bottomRows(1) = temp.F.row(triI);
                    for (int i = 0; i < 3; ++i) {
                        V_ind_component[C[triI]].insert(temp.F(triI, i));
                    }
                }
            }
        }

        int UVGridDim = 0;
        do {
            ++UVGridDim;
        } while (UVGridDim * UVGridDim < n_components);

        // a re-cut chart starts as a unit circle, which beside charts kept
        // from a packed input map is far bigger than its own rest shape, and
        // the optimizer would spend its iterations shrinking it. match the
        // kept charts' uv-to-3D scale instead, and keep the unit circle when
        // nothing is kept so that layout is untouched
        std::vector<double> chartRadius(n_components, 1.0);
        double gridCell = 2.1, gridOriginX = 0.0;
        if (keptCharts) {
            double keptUV = 0.0, kept3D = 0.0;
            std::vector<double> area3D(n_components, 0.0);
            for (int triI = 0; triI < temp.F.rows(); ++triI) {
                const Eigen::RowVector3i &tri = temp.F.row(triI);
                const Eigen::RowVector3d p0 = temp.V_rest.row(tri[0]);
                const Eigen::RowVector3d p1 = temp.V_rest.row(tri[1]);
                const Eigen::RowVector3d p2 = temp.V_rest.row(tri[2]);
                double a3 = (p1 - p0).cross(p2 - p0).norm() / 2;
                area3D[C[triI]] += a3;
                if (keepChart[C[triI]]) {
                    const Eigen::RowVector2d e1 =
                        temp.V.row(tri[1]) - temp.V.row(tri[0]);
                    const Eigen::RowVector2d e2 =
                        temp.V.row(tri[2]) - temp.V.row(tri[0]);
                    keptUV += std::abs(e1[0] * e2[1] - e1[1] * e2[0]) / 2;
                    kept3D += a3;
                }
            }
            double uvPerRest =
                (keptUV > 0.0 && kept3D > 0.0) ? std::sqrt(keptUV / kept3D) : 1.0;
            gridCell = 0.0;
            for (int c = 0; c < n_components; ++c) {
                chartRadius[c] = uvPerRest * std::sqrt(area3D[c] / M_PI);
                if (!keepChart[c]) {
                    gridCell = (std::max)(gridCell, 2.1 * chartRadius[c]);
                }
            }
            // and place them past the kept charts, since the output layout is
            // only normalized, never packed, so a circle landing on a kept
            // chart would stay on top of it
            for (int componentI = 0; componentI < n_components; ++componentI) {
                if (!keepChart[componentI]) {
                    continue;
                }
                for (const auto &vI : V_ind_component[componentI]) {
                    gridOriginX =
                        (std::max)(gridOriginX, temp.V(vI, 0) + gridCell / 2);
                }
            }
        }

        // compute boundary UV coordinates, using a grid layout for multiComp
        Eigen::VectorXi bnd_stacked;
        Eigen::MatrixXd bnd_uv_stacked;
        for (int componentI = 0; componentI < n_components; ++componentI) {
            if (keepChart[componentI]) {
                // pin every vertex of a kept chart, so the harmonic solve
                // reproduces its input UV exactly and only fills in the rest
                const std::set<int> &chartV = V_ind_component[componentI];
                int base = bnd_stacked.size();
                bnd_stacked.conservativeResize(base + chartV.size());
                bnd_uv_stacked.conservativeResize(base + chartV.size(), 2);
                for (const auto &vI : chartV) {
                    bnd_stacked[base] = vI;
                    bnd_uv_stacked.row(base) = temp.V.row(vI);
                    ++base;
                }
                continue;
            }
            std::vector<std::vector<int>> bnd_all;
            igl::boundary_loop(F_component[componentI], bnd_all);

            int longest_bnd_id = 0;
            for (int bnd_id = 1; bnd_id < bnd_all.size(); ++bnd_id) {
                if (bnd_all[longest_bnd_id].size() < bnd_all[bnd_id].size()) {
                    longest_bnd_id = bnd_id;
                }
            }

            bnd_stacked.conservativeResize(bnd_stacked.size() +
                                           bnd_all[longest_bnd_id].size());
            bnd_stacked.tail(bnd_all[longest_bnd_id].size()) =
                Eigen::VectorXi::Map(bnd_all[longest_bnd_id].data(),
                                     bnd_all[longest_bnd_id].size());

            Eigen::MatrixXd bnd_uv;
            if (n_components == 1) {
                // multiComp keeps unit circles so the 2.1 grid offsets hold
                uvgami::IglUtils::map_vertices_to_circle(
                    temp.V_rest,
                    bnd_stacked.tail(bnd_all[longest_bnd_id].size()), bnd_uv);
            } else {
                igl::map_vertices_to_circle(
                    temp.V_rest,
                    bnd_stacked.tail(bnd_all[longest_bnd_id].size()), bnd_uv);
            }
            double xOffset = gridOriginX + componentI % UVGridDim * gridCell,
                   yOffset = componentI / UVGridDim * gridCell;
            for (int bnd_uvI = 0; bnd_uvI < bnd_uv.rows(); bnd_uvI++) {
                bnd_uv(bnd_uvI, 0) =
                    bnd_uv(bnd_uvI, 0) * chartRadius[componentI] + xOffset;
                bnd_uv(bnd_uvI, 1) =
                    bnd_uv(bnd_uvI, 1) * chartRadius[componentI] + yOffset;
            }
            bnd_uv_stacked.conservativeResize(
                bnd_uv_stacked.rows() + bnd_uv.rows(), 2);
            bnd_uv_stacked.bottomRows(bnd_uv.rows()) = bnd_uv;
        }

        // Harmonic map with uniform weights
        Eigen::MatrixXd UV_Tutte;
        Eigen::SparseMatrix<double> A, M;
        uvgami::IglUtils::computeUniformLaplacian(temp.F, A);
        igl::harmonic(A, M, bnd_stacked, bnd_uv_stacked, 1, UV_Tutte);

        triSoup.emplace_back(
            new uvgami::TriMesh(V, F, UV_Tutte, temp.F, false));

        // a genus-0 one-point cut can invert under rounding on large or curvy
        // meshes; retry from successive split vertices until the map is valid.
        // upstream dropped this in 2218d87; kept here so such meshes still pass
        if (rand1PInitCut) {
            int splitVI = 0;
            while (!triSoup.back()->checkInversion(true) &&
                   splitVI + 1 < V.rows()) {
                std::cerr << "element inversion during UV init, trying another "
                             "vertex"
                          << std::endl;
                uvgami::TriMesh cutMesh(V, F, Eigen::MatrixXd(),
                                        Eigen::MatrixXi(), false);
                cutMesh.onePointCut(++splitVI);
                Eigen::VectorXi bnd;
                igl::boundary_loop(cutMesh.F, bnd);
                assert(bnd.size());
                Eigen::MatrixXd bnd_uv;
                uvgami::IglUtils::map_vertices_to_circle(cutMesh.V_rest, bnd,
                                                         bnd_uv);
                Eigen::SparseMatrix<double> A_retry, M_retry;
                uvgami::IglUtils::computeUniformLaplacian(cutMesh.F, A_retry);
                Eigen::MatrixXd UV_retry;
                igl::harmonic(A_retry, M_retry, bnd, bnd_uv, 1, UV_retry);
                delete triSoup.back();
                triSoup.back() =
                    new uvgami::TriMesh(V, F, UV_retry, cutMesh.F, false);
            }
        }
    }
    if (!fixedVerts.empty()) {
        if (*fixedVerts.begin() < 0 ||
            *fixedVerts.rbegin() >= keptInputMesh->V.rows()) {
            std::cerr << "pinned vertex index out of range" << std::endl;
            return UVGAMI_RC_INVALID_UV;
        }
        keptInputMesh->resetFixedVert(fixedVerts);
        pinnedMode = true;
    }

    // per-face importance: sidecar vertex weights averaged onto faces scale
    // the distortion energy. loaded before the optimizer copies the mesh so
    // the initial energy is already weighted
    if (maxFaceWeight > 1) {
        const Eigen::MatrixXi &F_in = hasUV ? FUV : F;
        const int nVW = hasUV ? (int)UV.rows() : (int)V.rows();
        Eigen::VectorXd vW = Eigen::VectorXd::Zero(nVW);
        if (loadWeightSidecar(std::string(inputFolderPath.u8string()) +
                                  pathSeparator() + meshName + "_importance",
                              vW)) {
            // the initial mesh is heap-allocated above, triSoup only stores
            // it as const
            uvgami::TriMesh &mesh0 = *const_cast<uvgami::TriMesh *>(triSoup[0]);
            if (F_in.rows() == mesh0.F.rows()) {
                for (int triI = 0; triI < F_in.rows(); triI++) {
                    const double avg = (vW[F_in(triI, 0)] + vW[F_in(triI, 1)] +
                                        vW[F_in(triI, 2)]) /
                                       3.0;
                    mesh0.faceWeight[triI] = 1.0 + avg * (maxFaceWeight - 1);
                }
                // area-weighted mean 1 keeps the -u bound comparable
                mesh0.faceWeight /=
                    mesh0.faceWeight.dot(mesh0.triArea) / mesh0.surfaceArea;
            } else {
                std::cerr << "importance weights skipped, face count mismatch"
                          << std::endl;
            }
        }
    }

    outputFolderPath += meshName;
    energyParams.emplace_back(1.0 - lambda_init);
    energyTerms.emplace_back(new uvgami::SymDirichletEnergy());

    try {
        // for random one point initial cut, don't need air meshes in the
        // beginning since it's impossible for a quad to intersect itself
        optimizer = new uvgami::Optimizer(
            *triSoup[0], energyTerms, energyParams, 0, true, !rand1PInitCut);
    } catch (UvgamiElementInversionException &eie) {
        (void)eie;
        return UVGAMI_RC_ELEMENT_INVERSION;
    }
    optimizer->precompute();
    triSoup.emplace_back(&optimizer->getResult());
    triSoup_backup = optimizer->getResult();
    triSoup.emplace_back(
        &optimizer->getData_findExtrema()); // for visualizing UV map for
                                            // finding extrema

    // regional seam placement
    uvgami::TriMesh &result = optimizer->getResult();
    Eigen::VectorXd sW = Eigen::VectorXd::Zero(result.vertWeight.size());
    if (loadWeightSidecar(std::string(inputFolderPath.u8string()) +
                              pathSeparator() + meshName + "_weights",
                          sW)) {
        result.vertWeight =
            (1.0 + sW.array() * (maxSeamWeight - 1)).matrix();
        uvgami::IglUtils::smoothVertField(result, result.vertWeight);
    }

    // an imported map that already meets the bound ships untouched. the
    // search treats the bound as an allowance and spends the slack on
    // shorter seams, drifting seams a preseed placed deliberately. modes
    // that exist to move the map (pins, stitch, nocut) still run
    if (keepInputUV && !pinnedMode && !stitchMode && !noCutMode &&
        importedMapMeasure(optimizer->getResult()) <= upperBound) {
        canSaveMesh = true;
        converge_preDrawFunc();
    }

    std::thread t(&stdin_listener);
    while (true) {
        preDrawFunc();
        if (postDrawFunc())
            break;
    }
    // cleanup
    t.detach();
    for (auto &eI : energyTerms)
        delete eI;
    delete optimizer;
    delete triSoup[0];

    return 0;
}
