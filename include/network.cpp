// Network-cpp - A c++ based tool to simulate electric power networks.
// https://github.com/julianfritzsch/network-cpp
// Copyright (C) 2022  Julian Fritzsch
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#include "../include/network.hpp"

#include <matplot/matplot.h>

#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace net {
/**
 * Constructor without initial angles.
 * @param adjlist Name of the file that contains the adjacency list. The
 * adjacency list needs to be in the format node0, node1, weight
 * @param coeffs Name of the file that contains the coefficients. The file needs
 * to be in the format node, inertia, damping, power production / consumption
 * */
Network::Network(std::string adjlist, std::string coeffs) {
  createAdjlist(adjlist);
  createCoeffLists(coeffs);
  setInitialAngles();
}

/**
 * Constructor with initial angles.
 * @param adjlist Name of the file that contains the adjacency list. The
 * adjacency list needs to be in the format node0, node1, weight
 * @param coeffs Name of the file that contains the coefficients. The file needs
 * to be in the format node, inertia, damping, power production / consumption
 * @param angles Name of the file containing the initial angles. The initial
 * angle for each node should be on a single row.
 * */
Network::Network(std::string adjlist, std::string coeffs, std::string angles) {
  createAdjlist(adjlist);
  createCoeffLists(coeffs);
  setInitialAngles(angles);
}

/**
 * Read in the adjacency list.
 * The file needs to be given in the format node1, node2, weight.
 * The list is stored into Network::adjacencyList.
 * @param adjlist Name of the file containing the adjacency list
 */
void Network::createAdjlist(std::string adjlist) {
  assert(std::filesystem::exists(adjlist));
  // Create necessary temporary variables and read file into vectors
  std::ifstream adjinput(adjlist);
  std::string line;
  std::size_t startnode, endnode;
  double lineweight;
  int colindex;
  double tempin;
  while (getline(adjinput, line)) {
    std::stringstream ss(line);
    colindex = 0;
    while (ss >> tempin) {
      switch (colindex) {
        case 0:
          startnode = static_cast<int>(round(tempin));
          break;
        case 1:
          endnode = static_cast<int>(round(tempin));
          break;
        case 2:
          lineweight = tempin;
          break;
        default:
          break;
      }
      if (ss.peek() == ',') {
        ss.ignore();
      }
      ++colindex;
    }
    while (adjacencyList.size() <= startnode ||
           adjacencyList.size() <= endnode) {
      adjacencyList.push_back(std::vector<std::pair<int, double>>());
    }
    adjacencyList[startnode].push_back({endnode, lineweight});
    adjacencyList[endnode].push_back({startnode, lineweight});
  }
  adjinput.close();
  nNodes = adjacencyList.size();
}

/**
 * Read in the coefficients list.
 * The file needs to be given in the format node, inertia, damping, power.
 * At the moment the inertia-full nodes need to be consecutive at the beginning.
 * @param coeffs Name of the file containing the coefficients
 * @todo Allow for arbitrarily ordered inertia-full and inertia-less nodes.
 */
void Network::createCoeffLists(std::string coeffs) {
  assert(std::filesystem::exists(coeffs));
  // Allocate space for inertia, damping, and power
  inertia = arma::vec(nNodes);
  damping = arma::vec(nNodes);
  power = arma::vec(nNodes);
  // Read file into respective vectors
  std::ifstream coeffin(coeffs);
  std::string line;
  int colindex;
  double tempin;
  std::size_t gen;
  while (getline(coeffin, line)) {
    colindex = 0;
    std::stringstream ss(line);
    while (ss >> tempin) {
      switch (colindex) {
        case 0:
          gen = static_cast<arma::size_t>(round(tempin));
          break;
        case 1:
          inertia(gen) = tempin;
          break;
        case 2:
          damping(gen) = tempin;
          break;
        case 3:
          power(gen) = tempin;
          break;
        default:
          break;
      }
      if (ss.peek() == ',') {
        ss.ignore();
      }
      ++colindex;
    }
  }
  coeffin.close();
  // Get number of nodes with inertia
  arma::vec tmp(nNodes, arma::fill::value(1.0e-6));
  nInertia = arma::sum(inertia > tmp);
}

/** Set initial angles to zero. */
void Network::setInitialAngles() {
  initAngles = arma::vec(nNodes, arma::fill::zeros);
}

/**
 * Set initial angles to given values.
 * @param angles Name of the file containing the inital angles
 */
void Network::setInitialAngles(std::string angles) {
  assert(std::filesystem::exists(angles));
  initAngles = arma::vec(nNodes);
  initAngles.load(angles, arma::csv_ascii);
}

/**
 * Create a step perturbation.
 * Changes the power of the given node by the given power change:
 * `power(node) += powerChange`
 * @param node Node whose power is changed
 * @param powerChange amount of change in power
 */
void Network::step(std::size_t node, double powerChange) {
  power(node) += powerChange;
}

/**
 * Create a box perturbation.
 * Changes the power of the given node by the given power change:
 * `power(node) += powerChange`. In a dynamical simulation the power will be
 * reset after `time` has passed.
 * @param node Node whose power is changed
 * @param powerChange amount of change in power
 * @param time Duration of perturbation
 */
void Network::box(std::size_t node, double powerChange, double time) {
  boxPer = true;
  boxIx = node;
  boxPower = powerChange;
  power(node) += powerChange;
  boxTime = time;
}

/**
 * Run a dynamical simulation.
 * Do a dynamical simulation of the swing equations using a semi-implicit
 * mid-point method.
 * This method is extremely fast as it only requires a single inversion of the
 * Jacobian but might lead to a small zick-zack behavior. This functions sets
 * the values of Network::t and Network::yData. A description of the algorithm
 * is given in G. Bader and P. Deuflhard, Numer. Math. 41, 373 (1983).
 * @param t0 Start time
 * @param tf Finish time
 * @param dt Time step
 * @param se Save only every `se`th step. Recommended to leave at default unless
 * memory problems are encountered.
 * @todo This functions should be refactored to be an interface that allows
 * different methods to be chosen.
 */
void Network::dynamicalSimulation(double t0, double tf, double dt, int se) {
  double tt{t0};
  // Calculate number of steps and number of points saved
  arma::size_t nSteps{static_cast<arma::size_t>(std::round((tf - t0) / dt)) +
                      1};
  arma::size_t saveN{nSteps % se == 0 ? nSteps / se : nSteps / se + 1};

  // Setup initial conditions
  arma::vec y{arma::join_cols(initAngles, arma::vec(nInertia))};
  t = arma::vec(saveN);
  yData = arma::mat(nNodes + nInertia, saveN);
  yData.col(0) = y;
  arma::vec deltaCurrent(nNodes + nInertia);
  t(0) = tt;

  // Do first step
  arma::mat toinv =
      arma::eye(nNodes + nInertia, nNodes + nInertia) - dt * arma::mat(df(y));
  arma::mat inv = arma::inv(toinv);
  deltaCurrent = dt * inv * f(y);
  y += deltaCurrent;
  tt += dt;

  // Main loop
  for (int i = 1; i < nSteps - 1; ++i) {
    // Create box perturbation
    if (boxPer) {
      if (tt >= boxTime) {
        power(boxIx) -= boxPower;
        boxPer = false;
      }
    }
    if (i % se == 0) {
      // Save data and print some useful information
      std::cout << std::fixed << std::setprecision(1)
                << (tt - t0) / (tf - t0) * 100
                << "%\nMax omega: " << std::setprecision(4)
                << arma::max(
                       arma::abs(y(arma::span(nNodes, nNodes + nInertia - 1))))
                << "\n\x1b[A\u001b[2K\x1b[A\u001b[2K";
      yData.col(i / se) = y;
      t(i / se) = tt;
    }
    deltaCurrent += 2 * inv * (dt * f(y) - deltaCurrent);
    y += deltaCurrent;
    tt += dt;
  }

  // Do special smoothing step only if the last data point is saved
  if ((nSteps - 1) % se == 0) {
    deltaCurrent = inv * (dt * f(y) - deltaCurrent);
    y += deltaCurrent;
    t(saveN - 1) = tt;
    yData.col(saveN - 1) = y;
  }

  // Insert the load frequencies
  yData.insert_rows(yData.n_rows, calculateLoadFrequencies());
  std::cout << "Final max omega: "
            << arma::max(
                   arma::abs(y(arma::span(nNodes, nNodes + nInertia - 1))))
            << "\n";
}

/**
 * Run a dynamical simulation.
 * Run a dynamical simulation using the fourth order Kaps-Rentrop method with
 * adaptive stepsize. This method is more precise and stable but considerably
 * slower as it require 4 linear systems to be solved at every step. For a
 * description of the algorithm and the different constants see J. R. Winkler,
 * Endeavour 17, 201 (1993). This functions sets the values of Network::t and
 * Network::yData.
 * @param t0 Start time
 * @param tf Finish time
 * @param dtStart First time step
 * @param dtMax Maximum allowed time step
 * @param eps Maximum allowed error at each step. \f$\text{error} < \varepsilon
 * y + \varepsilon\f$
 * @param maxTries Maximum allowed tries for one step. The method will abort if
 * maxTries is reached.
 * @todo Use a different solver to directly obtain LU-decomposition to only do
 * it once per step, that might require rewriting this library using Eigen
 * @todo Change data storage to avoid constant reallocations
 */
void Network::kapsRentrop(double t0, double tf, double dtStart, double dtMax,
                          double eps, int maxTries) {
  // Setup constants
  const double gam{1.0 / 2.0};
  const double a21{2.0};
  const double a31{48.0 / 25.0};
  const double a32{6.0 / 25.0};
  const double c21{-8.0};
  const double c31{372.0 / 25.0};
  const double c32{12.0 / 5.0};
  const double c41{-112.0 / 125.0};
  const double c42{-54.0 / 125.0};
  const double c43{-2.0 / 5.0};
  const double b1{19.0 / 9.0};
  const double b2{1.0 / 2.0};
  const double b3{25.0 / 108.0};
  const double b4{125.0 / 108.0};
  const double e1{17.0 / 54.0};
  const double e2{7.0 / 36.0};
  const double e3{0.0};
  const double e4{125.0 / 108.0};
  const double c1x{1.0 / 2.0};
  const double c2x{-3.0 / 2.0};
  const double c3x{121.0 / 50.0};
  const double c4x{29.0 / 250.0};

  // Setup initial values
  double tt{t0};
  double dt{dtStart};
  arma::vec y{arma::join_cols(initAngles, arma::vec(nInertia))};
  yData.insert_cols(0, y);
  t.insert_rows(0, arma::rowvec{tt});

  // Create needed arrays
  arma::sp_mat jacobian, leftSide;
  arma::vec k1, k2, k3, k4, yscale;
  int tries{0};

  // Main loop
  while (tt < tf) {
    // Create box perturbation
    if (boxPer) {
      if (tt >= boxTime) {
        power(boxIx) -= boxPower;
        boxPer = false;
      }
    }

    // Do one step
    jacobian = df(y);
    leftSide =
        arma::speye(nNodes + nInertia, nNodes + nInertia) / gam / dt - jacobian;
    yscale = eps * arma::abs(y) + eps;
    k1 = arma::spsolve(leftSide, f(y));
    k2 = arma::spsolve(leftSide, f(y + a21 * k1) + c21 * k1 / dt);
    arma::vec f34 = f(y + a31 * k1 + a32 * k2);
    k3 = arma::spsolve(leftSide, f34 + c31 * k1 / dt + c32 * k2 / dt);
    k4 = arma::spsolve(leftSide,
                       f34 + c41 * k1 / dt + c42 * k2 / dt + c43 * k3 / dt);

    // Error calculation. If error < 1 save data and increase step size, else
    // decrease and redo step
    arma::vec err = e1 * k1 + e2 * k2 + e3 * k3 + e4 * k4;
    double error = arma::max(arma::abs(err / yscale));
    if (error < 1) {
      y += b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4;
      tt += dt;
      dt = std::min(0.95 * std::pow(1.0 / error, 1.0 / 4.0) * dt,
                    std::min(1.5 * dt, dtMax));
      tries = 0;
      yData.insert_cols(yData.n_cols, y);
      t.insert_rows(t.n_rows, arma::rowvec{tt});
      // Print some useful information
      std::cout << std::fixed << std::setprecision(1)
                << (tt - t0) / (tf - t0) * 100
                << "%\nMax omega: " << std::setprecision(4)
                << arma::max(
                       arma::abs(y(arma::span(nNodes, nNodes + nInertia - 1))))
                << "\n\x1b[A\u001b[2K\x1b[A\u001b[2K";
    } else {
      dt = std::max(0.95 * std::pow(1.0 / error, 1.0 / 3.0) * dt, 0.5 * dt);
      if (dt < 1.0e-5) {
        std::cout << "Step size effectively zero.\n";
        break;
      }
      tries++;
      if (tries == maxTries) {
        std::cout << "Max tries reached!\n";
        break;
      }
    }
  }

  // Insert load frequency data
  yData.insert_rows(yData.n_rows, calculateLoadFrequencies());
  std::cout << "Final max omega: "
            << arma::max(
                   arma::abs(y(arma::span(nNodes, nNodes + nInertia - 1))))
            << "\n";
}

/** Return function value.
 * Returns the value of the swing equations for the given angles and
 * frequencies. For a definition of the swing equations see, e.g., A. R. Bergen
 * and V. Vittal, Power Systems Analysis, 2nd ed (Prentice Hall, Upper Saddle
 * River, NJ, 2000).
 * @param y Vector containing angles and frequencies
 */
arma::vec Network::f(arma::vec y) {
  arma::vec re(nNodes + nInertia);
  for (std::size_t i = 0; i < nInertia; ++i) {
    re(i) = y(nNodes + i);
    re(nNodes + i) = power(i) - damping(i) * y(nNodes + i);
    for (auto &j : adjacencyList[i]) {
      re(nNodes + i) -= j.second * std::sin(y(i) - y(j.first));
    }
    re(nNodes + i) /= inertia(i);
  }
  for (std::size_t i = nInertia; i < nNodes; ++i) {
    re(i) = power(i);
    for (auto &j : adjacencyList[i]) {
      re(i) -= j.second * std::sin(y(i) - y(j.first));
    }
    re(i) /= damping(i);
  }
  return re;
}

/**
 * Calculate the Jacobian for specific values of theta
 *
 * It is given by the following expression
 *
 * \f$\begin{bmatrix}
 * 0_{N_{in} \times N_{in}} & 0_{N_{in} \times N_{in}} & 1_{N_{in} \times
 * N_{in}} \\ -D_{no}^{-1} L_{no \times in} & -D_{no}^{-1} L_{no \times in}
 * & 0_{no \times in}\\-M_{in}^{-1}L_{in \times in} & -M_{in}L_{in\times no}
 * &-M_{in}^{-1} D_{in}\end{bmatrix}\f$
 *
 * For a more detailed explanation see, e.g,
 * J. Fritzsch and P. Jacquod, IEEE Access 10, 19986 (2022).
 *
 * @param y Current function values
 * @return The Jacobian of the system for the function value y
 */
arma::sp_mat Network::df(arma::vec &y) {
  arma::sp_mat Lap(nNodes, nNodes);
  for (std::vector<double>::size_type i = 0; i < nNodes; ++i) {
    for (auto &j : adjacencyList[i]) {
      Lap(i, j.first) = -j.second * std::cos(y(i) - y(j.first));
    }
  }
  for (std::size_t i = 0; i < nNodes; ++i) {
    Lap(i, i) = -arma::sum(Lap.row(i));
  }
  for (std::size_t i = 0; i < nInertia; ++i) {
    Lap.row(i) /= inertia(i);
  }
  for (std::size_t i = nInertia; i < nNodes; ++i) {
    Lap.row(i) /= damping(i);
  }
  arma::sp_mat gamma(nInertia, nInertia);
  gamma.diag() = -damping.head_rows(nInertia) / inertia.head_rows(nInertia);
  arma::sp_mat A = arma::join_cols(
      arma::join_rows(arma::sp_mat(nInertia, nNodes),
                      arma::speye(nInertia, nInertia)),
      arma::join_rows(-Lap.tail_rows(nNodes - nInertia),
                      arma::sp_mat(nNodes - nInertia, nInertia)),
      arma::join_rows(-Lap.head_rows(nInertia), gamma));
  return A;
}

/**
 * Save simulation data.
 * Save the data for the type specified ("angles", "frequency") to the specified
 * time including or excluding the time data.
 * @param path Path and name of the file that should be written to
 * @param type Whether to save frequencies or angles. Must be `"frequency"` or
 * `"angles"`
 * @param se Save every `se`th time step to reduce file size
 * @param time Whether to include the time data. Recommened to leave true.
 */
void Network::saveData(std::string path, std::string type, int se, bool time) {
  arma::uvec ix(yData.n_cols % se == 0 ? yData.n_cols / se
                                       : yData.n_cols / se + 1);
  for (int i = 0; i < ix.n_elem; ++i) {
    ix(i) = i * se;
  }
  if (type.compare("frequency") == 0) {
    if (time) {
      arma::mat tmp = arma::join_cols(arma::conv_to<arma::rowvec>::from(t),
                                      // ydata.rows(N, N + Nin - 1));
                                      yData.rows(nNodes, 2 * nNodes - 1));
      arma::conv_to<arma::mat>::from(tmp.cols(ix)).save(path, arma::csv_ascii);
    } else {
      arma::conv_to<arma::mat>::from(
          yData.submat(arma::linspace<arma::uvec>(nNodes, 2 * nNodes - 1), ix))
          .save(path, arma::csv_ascii);
    }
  } else if (type.compare("angles") == 0) {
    if (time) {
      arma::mat tmp = arma::join_cols(arma::conv_to<arma::rowvec>::from(t),
                                      yData.rows(0, nNodes - 1));
      arma::conv_to<arma::mat>::from(tmp.cols(ix)).save(path, arma::csv_ascii);
    } else {
      arma::conv_to<arma::mat>::from(
          yData.submat(arma::linspace<arma::uvec>(0, nNodes - 1), ix))
          .save(path, arma::csv_ascii);
    }
  }
}

/**
 * Calculate the frequencies of the load buses.
 * A second order approximation of the derivative is used. To obtain the values
 * for an unevenly spaced grid the algorithm described in B. Fornberg, Math.
 * Comp. 51, 699 (1988). is implemented.
 */
arma::mat Network::calculateLoadFrequencies() {
  arma::mat re(nNodes - nInertia, yData.n_cols);
  arma::span ix{arma::span(nInertia, nNodes - 1)};
  std::size_t NN{yData.n_cols};

  // For the edge cases use a forward / backward differences method
  re.col(0) = ((2 * t(0) - t(1) - t(2)) / ((t(1) - t(0)) * (t(2) - t(0))) *
                   yData(ix, 0) +
               (t(2) - t(0)) / ((t(1) - t(0)) * (t(2) - t(1))) * yData(ix, 1) -
               (t(1) - t(0)) / ((t(2) - t(0)) * (t(2) - t(1))) * yData(ix, 2));
  re.col(NN - 1) = ((2 * t(NN - 1) - t(NN - 2) - t(NN - 3)) /
                        ((t(NN - 1) - t(NN - 3)) * (t(NN - 1) - t(NN - 2))) *
                        yData(ix, NN - 1) -
                    (t(NN - 1) - t(NN - 3)) /
                        ((t(NN - 2) - t(NN - 3)) * (t(NN - 1) - t(NN - 2))) *
                        yData(ix, NN - 2) +
                    (t(NN - 1) - t(NN - 2)) /
                        ((t(NN - 2) - t(NN - 3)) * (t(NN - 1) - t(NN - 3))) *
                        yData(ix, NN - 3));

  // For the central points use cenetered differences method
  for (std::size_t i = 1; i < NN - 1; ++i) {
    re.col(i) =
        ((t(i) - t(i + 1)) / ((t(i) - t(i - 1)) * (t(i + 1) - t(i - 1))) *
             yData(ix, i - 1) +
         (1 / (t(i) - t(i - 1)) - 1 / (t(i + 1) - t(i))) * yData(ix, i) +
         (t(i) - t(i - 1)) / ((t(i + 1) - t(i - 1)) * (t(i + 1) - t(i))) *
             yData(ix, i + 1));
  }
  return re;
}

/**
 * Scale dynamical parameters.
 * Multiply the specified dynamical parameters by the given factor.
 * @param factor The value by which the parameters are multiplied
 * @param type Which parameters to scale. Must be `"inertia"` or `"inertia"`
 * @todo Allow to specify single machines
 */
void Network::scaleParameters(double factor, std::string type) {
  if (type.compare("damping") == 0) {
    damping *= factor;
  } else if (type.compare("inertia") == 0) {
    inertia *= factor;
  }
}

/**
 * Plot results of dynamical simulations.
 * Plot all variables of the given type. Not recommended for large networks as
 * the GNUplot pipeline is extremely slows.
 * @param type Whether to plot frequencies or angles. Must be `"frequency"` or
 * `"angles"`
 */
void Network::plotResults(std::string type) {
  if (type.compare("frequency") == 0) {
    std::vector<std::vector<double>> tmp;
    for (int i = 0; i < nNodes; ++i) {
      tmp.push_back(
          arma::conv_to<std::vector<double>>::from(yData.row(i + nNodes)));
    }
    matplot::plot(t, tmp, "k");
  } else if (type.compare("angles") == 0) {
    std::vector<std::vector<double>> tmp;
    for (int i = 0; i < nNodes; ++i) {
      tmp.push_back(arma::conv_to<std::vector<double>>::from(yData.row(i)));
    }
    matplot::plot(t, tmp, "k");
  }
}

/**
 * Plot results of dynamical simulations divided into areas.
 * Plot all variables of the given type divided into given areas. The plot
 * consists of a subplot for each area with 2 pots per row. Not recommended for
 * large networks as the GNUplot pipeline is extremely slows.
 * @param areafile File containing the different areas. It must contain the
 * nodes in one area on one line separated by spaces.
 * @param type Whether to plot frequencies or angles. Must be `"frequency"` or
 * `"angles"`
 */
void Network::plotResults(std::string areafile, std::string type) {
  assert(std::filesystem::exists(areafile));
  std::ifstream areain{areafile};
  std::vector<std::vector<std::size_t>> areas;
  std::string line;
  while (std::getline(areain, line)) {
    std::vector<std::size_t> tmp;
    std::size_t node;
    std::istringstream ss{line};
    while (ss >> node) {
      tmp.push_back(node);
    }
    areas.push_back(tmp);
  }

  // only plot a point every 0.05 seconds
  int se{static_cast<int>(round(0.05 / (t[1] - t[0])))};
  arma::uvec ix(yData.n_cols % se == 0 ? yData.n_cols / se
                                       : yData.n_cols / se + 1);
  for (int i = 0; i < ix.n_elem; ++i) {
    ix(i) = i * se;
  }
  int n_rows = areas.size() / 2 + 1;
  for (int i = 0; i < areas.size(); ++i) {
    std::vector<std::vector<double>> tmp;
    for (auto &j : areas[i]) {
      tmp.push_back(arma::conv_to<std::vector<double>>::from(
          yData.submat(arma::uvec{j + nNodes}, ix)));
    }
    matplot::subplot(n_rows, 2, i);
    matplot::plot(arma::conv_to<arma::vec>::from(t.rows(ix)), tmp, "k");
  }
  matplot::show();
}
}  // namespace net
