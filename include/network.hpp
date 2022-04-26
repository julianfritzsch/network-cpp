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

#ifndef PANTAGRUEL_HPP
#define PANTAGRUEL_HPP

#include <matplot/matplot.h>

#include <armadillo>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace net {
/**
 * @class Network
 * Simulate electrical power networks. Create an electric power network from an
 * adjacency list and a list of coefficients. Functionality to create various
 * faults such as step and box perturbations and noisy fluctuations (to be
 * implemented). The dynamic behavior can be simulated using various numerical
 * methods and the results exported to a file. Also provides basic plotting
 * functionality.
 * @todo Add explicit solvers for non-stiff problems
 * @todo Add possibility to use noise as perturbation
 */
class Network {
 public:
  // Member functions
  Network(std::string adjlist, std::string coeffs);
  Network(std::string adjlist, std::string coeffs, std::string angles);
  void step(std::size_t node, double powerChange);
  void box(std::size_t node, double powerChange, double time);
  void dynamicalSimulation(double t0, double tf,
                           std::string method = "midpoint", double dt = 5.0e-3,
                           double dtMax = 1e-1, double eps = 1.0e-3,
                           int maxTries = 40);
  void saveData(std::string path, std::string type = "frequency", int se = 10,
                bool time = true);
  void plotResults(std::string type = "frequency");
  void plotResults(std::string areafile, std::string type = "frequency");
  void scaleParameters(double factor, std::string type);
  void noise(std::size_t node, double tau0, double stddev);
  void noise(std::size_t node, double tau0, double stddev, unsigned int seed);

 private:
  void createAdjlist(std::string adjlist);
  void createCoeffLists(std::string coeffs);
  void setInitialAngles();
  void setInitialAngles(std::string angles);
  arma::mat calculateLoadFrequencies();
  arma::vec f(arma::vec y);
  arma::sp_mat df(arma::vec& y);
  void midpoint(double t0, double tf, double dt = 5.0e-3);
  void midpointNoise(double t0, double tf, double dt = 5.0e-3);
  void kapsRentrop(double t0, double tf, double dtStart = 5.0e-3,
                   double dtMax = 1e-1, double eps = 1.0e-3, int maxTries = 40);

  // Data members
  /**
   * Adjacency list. It is setup as follows: the vector entry at point i
   * contains all the nodes connected to i and their weight saved as a pair
   * <node, weight>
   */
  std::vector<std::vector<std::pair<int, double>>> adjacencyList;
  /** Number of nodes */
  std::size_t nNodes;
  /** Number of nodes with inertia (generator nodes) */
  std::size_t nInertia;
  /** Vector containing inertia of all nodes (zero for inertia-less nodes) */
  arma::vec inertia;
  /** Vector containing damping of all nodes */
  arma::vec damping;
  /** Vector containing power production / consumptions of all nodes */
  arma::vec power;
  /** Vector containing initial angles */
  arma::vec initAngles;
  /** Vector containing time stamps after a dynamical simulation */
  arma::vec t;
  /** Angle and frequency data.
   * Contains the angle and frequency data after a dynamical simulation has been
   * performed. The first Network::nNodes rows contain the angle values and rows
   * Network::nNodes till end contain the frequencies. Each column `i`
   * corresponds to time_step Network::t(i).
   */
  arma::mat yData;
  /** Whether there is a box perturbation */
  bool boxPer{false};
  /** Index of node affected by box perturbation */
  std::size_t boxIx;
  /** Change in power due to box perturbation */
  double boxPower;
  /** Duration of box perturabtion */
  double boxTime;
  /** Whether there is a noisy perturbation */
  bool noisePer{false};
  /** Vector of indices to which a noisy perturbation is applied */
  arma::uvec noiseIndices;
  /** Vector of correlation times */
  std::vector<double> tau;
  /** Vector of seeds for noisy perturbation */
  std::vector<unsigned int> seeds;
  /** Vector of random number generators */
  std::vector<std::mt19937> gen;
  /** Vector of normal distributions */
  std::vector<std::normal_distribution<>> normalDist;
};
}  // namespace net
#endif
