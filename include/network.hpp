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
#include <string>
#include <utility>
#include <vector>

namespace net {
// Network class containing both the setup of the network as well as functions
// to set different types of perturbations and do dynamical simulations.
class Network {
 public:
  // Network();
  Network(std::string adjlist, std::string coeffs);
  Network(std::string adjlist, std::string coeffs, std::string angles);
  void step(std::size_t node, double power);
  void box(std::size_t node, double power, double time);
  void dynamical_simulation(double t0, double tf, double dt = 5.0e-3,
                            int se = 1);
  void save_data(std::string path, std::string type = "frequency", int se = 10,
                 bool time = true);
  void plot_results(std::string type = "frequency");
  void plot_results(std::string areafile, std::string type = "frequency");
  void kaps_rentrop(double t0, double tf, double dtstart = 5.0e-3,
                    double dtmax = 1e-1, double eps = 1.0e-3, int mtries = 40);

 private:
  // The adjacency list is setup as follows: the vector entry at point i
  // contains all the nodes connected to i and their weight saved as a pair
  // <node, weight>
  std::vector<std::vector<std::pair<int, double>>> al;  // Adjacency list
  std::size_t N;                                        // Number of nodes
  std::size_t Nin;   // Number of nodes with inertia
  arma::vec m;       // Inertia
  arma::vec d;       // Damping
  arma::vec p;       // Power
  arma::vec theta0;  // Vector of initial angles
  arma::vec t;       // Vector of time stapms for a dynamical simulation
  arma::mat ydata;   // Matrix containing angles and frequencies

  void create_adjlist(std::string adjlist);  // Create the adjacency list from
                                             // the file adjlist
  void create_coefflists(
      std::string coeffs);    // Create the inertia, damping, and
                              // power vectors from the file coeffs
  void set_initial_angles();  // Set initial angles to zero
  void set_initial_angles(
      std::string angles);  // Set initial angles to the values
                            // given in the file angles
  arma::mat
  calculate_load_frequencies();   // Get frequencies of the non-inertia nodes
  arma::vec f(arma::vec y);       // Function value
  arma::sp_mat df(arma::vec& y);  // Jacobian of the system
  bool boxp{false};
  std::size_t boxix;
  double boxpower;
  double boxtime;
};
}  // namespace net
#endif
