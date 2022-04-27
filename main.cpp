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

#include <matplot/matplot.h>

#include <armadillo>
#include <iostream>
#include <string>

#include "ProjectConfig.hpp"
#include "include/network.hpp"

int main(int argv, char **argc) {
  if (argv == 2) {
    if (!std::string_view(argc[1]).compare("-v")) {
      std::cout << "Network-cpp version " << VERSION_MAJOR << "."
                << VERSION_MINOR << "." << VERSION_PATCH << std::endl;
      return 0;
    }
  }
  std::string adjlist = std::string(SOURCE_DIR) + "/data/adjlist.csv";
  std::string coeffs = std::string(SOURCE_DIR) + "/data/coeffs.csv";
  std::string angles = std::string(SOURCE_DIR) + "/data/syncedangles.csv";

  std::vector<std::pair<double, std::string>> scaling{
      {0.01, "0.01"}, {0.05, "0.05"}, {0.1, "0.1"}, {0.5, "0.5"},
      {1, "1"},       {5, "5"},       {10, "10"}};

  net::Network panta{adjlist, coeffs, angles};
  panta.noise(419, 25, 0.01);
  panta.dynamicalSimulation(0, 25, "midpoint", 5.0e-3, 5.0e-2);
  panta.saveData("noisetestmid.csv", "frequency", 1);

  for (auto &i : scaling) {
    std::cout << i.second << '\n';
    net::Network tmp{adjlist, coeffs, angles};
    tmp.step(419, -9.0);
    tmp.scaleParameters(i.first, "damping");
    tmp.dynamicalSimulation(0, 40);
    tmp.saveData("damping" + i.second + ".csv", "frequency");
  }

  for (auto &i : scaling) {
    std::cout << i.second << '\n';
    net::Network tmp{adjlist, coeffs, angles};
    tmp.step(419, -9.0);
    tmp.scaleParameters(i.first, "inertia");
    tmp.dynamicalSimulation(0, 40);
    tmp.saveData("inertia" + i.second + ".csv", "frequency");
  }

  return 0;
}
