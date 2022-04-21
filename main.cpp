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

#include <armadillo>
#include <iostream>
#include <string>
#include <matplot/matplot.h>

#include "include/network.hpp"
#include "ProjectConfig.hpp"

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
  net::Network test{adjlist, coeffs, angles};
  test.step(419, -9.0);
  // test.box(419, -9.0, 10);
  test.kapsRentrop(0, 20, 1.0e-2);
  test.saveData("testrk.csv", "frequency", 1);
  test.saveData("testrka.csv", "angles", 1);

  return 0;
}
