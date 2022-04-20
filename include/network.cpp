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
Network::Network(std::string adjlist, std::string coeffs) {
  create_adjlist(adjlist);
  create_coefflists(coeffs);
  set_initial_angles();
}

Network::Network(std::string adjlist, std::string coeffs, std::string angles) {
  create_adjlist(adjlist);
  create_coefflists(coeffs);
  set_initial_angles(angles);
}

// Read in the adjacency list given in the format:
// node1, node2, weight
// The vector at position i contains all nodes connected to node i with the
// weight of the line. The connected node and the weight are stored as a pair.
void Network::create_adjlist(std::string adjlist) {
  assert(std::filesystem::exists(adjlist));
  std::ifstream adjinput(adjlist);
  std::string line;
  std::vector<double>::size_type startnode{}, endnode{};
  double lineweight{};
  int colindex{};
  double tempin{};
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
    while (al.size() <= startnode || al.size() <= endnode) {
      al.push_back(std::vector<std::pair<int, double>>());
    }
    al[startnode].push_back({endnode, lineweight});
    al[endnode].push_back({startnode, lineweight});
  }
  adjinput.close();
  N = al.size();
}

// Read in the coefficients list in the format
// node, inertia, damping, power
// At the moment the inertia-full nodes need to be consecutive at the beginning
// TODO: Allow for arbitrarily mixed inertia-full and inertia-less nodes.
void Network::create_coefflists(std::string coeffs) {
  assert(std::filesystem::exists(coeffs));
  // Make m, d, and p to be of appropriate size
  m = arma::vec(N, arma::fill::zeros);
  d = arma::vec(N, arma::fill::zeros);
  p = arma::vec(N, arma::fill::zeros);
  std::ifstream coeffin(coeffs);
  std::string line;
  int colindex{};
  double tempin{};
  std::size_t gen{};
  while (getline(coeffin, line)) {
    colindex = 0;
    std::stringstream ss(line);
    while (ss >> tempin) {
      switch (colindex) {
        case 0:
          gen = static_cast<arma::size_t>(round(tempin));
          break;
        case 1:
          m(gen) = tempin;
          break;
        case 2:
          d(gen) = tempin;
          break;
        case 3:
          p(gen) = tempin;
          break;
        default:
          break;
      }
      if (ss.peek() == ',') {
        ss.ignore();
      }
      arma::vec tmp(N, arma::fill::value(1.0e-6));
      Nin = arma::sum(m > tmp);
      ++colindex;
    }
  }
  coeffin.close();
}

void Network::set_initial_angles() { theta0 = arma::vec(N, arma::fill::zeros); }

void Network::set_initial_angles(std::string angles) {
  assert(std::filesystem::exists(angles));
  theta0 = arma::vec(N);
  theta0.load(angles, arma::csv_ascii);
}

void Network::step(std::size_t node, double power) { p(node) += power; }
void Network::box(std::size_t node, double power, double time) {
  boxp = true;
  boxix = node;
  boxpower = power;
  p(node) += power;
  boxtime = time;
}

// Do a dynamical simulation using a semi-implicit mid-point method
void Network::dynamical_simulation(double t0, double tf, double dt, int se) {
  double tt{t0};
  arma::size_t Nstep{static_cast<arma::size_t>(std::round((tf - t0) / dt)) +
                     1};  // Number of steps
  arma::size_t saveN{Nstep % se == 0 ? Nstep / se : Nstep / se + 1};
  arma::vec y{arma::join_cols(theta0, arma::vec(Nin))};
  t = arma::vec(saveN);
  ydata = arma::mat(N + Nin, saveN);
  ydata.col(0) = y;
  arma::vec deltaCurrent(N + Nin);
  t(0) = tt;
  arma::mat toinv = arma::eye(N + Nin, N + Nin) - dt * arma::mat(df(y));
  arma::mat inv = arma::inv(toinv);
  deltaCurrent = dt * inv * f(y);
  y += deltaCurrent;
  tt += dt;

  for (int i = 1; i < Nstep - 1; ++i) {
    // if (boxp) {
    //   if (tt >= boxtime) {
    //     p(boxix) -= boxpower;
    //     boxp = false;
    //   }
    // }
    if (i % se == 0) {
      std::cout << std::fixed << std::setprecision(1)
                << (tt - t0) / (tf - t0) * 100
                << "%\nMax omega: " << std::setprecision(4)
                << arma::max(arma::abs(y(arma::span(N, N + Nin - 1))))
                << "\n\x1b[A\u001b[2K\x1b[A\u001b[2K";
      ydata.col(i / se) = y;
      t(i / se) = tt;
    }
    deltaCurrent += 2 * inv * (dt * f(y) - deltaCurrent);
    y += deltaCurrent;
    tt += dt;
  }
  if ((Nstep - 1) % se == 0) {
    deltaCurrent = inv * (dt * f(y) - deltaCurrent);
    y += deltaCurrent;
    t(saveN - 1) = tt;
    ydata.col(saveN - 1) = y;
  }
  ydata.insert_rows(ydata.n_rows, calculate_load_frequencies(dt, se));
  std::cout << "Final max omega: "
            << arma::max(arma::abs(y(arma::span(N, N + Nin - 1)))) << "\n";
}

// Do a dynamical simulation using a the fourth order Kaps-Rentrop method
void Network::kaps_rentrop(double t0, double tf, double dt, int se,
                           int mtries) {
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
  // const double e1{17.0 / 54.0};
  // const double e2{7.0 / 36.0};
  // const double e3{0.0};
  // const double e4{125.0 / 108.0};
  const double c1x{1.0 / 2.0};
  const double c2x{-3.0 / 2.0};
  const double c3x{121.0 / 50.0};
  const double c4x{29.0 / 250.0};

  double tt{t0};
  arma::size_t Nstep{static_cast<arma::size_t>(std::round((tf - t0) / dt)) +
                     1};  // Number of steps
  arma::size_t saveN{Nstep % se == 0 ? Nstep / se : Nstep / se + 1};
  arma::vec y{arma::join_cols(theta0, arma::vec(Nin))};
  t = arma::vec(saveN);
  ydata = arma::mat(N + Nin, saveN);
  arma::vec deltaCurrent(N + Nin);
  ydata.col(0) = y;
  t(0) = tt;
  for (int i = 1; i < Nstep; ++i) {
    // if (boxp) {
    //   if (tt >= boxtime) {
    //     p(boxix) -= boxpower;
    //     boxp = false;
    //   }
    // }
    if (i % se == 0) {
      std::cout << std::fixed << std::setprecision(1)
                << (tt - t0) / (tf - t0) * 100
                << "%\nMax omega: " << std::setprecision(4)
                << arma::max(arma::abs(y(arma::span(N, N + Nin - 1))))
                << "\n\x1b[A\u001b[2K\x1b[A\u001b[2K";
      ydata.col(i / se) = y;
      t(i / se) = tt;
    }
    arma::sp_mat jac = df(y);
    arma::sp_mat ls{arma::speye(N + Nin, N + Nin) / gam / dt - jac};
    arma::vec k1{arma::spsolve(ls, f(y))};
    arma::vec k2{arma::spsolve(ls, f(y + a21 * k1) + c21 * k1 / dt)};
    arma::vec k3{arma::spsolve(
        ls, f(y + a31 * k1 + a32 * k2) + c31 * k1 / dt + c32 * k2 / dt)};
    arma::vec k4{arma::spsolve(ls, f(y + a31 * k1 + a32 * k2) + c41 * k1 / dt +
                                       c42 * k2 / dt + c43 * k3 / dt)};
    // arma::vec k1{bigradient(ls, f(y), arma::vec{ls.n_rows,
    // arma::fill::zeros})}; arma::vec k2{bigradient(ls, f(y + a21 * k1) + c21 *
    // k1 / dt, k1)}; arma::vec k3{bigradient(
    //     ls, f(y + a31 * k1 + a32 * k2) + c31 * k1 / dt + c32 * k2 / dt, k2)};
    // arma::vec k4{bigradient(ls,
    //                         f(y + a31 * k1 + a32 * k2) + c41 * k1 / dt +
    //                             c42 * k2 / dt + c43 * k3 / dt,
    //                         k3)};
    // arma::mat L, U, P;
    // arma::lu(L, U, P, ls);
    // arma::vec k1{f(y)};
    // lusolve(L, U, P, k1);
    // arma::vec k2{f(y + a21 * k1) + c21 * k1 / dt};
    // lusolve(L, U, P, k2);
    // arma::vec k3{f(y + a31 * k1 + a32 * k2) + c31 * k1 / dt + c32 * k2 / dt};
    // lusolve(L, U, P, k3);
    // arma::vec k4{f(y + a31 * k1 + a32 * k2) + c41 * k1 / dt + c42 * k2 / dt +
    //              c43 * k3 / dt};
    // lusolve(L, U, P, k4);
    y += b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4;
    tt += dt;
  }
  ydata.insert_rows(ydata.n_rows, calculate_load_frequencies(dt, se));
  std::cout << "Final max omega: "
            << arma::max(arma::abs(y(arma::span(N, N + Nin - 1)))) << "\n";
}

template <typename T>
arma::vec Network::bigradient(T &A, arma::vec b, arma::vec xk, double error) {
  // First check if xk is already a solution
  if (arma::approx_equal(A * xk, b, "absdiff", error)) {
    return xk;
  }
  // Set up initial values and define all vectors
  arma::vec rk{b - A * xk};
  arma::vec rbk = rk;
  arma::vec pk = rk;
  arma::vec pbk = rk;
  arma::vec rk1, rbk1, pk1, pbk1;
  double bk;
  // Do the first step
  double ak{arma::conv_to<double>::from((rbk.t() * rk) / (pbk.t() * A * pk))};
  arma::vec xk1{xk + ak * pk};
  if (arma::max(arma::abs(xk - xk1)) < error) {
    return xk1;
  }
  xk = xk1;
  // Do the iteration
  for (std::size_t i = 0; i < b.n_rows - 1; ++i) {
    rk1 = rk - ak * A * pk;
    rbk1 = rbk - ak * A.t() * pbk;
    bk = arma::conv_to<double>::from((rbk1.t() * rk1) / (rbk.t() * rk));
    pk1 = rk1 + bk * pk;
    pbk1 = rbk1 + bk * pbk;
    rk = rk1;
    rbk = rbk1;
    pk = pk1;
    pbk = pbk1;
    ak = arma::conv_to<double>::from((rbk.t() * rk) / (pbk.t() * A * pk));
    xk1 = xk + ak * pk;
    if (arma::max(arma::abs(xk - xk1)) < error) {
      return xk1;
    }
    xk = xk1;
  }
  return xk1;
}

void Network::lusolve(arma::mat &L, arma::mat &U, arma::mat &P, arma::vec &x) {
  x = P * x;
  for (std::size_t i = 0; i < x.n_rows; ++i) {
    for (std::size_t j = 0; j < i; ++j) {
      x(i) -= L(i, j) * x(j);
    }
    x(i) /= L(i, i);
  }
  for (std::size_t i = x.n_rows; i-- > 0;) {
    for (std::size_t j = i + 1; j < x.n_rows; ++j) {
      x(i) -= U(i, j) * x(j);
    }
    x(i) /= U(i, i);
  }
}

arma::vec Network::f(arma::vec y) {
  arma::vec re(N + Nin);
  for (std::size_t i = 0; i < Nin; ++i) {
    re(i) = y(N + i);
    re(N + i) = p(i) - d(i) * y(N + i);
    for (auto &j : al[i]) {
      re(N + i) -= j.second * std::sin(y(i) - y(j.first));
    }
    re(N + i) /= m(i);
  }
  for (std::size_t i = Nin; i < N; ++i) {
    re(i) = p(i);
    for (auto &j : al[i]) {
      re(i) -= j.second * std::sin(y(i) - y(j.first));
    }
    re(i) /= d(i);
  }
  return re;
}

// Calculate the Jacobian for specific values of theta
// It is given by the following expression
//┏                                                                         ┓
//┃ 0(Nin x Nin)   0(Nin x Nno)   I(Nin x Nin)                              ┃
//┃                                                                         ┃
//┃ -D(Nno)^(-1)*L(Nno x Nin) -D(Nno)^(-1)*L(Nno x Nno) 0(Nno x Nin)        ┃
//┃                                                                         ┃
//┃ -M(Nno)^(-1)*L(Nin x Nin) -M(Nno)^(-1)*L(Nin x Nno) -M^(-1)*D(Nin xNin) ┃
//┗                                                                         ┛
arma::sp_mat Network::df(arma::vec &y) {
  arma::sp_mat Lap(N, N);
  for (std::vector<double>::size_type i = 0; i < N; ++i) {
    for (auto &j : al[i]) {
      Lap(i, j.first) = -j.second * std::cos(y(i) - y(j.first));
    }
  }
  for (std::size_t i = 0; i < N; ++i) {
    Lap(i, i) = -arma::sum(Lap.row(i));
  }
  for (std::size_t i = 0; i < Nin; ++i) {
    Lap.row(i) /= m(i);
  }
  for (std::size_t i = Nin; i < N; ++i) {
    Lap.row(i) /= d(i);
  }
  arma::sp_mat gamma(Nin, Nin);
  gamma.diag() = -d.head_rows(Nin) / m.head_rows(Nin);
  arma::sp_mat A = arma::join_cols(
      arma::join_rows(arma::sp_mat(Nin, N), arma::speye(Nin, Nin)),
      arma::join_rows(-Lap.tail_rows(N - Nin), arma::sp_mat(N - Nin, Nin)),
      arma::join_rows(-Lap.head_rows(Nin), gamma));
  return A;
}

// Save the data for the type specified ("angles", "frequency") to path with
// time included if time = true
void Network::save_data(std::string path, std::string type, int se, bool time) {
  arma::uvec ix(ydata.n_cols % se == 0 ? ydata.n_cols / se
                                       : ydata.n_cols / se + 1);
  for (int i = 0; i < ix.n_elem; ++i) {
    ix(i) = i * se;
  }
  if (type.compare("frequency") == 0) {
    if (time) {
      arma::mat tmp = arma::join_cols(arma::conv_to<arma::rowvec>::from(t),
                                      ydata.rows(N, 2 * N - 1));
      arma::conv_to<arma::mat>::from(tmp.cols(ix)).save(path, arma::csv_ascii);
    } else {
      arma::conv_to<arma::mat>::from(
          ydata.submat(arma::linspace<arma::uvec>(N, 2 * N - 1), ix))
          .save(path, arma::csv_ascii);
    }
  } else if (type.compare("angles") == 0) {
    if (time) {
      arma::mat tmp = arma::join_cols(arma::conv_to<arma::rowvec>::from(t),
                                      ydata.rows(0, N - 1));
      arma::conv_to<arma::mat>::from(tmp.cols(ix)).save(path, arma::csv_ascii);
    } else {
      arma::conv_to<arma::mat>::from(
          ydata.submat(arma::linspace<arma::uvec>(0, N - 1), ix))
          .save(path, arma::csv_ascii);
    }
  }
}

// Calculate the frequencies of the load buses with a second order approximation
// of the derivative
arma::mat Network::calculate_load_frequencies(double dt, int se) {
  arma::mat re(N - Nin, ydata.n_cols);
  re.col(0) = (-1.5 * ydata(arma::span(Nin, N - 1), 0) +
               2 * ydata(arma::span(Nin, N - 1), 1) -
               0.5 * ydata(arma::span(Nin, N - 1), 2)) /
              (dt * se);
  for (std::size_t i = 1; i < re.n_cols - 1; ++i) {
    re.col(i) = (-ydata(arma::span(Nin, N - 1), i - 1) +
                 ydata(arma::span(Nin, N - 1), i + 1)) /
                (2 * dt * se);
  }
  re.col(re.n_cols - 1) = (1.5 * ydata(arma::span(Nin, N - 1), re.n_cols - 1) -
                           2 * ydata(arma::span(Nin, N - 1), re.n_cols - 2) +
                           0.5 * ydata(arma::span(Nin, N - 1), re.n_cols - 3)) /
                          (dt * se);
  return re;
}

void Network::plot_results(std::string type) {
  if (type.compare("frequency") == 0) {
    std::vector<std::vector<double>> tmp;
    for (int i = 0; i < N; ++i) {
      tmp.push_back(arma::conv_to<std::vector<double>>::from(ydata.row(i + N)));
    }
    matplot::plot(t, tmp, "k");
  } else if (type.compare("angles") == 0) {
    std::vector<std::vector<double>> tmp;
    for (int i = 0; i < N; ++i) {
      tmp.push_back(arma::conv_to<std::vector<double>>::from(ydata.row(i)));
    }
    matplot::plot(t, tmp, "k");
  }
}

void Network::plot_results(std::string areafile, std::string type) {
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
  int se{static_cast<int>(round(0.05 / (t[1] - t[0])))};
  arma::uvec ix(ydata.n_cols % se == 0 ? ydata.n_cols / se
                                       : ydata.n_cols / se + 1);
  for (int i = 0; i < ix.n_elem; ++i) {
    ix(i) = i * se;
  }
  int n_rows = areas.size() / 2 + 1;
  for (int i = 0; i < areas.size(); ++i) {
    std::vector<std::vector<double>> tmp;
    for (auto &j : areas[i]) {
      tmp.push_back(arma::conv_to<std::vector<double>>::from(
          ydata.submat(arma::uvec{j + N}, ix)));
    }
    matplot::subplot(n_rows, 2, i);
    matplot::plot(arma::conv_to<arma::vec>::from(t.rows(ix)), tmp, "k");
  }
  matplot::show();
}
}  // namespace net
