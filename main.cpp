#include "include/network.hpp"
#include <armadillo>
#include <iostream>

int main(int arv, char **argc) {
  arma::arma_version ver;
  std::cout << "Armadillo version: " << ver.as_string() << '\n';
  net::Network test;
}
