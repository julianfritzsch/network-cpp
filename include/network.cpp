#include "../include/network.hpp"
#include <iostream>

namespace net {
// Network::Network() {}

Network::Network(std::string adjlist, std::string coeffs) {
  create_adjlist(adjlist);
  create_coefflists(coeffs);
  set_initial_angles();
  std::cout << "The network has been set-up!" << '\n';
}

Network::Network(std::string adjlist, std::string coeffs, std::string angles) {
  create_adjlist(adjlist);
  create_coefflists(coeffs);
  set_initial_angles(angles);
  std::cout << "The network has been set-up!" << '\n';
}

} // namespace net
