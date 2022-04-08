#ifndef PANTAGRUEL_HPP
#define PANTAGRUEL_HPP

#include <armadillo>
#include <map>
#include <utility>

namespace net {
class Network {
  std::map<std::pair<int, int>, double> al; // Adjacency list
  arma::vec m;                              // Inertia
  arma::vec d;                              // Damping
  arma::vec p;                              // Power
  arma::vec theta0;                         // Vector of initial angles

  void create_adjlist(std::string adjlist); // Create the adjacency list from
                                            // the file adjlist
  void
  create_coefflists(std::string coeffs); // Create the inertia, damping, and
                                         // power vectors from the file coeffs
  void set_initial_angles(); // Set initial angles to zero
  void
  set_initial_angles(std::string angles); // Set initial angles to the values
                                          // given in the file angles
public:
  // Network();
  Network(std::string adjlist, std::string coeffs);
  Network(std::string adjlist, std::string coeffs, std::string angles);
};
} // namespace net
#endif
