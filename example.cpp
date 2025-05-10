#include <iostream>

#include "Eigen/Dense"
#include "absl/status/statusor.h"
#include "riccati/riccati.h"

// Run with one of
// `bazel run --config=macos //:example`
// `bazel run --config=linux //:example`
int main() {
    // Triple integrator sampled at 10 Hz (zero-order hold) with unity weights:
    static constexpr double dt = 0.1;
    const Eigen::Matrix3d Ac{
          {0.0, 1.0, 0.0},
          {0.0, 0.0, 1.0},
          {0.0, 0.0, 0.0}
    };
    const Eigen::Matrix3d A =
          Eigen::Matrix3d::Identity() + dt * Ac + (1.0 / 2.0) * dt * dt * Ac * Ac;
    const Eigen::Vector3d B{
          {(1.0 / 6.0) * dt * dt * dt, (1.0 / 2.0) * dt * dt, dt}
    };
    const Eigen::DiagonalMatrix<double, 3> Q = Eigen::DiagonalMatrix<double, 3>::Identity();
    const Eigen::DiagonalMatrix<double, 1> R = Eigen::DiagonalMatrix<double, 1>::Identity();

    // Solve the discrete-time algebraic Riccati equation with the approach from Van Dooren (1981):
    const absl::StatusOr<Eigen::MatrixXd> maybeX = riccati::SolveDiscrete(
          A, B, Q.toDenseMatrix(), R.toDenseMatrix(), riccati::Solver::VanDooren);
    if (!maybeX.ok()) {
        std::cout << "Error -- " << maybeX.status().message() << std::endl;
        return 1;
    }
    // Check numerical error:
    const Eigen::Matrix3d residual =
          riccati::Residual(A, B, Q.toDenseMatrix(), R.toDenseMatrix(), maybeX.value());

    std::cout << "X =\n" << maybeX.value() << std::endl;
    std::cout << "residual:\n" << residual << std::endl;
    std::cout << "residual.norm() = " << residual.norm() << std::endl;

    return 0;
}
