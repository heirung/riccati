#pragma once

#include "Eigen/Dense"
#include "absl/status/statusor.h"

namespace laub {
namespace internal {
// Construct the symplectic matrix Z =
//   [ A + GA⁻ᵀQ, -GA⁻ᵀ ]
//   [     -A⁻ᵀQ,   A⁻ᵀ ]
// where G = BR⁻¹Bᵀ. See Eq. (9) in Laub (1979).
absl::StatusOr<Eigen::MatrixXd>
GetZ(const Eigen::Ref<const Eigen::MatrixXd>& A,
     const Eigen::Ref<const Eigen::MatrixXd>& B,
     const Eigen::Ref<const Eigen::MatrixXd>& Q,
     const Eigen::Ref<const Eigen::MatrixXd>& R);

// Determine U, Z, and the eigenvalues in the transformation UᵀZU = S, where S is upper
// quasi-triangular,
//   S = [S₁₁, S₁₂], and U = [U₁₁, U₁₂]
//       [  0, S₂₂]          [U₂₁, U₂₂]
// The eigenvalues of the blocks S₁₁ and S₂₂ are stable and unstable, respectively. The n leading
// columns of the matrix U are the Schur vectors that span the stable invariant subspace. The
// solution to the Riccati equation is X = U₂₁U₁₁⁻¹.
// Returns {U₁₁, U₂₁}.
absl::StatusOr<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>>
GetSchurVectors(const Eigen::Ref<const Eigen::MatrixXd>& Z);
}  // namespace internal

// Solve the discrete-time algebraic Riccati equation (DARE) using the approach by Laub (1979). This
// approach requires that both A and R be nonsingular (not checked).
// Laub, A. J. A Schur method for solving algebraic Riccati equations. IEEE Transactions on
//   Automatic Control, 24(6):913–921, 1979. https://doi.org/10.1109/TAC.1979.1102178
absl::StatusOr<Eigen::MatrixXd> SolveDare(
      const Eigen::Ref<const Eigen::MatrixXd>& A,
      const Eigen::Ref<const Eigen::MatrixXd>& B,
      const Eigen::Ref<const Eigen::MatrixXd>& Q,
      const Eigen::Ref<const Eigen::MatrixXd>& R);
}  // namespace laub

namespace riccati {

enum class Solver { Laub };

absl::StatusOr<Eigen::MatrixXd> SolveDiscrete(
      const Eigen::Ref<const Eigen::MatrixXd>& A,
      const Eigen::Ref<const Eigen::MatrixXd>& B,
      const Eigen::Ref<const Eigen::MatrixXd>& Q,
      const Eigen::Ref<const Eigen::MatrixXd>& R,
      Solver method);

// Compute the (matrix) residual of the discrete-time algebraic Riccati equation (DARE). The
// alternative form produces `NAN`s if R is singular.
Eigen::MatrixXd Residual(
      const Eigen::Ref<const Eigen::MatrixXd>& A,
      const Eigen::Ref<const Eigen::MatrixXd>& B,
      const Eigen::Ref<const Eigen::MatrixXd>& Q,
      const Eigen::Ref<const Eigen::MatrixXd>& R,
      const Eigen::Ref<const Eigen::MatrixXd>& X,
      bool alternativeForm = false);
}  // namespace riccati
