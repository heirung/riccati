#pragma once

#include "Eigen/Dense"
#include "absl/status/statusor.h"

// This header declares functions to solve the discrete-time algebraic Riccati equation (DARE) using
// the approaches by Laub (1979) and Van Dooren (1981). The implementations rely on the LAPACK
// wrappers in `lapack_wrappers.h`. `laub::SolveDare` requires that both A and R be nonsingular,
// whereas `van_dooren::SolveDare` does not. There's little to no error checking in these functions.
//
// Laub, A. J. A Schur method for solving algebraic Riccati equations. IEEE Transactions on
//   Automatic Control, 24(6):913–921, 1979. https://doi.org/10.1109/TAC.1979.1102178
// Van Dooren, P. M. A generalized eigenvalue approach for solving Riccati equations. SIAM Journal
//   on Scientific and Statistical Computing, 2(2):121–135, 1981. https://doi.org/10.1137/0902010
// Arnold, N. III, Laub, A. J. Generalized eigenproblem algorithms and software for algebraic
//   Riccati equations. Proceedings of the IEEE, 72(12):1746–1754, 1984.
//   https://doi.org/10.1109/PROC.1984.13083

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

namespace van_dooren {
namespace internal {
// Construct the matrices L and M in the matrix pencil 𝜆L - M for the discrete algebraic Riccati
// equation. See Eq. (9) in Arnold and Laub (1984).
// Returns {L, M}.
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> GetPencil(
      const Eigen::Ref<const Eigen::MatrixXd>& A,
      const Eigen::Ref<const Eigen::MatrixXd>& B,
      const Eigen::Ref<const Eigen::MatrixXd>& Q,
      const Eigen::Ref<const Eigen::MatrixXd>& R);
}  // namespace internal

// Solve the discrete-time algebraic Riccati equation (DARE) using the approach by Van Dooren
// (1981). The implementation largely follows Arnold and Laub (1984). Assumes E = I and S = 0.
// Van Dooren, P. M. A generalized eigenvalue approach for solving Riccati equations. SIAM Journal
//   on Scientific and Statistical Computing, 2(2):121–135, 1981. https://doi.org/10.1137/0902010
// Arnold, N. III, Laub, A. J. Generalized eigenproblem algorithms and software for algebraic
//   Riccati equations. Proceedings of the IEEE, 72(12):1746–1754, 1984.
//   https://doi.org/10.1109/PROC.1984.13083
absl::StatusOr<Eigen::MatrixXd> SolveDare(
      const Eigen::Ref<const Eigen::MatrixXd>& A,
      const Eigen::Ref<const Eigen::MatrixXd>& B,
      const Eigen::Ref<const Eigen::MatrixXd>& Q,
      const Eigen::Ref<const Eigen::MatrixXd>& R);
}  // namespace van_dooren

namespace riccati {

enum class Solver { Laub, VanDooren };

absl::StatusOr<Eigen::MatrixXd> SolveDiscrete(
      const Eigen::Ref<const Eigen::MatrixXd>& A,
      const Eigen::Ref<const Eigen::MatrixXd>& B,
      const Eigen::Ref<const Eigen::MatrixXd>& Q,
      const Eigen::Ref<const Eigen::MatrixXd>& R,
      Solver method = Solver::VanDooren);

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
