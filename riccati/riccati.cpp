#include "riccati.h"

#include "Eigen/Dense"
#include "absl/status/statusor.h"
#include "lapack_wrappers.h"

namespace internal {
absl::StatusOr<Eigen::MatrixXd> SolveRiccatiFromSubspaces(
      const Eigen::Ref<const Eigen::MatrixXd>& U_11,
      const Eigen::Ref<const Eigen::MatrixXd>& U_21) {
    // Solve the discrete-time algebraic Riccati equation (DARE) using the subspaces U₁₁ and U₂₁.
    // X = U₂₁U₁₁⁻¹, so U₁₁ᵀX = U₂₁ᵀ (X is symmetric), or X = U₁₁⁻ᵀU₂₁ᵀ:
    Eigen::FullPivLU<Eigen::MatrixXd> lu(U_11.transpose());
    if (!lu.isInvertible()) {
        return absl::InternalError(std::string(__func__) + "U_11 is not invertible");
    }
    return lu.solve(U_21.transpose());
}
}  // namespace internal

namespace laub {
namespace internal {
absl::StatusOr<Eigen::MatrixXd>
GetZ(const Eigen::Ref<const Eigen::MatrixXd>& A,
     const Eigen::Ref<const Eigen::MatrixXd>& B,
     const Eigen::Ref<const Eigen::MatrixXd>& Q,
     const Eigen::Ref<const Eigen::MatrixXd>& R) {
    const int n = A.cols();  // x in R^n
    const int m = B.cols();  // u in R^m
    // Avoid computing these quantities more than once:
    const Eigen::FullPivLU<Eigen::MatrixXd> luA(A);
    const Eigen::FullPivLU<Eigen::MatrixXd> luR(R);
    if (luA.rank() < n || luR.rank() < m) {
        return absl::InvalidArgumentError(std::string(__func__) + ": A or R is singular.");
    }
    const Eigen::MatrixXd G = B * (luR.solve(B.transpose()));  // G = BR⁻¹Bᵀ
    // Construct the symplectic matrix Z =
    //   [ A + GA⁻ᵀQ, -GA⁻ᵀ ]
    //   [     -A⁻ᵀQ,   A⁻ᵀ ]
    // For GA⁻ᵀ, let Mᵀ = GA⁻ᵀ, so that M = A⁻¹Gᵀ, or AM = Gᵀ. Solve AM = Gᵀ for M, then transpose.
    Eigen::MatrixXd Z = Eigen::MatrixXd::Zero(2 * n, 2 * n);
    Z.block(0, 0, n, n) = A + G * (luA.transpose().solve(Q)).eval();  // A + GA⁻ᵀQ
    Z.block(0, n, n, n) = -(luA.solve(G.transpose())).transpose();    // -GA⁻ᵀ
    Z.block(n, 0, n, n) = -(luA.transpose().solve(Q)).eval();         // -A⁻ᵀQ
    Z.block(n, n, n, n) = luA.inverse().transpose();                  // A⁻ᵀ
    return Z;
}

absl::StatusOr<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>>
GetSchurVectors(const Eigen::Ref<const Eigen::MatrixXd>& Z) {
    // Determine U in the transformation UᵀZU = S, where S is upper quasi-triangular,
    //   S = [S₁₁, S₁₂], and U = [U₁₁, U₁₂]
    //       [  0, S₂₂]          [U₂₁, U₂₂]
    // The eigenvalues of the blocks S₁₁ and S₂₂ are stable and unstable, respectively. The n
    // leading columns of the matrix U are the Schur vectors that span the stable invariant
    // subspace. Return the blocks U₁₁ and U₂₁, which are sufficient to solve the Riccati equation.
    const auto maybeSchur = lapack::Schur(Z, lapack::SelectorName::insideUnitCircle);
    if (!maybeSchur.ok()) {
        return maybeSchur.status();
    }
    const auto [Z_, U, eig_] = maybeSchur.value();
    const int n = Z.rows() / 2;  // Z is 2n x 2n
    const Eigen::MatrixXd U_11 = U.topLeftCorner(n, n);
    const Eigen::MatrixXd U_21 = U.bottomLeftCorner(n, n);
    return std::make_pair(U_11, U_21);
}
}  // namespace internal

absl::StatusOr<Eigen::MatrixXd> SolveDare(
      const Eigen::Ref<const Eigen::MatrixXd>& A,
      const Eigen::Ref<const Eigen::MatrixXd>& B,
      const Eigen::Ref<const Eigen::MatrixXd>& Q,
      const Eigen::Ref<const Eigen::MatrixXd>& R) {
    const int n = B.rows();  // x in R^n
    const int m = B.cols();  // u in R^m
    if (A.cols() != A.rows() || A.rows() != n || Q.rows() != n || Q.cols() != n || R.rows() != m ||
        R.cols() != m) {
        return absl::InvalidArgumentError(
              std::string(__func__) + ": at least one of A, B, Q, and R has incorrect dimensions.");
    }
    const absl::StatusOr<Eigen::MatrixXd> Z = internal::GetZ(A, B, Q, R);
    if (!Z.ok()) {
        return Z.status();
    }
    const absl::StatusOr<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> maybeSchurVectors =
          internal::GetSchurVectors(Z.value());
    if (!maybeSchurVectors.ok()) {
        return maybeSchurVectors.status();
    }
    const auto [U_11, U_21] = maybeSchurVectors.value();
    // X = U₂₁U₁₁⁻¹, so U₁₁ᵀX = U₂₁ᵀ (X is symmetric), or X = U₁₁⁻ᵀU₂₁ᵀ:
    const absl::StatusOr<Eigen::MatrixXd> maybeX =
          ::internal::SolveRiccatiFromSubspaces(U_11, U_21);
    if (!maybeX.ok()) {
        return maybeX.status();
    }
    const Eigen::MatrixXd X = maybeX.value();
    return (X + X.transpose()) / 2.0;
}
}  // namespace laub

namespace van_dooren {
namespace internal {
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> GetPencil(
      const Eigen::Ref<const Eigen::MatrixXd>& A,
      const Eigen::Ref<const Eigen::MatrixXd>& B,
      const Eigen::Ref<const Eigen::MatrixXd>& Q,
      const Eigen::Ref<const Eigen::MatrixXd>& R) {
    const int n = A.cols();  // x in R^n
    const int m = B.cols();  // u in R^m
    const Eigen::MatrixXd E = Eigen::MatrixXd::Identity(n, n);
    const int matrixDim = 2 * n + m;
    // Construct the pencil 𝜆L - M.
    Eigen::MatrixXd L = Eigen::MatrixXd::Zero(matrixDim, matrixDim);
    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(matrixDim, matrixDim);
    //     [E,  0,  0]
    // L = [0,  Aᵀ, 0]
    //     [0, -Bᵀ, 0]
    L.block(0, 0, n, n) = E;
    L.block(n, n, n, n) = A.transpose();
    L.block(2 * n, n, m, n) = -B.transpose();
    //     [ A, 0,  B]
    // M = [-Q, Eᵀ, 0]
    //     [ 0, 0,  R]
    M.block(0, 0, n, n) = A;
    M.block(0, 2 * n, n, m) = B;
    M.block(n, 0, n, n) = -Q;
    M.block(n, n, n, n) = E.transpose();
    M.block(2 * n, 2 * n, m, m) = R;
    return {L, M};
}
}  // namespace internal

absl::StatusOr<Eigen::MatrixXd> SolveDare(
      const Eigen::Ref<const Eigen::MatrixXd>& A,
      const Eigen::Ref<const Eigen::MatrixXd>& B,
      const Eigen::Ref<const Eigen::MatrixXd>& Q,
      const Eigen::Ref<const Eigen::MatrixXd>& R) {
    const int n = B.rows();  // x in R^n
    const int m = B.cols();  // u in R^m
    if (A.cols() != A.rows() || A.rows() != n || Q.rows() != n || Q.cols() != n || R.rows() != m ||
        R.cols() != m) {
        return absl::InvalidArgumentError(
              std::string(__func__) + ": at least one of A, B, Q, and R has incorrect dimensions.");
    }
    const auto [L, M] = internal::GetPencil(A, B, Q, R);
    // Following Arnold and Laub (1984) Eq. (11): "Determine an orthogonal matrix P
    // ((2n + m) x (2n + m)) such that
    //            [B]   [0]
    // [P₁₁, P₁₂] [0] = [0]
    // [P₁₂, P₂₂] [R]   [R_bar]
    // where R_bar is m x m and nonsingular. P can be found from a series of Householder
    // transformations. The matrix P is then used to deflate the pencil." (Deflate out the infinite
    // generalized eigenvalues.) This corresponds to finding U in Eq. (55) in Laub (1979), which
    // reduces the rightmost block column (of width m) in Eq. (53) by zeroing out the top block (of
    // height 2n).
    const Eigen::HouseholderQR<Eigen::MatrixXd> qr(M.rightCols(m));
    const Eigen::MatrixXd P =
          qr.householderQ().transpose();  // "the m rightmost columns of M" = Q*R, P = Qᵀ
    // Rather than using a permutation matrix to rearrange the transformation to the form above,
    // just index appropriately: (accordingly, P.bottomRows(2 * n) * M.rightCols(m) is all zeroes.)
    const Eigen::MatrixXd LDeflated = P.bottomRows(2 * n) * L.leftCols(2 * n);  // L tilde
    const Eigen::MatrixXd MDeflated = P.bottomRows(2 * n) * M.leftCols(2 * n);  // M tilde

    const auto maybeQz = lapack::Qz(MDeflated, LDeflated, lapack::SelectorName::insideUnitCircle);
    if (!maybeQz.ok()) {
        return maybeQz.status();
    }
    const auto [ordS, ordT, ordQ, ordZ, ordAlphaRe, ordAlphaIm, ordBeta] = maybeQz.value();

    const Eigen::MatrixXd U_11 = ordZ.topLeftCorner(n, n);
    const Eigen::MatrixXd U_21 = ordZ.bottomLeftCorner(n, n);
    // X = U₂₁U₁₁⁻¹, so U₁₁ᵀX = U₂₁ᵀ (X is symmetric), or X = U₁₁⁻ᵀU₂₁ᵀ:
    const absl::StatusOr<Eigen::MatrixXd> maybeX =
          ::internal::SolveRiccatiFromSubspaces(U_11, U_21);
    if (!maybeX.ok()) {
        return maybeX.status();
    }
    const Eigen::MatrixXd X = maybeX.value();
    return (X + X.transpose()) / 2.0;
}
}  // namespace van_dooren

namespace riccati {

absl::StatusOr<Eigen::MatrixXd> SolveDiscrete(
      const Eigen::Ref<const Eigen::MatrixXd>& A,
      const Eigen::Ref<const Eigen::MatrixXd>& B,
      const Eigen::Ref<const Eigen::MatrixXd>& Q,
      const Eigen::Ref<const Eigen::MatrixXd>& R,
      Solver method) {
    switch (method) {
        case Solver::Laub:
            return laub::SolveDare(A, B, Q, R);
        case Solver::VanDooren:
            return van_dooren::SolveDare(A, B, Q, R);
        default:
            return absl::InvalidArgumentError("Unknown method argument.");
    }
}

// Compute the residual of the discrete-time algebraic Riccati equation (DARE).
Eigen::MatrixXd Residual(
      const Eigen::Ref<const Eigen::MatrixXd>& A,
      const Eigen::Ref<const Eigen::MatrixXd>& B,
      const Eigen::Ref<const Eigen::MatrixXd>& Q,
      const Eigen::Ref<const Eigen::MatrixXd>& R,
      const Eigen::Ref<const Eigen::MatrixXd>& X,
      bool alternativeForm) {
    if (!alternativeForm) {
        // Classic form:
        //   AᵀXA − AᵀXB(R + BᵀXB)⁻¹BᵀXA + Q - X = 0
        const Eigen::MatrixXd ATXB = A.transpose() * X * B;
        const Eigen::MatrixXd R_plus_BTXB = R + B.transpose() * X * B;
        return A.transpose() * X * A + Q - X -
              ATXB * (R_plus_BTXB.fullPivLu().solve(ATXB.transpose()));
    } else {
        // Alternative form (NAN if R is singular):
        //   AᵀX(I + BR⁻¹BᵀX)⁻¹A + Q - X = 0
        const int n = A.rows();
        const Eigen::MatrixXd I_plus_BRinvBTX =
              Eigen::MatrixXd::Identity(n, n) + B * (R.fullPivLu().solve(B.transpose())) * X;
        return A.transpose() * X * (I_plus_BRinvBTX.fullPivLu().solve(A)) + Q - X;
    }
}
}  // namespace riccati
