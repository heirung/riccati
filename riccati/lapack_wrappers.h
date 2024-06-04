#pragma once

#include "Eigen/Dense"
#include "absl/status/statusor.h"
#include "lapack_compat.h"

// This header declares wrappers for some LAPACK routines, with some helper functions. The wrapper
// descriptions are adapted from the original LAPACK documentation.

namespace lapack {
// Non-internal enum for the eigenvalue selectors.
enum class SelectorName {
    none,
    insideUnitCircle,
    outsideUnitCircle,
    leftHalfPlane,
    rightHalfPlane
};

namespace internal {
using schurSelectorFunc = __LAPACK_dgees_func_ptr;
using qzSelectorFunc = __LAPACK_dgges_func_ptr;
// Constant specifically for use when checking whether an eigenvalue parameterized as
// (ALPHAR + i * ALPHAI) / BETA has a denominator that is zero. (See the LAPACK docs for `dgges` --
// "there is a reasonable interpretation for beta=0".) Some LAPACK distributions return a `BETA` for
// which `BETA == 0.0` is `true` in this case, and others do not. In these cases, it does appear
// to be consistent across platforms that dividing a non-zero (or zero) numerator by a `BETA` meant
// to represent zero results in an `inf` (or `nan`) eigenvalue. The multiplier `100.0` seems like a
// reasonable choice without doing extensive cross-platform testing. (On some LAPACK distributions
// where a `BETA` meant to represent zero and where `BETA == 0.0` is `false`, the test below returns
// `true` for a multiplier as small as `1.0`, while on others it returns `false`.)
static constexpr double ZERO_TOLERANCE = std::numeric_limits<double>::epsilon() * 100.0;

// Check whether an eigenvalue denominator `BETA` is zero; cf. the description of `ZERO_TOLERANCE`.
bool IsZero(double value);

// Compute eigenvalues. Note that `BETA` may contain zeros, which leads to `inf`s or `nan`s in the
// returned vector of eigenvalues.
Eigen::VectorXcd GetEigenvalues(
      const Eigen::Ref<const Eigen::VectorXd>& ALPHAR,
      const Eigen::Ref<const Eigen::VectorXd>& ALPHAI,
      const Eigen::Ref<const Eigen::VectorXd>& BETA);

namespace schur {
// Simple functions to select eigenvalues that satisfy a specific criterion. Uses non-const
// parameters to match the LAPACK interface. The eigenvalues are of the form eigRe + i * eigIm.
__LAPACK_bool InsideUnitCircle(double* eigRe, double* eigIm);
__LAPACK_bool OutsideUnitCircle(double* eigRe, double* eigIm);
__LAPACK_bool LeftHalfPlane(double* eigRe, [[maybe_unused]] double* eigIm);
__LAPACK_bool RightHalfPlane(double* eigRe, double* eigIm);
}  // namespace schur

namespace qz {
// Simple functions to select eigenvalues that satisfy a specific criterion. Uses non-const
// parameters to match the LAPACK interface. The eigenvalues are of the form
// (ALPHAR + i * ALPHAI) / BETA. If `BETA` is zero, `sign(ALPHAR)` determines left vs right
// half-plane.
__LAPACK_bool InsideUnitCircle(double* ALPHAR, double* ALPHAI, double* BETA);
__LAPACK_bool OutsideUnitCircle(double* ALPHAR, double* ALPHAI, double* BETA);
__LAPACK_bool LeftHalfPlane(double* ALPHAR, [[maybe_unused]] double* ALPHAI, double* BETA);
__LAPACK_bool RightHalfPlane(double* ALPHAR, [[maybe_unused]] double* ALPHAI, double* BETA);
}  // namespace qz

namespace order_qz {
// Generate a vector of boolean values to select among already computed eigenvalues (parameterized
// as (ALPHAR(j) + ALPHAI(j)) / BETA(j)). For use with `dtgsen`.
std::vector<__LAPACK_bool> EigenvaluesToSelect(
      const Eigen::Ref<const Eigen::VectorXd>& alphar,
      const Eigen::Ref<const Eigen::VectorXd>& alphai,
      const Eigen::Ref<const Eigen::VectorXd>& beta,
      const SelectorName selectorName);
}  // namespace order_qz
}  // namespace internal

// Compute for a N x N real nonsymmetric matrix A the real Schur matrix, the Schur vectors, and the
// eigenvalues (sorted, optionally). (Wrapper for `DGEES` -- Double precision, GEneral matrix,
// Eigenvalues, Schur.) The real Schur decomposition for A is A = ZTZᵀ, where
//   - T is the real Schur form (upper quasi-triangular, with 1x1 and 2x2 blocks), and
//   - Z is an orthogonal matrix containing the Schur vectors (with Zᵀ = Z⁻¹).
// Returns {T, Z, eigenvalues}.
absl::StatusOr<std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXcd>>
Schur(const Eigen::Ref<const Eigen::MatrixXd>& A, const SelectorName selector = SelectorName::none);

// Compute for a pair of N x N real nonsymmetric matrices (A, B) the left and right matrices of
// Schur vectors (Q, Z), the generalized real Schur form (S, T), and the generalized eigenvalues
// (sorted, optionally). (Wrapper for `DGGES` -- Double precision, Generalized matrices and
// Generalized problem, Eigenvalues, Schur.) The generalized Schur factorization is
//   (A, B) = (QSZᵀ, QTZᵀ).
// The pair (S, T) is in generalized real Schur form, meaning T is upper triangular and S upper
// quasi-triangular, with 1x1 and 2x2 blocks. Each eigenvalue j is represented by (ALPHAR(j) +
// ALPHAI(j)) / BETA(j).
// Returns {S, T, Q, Z, ALPHAR, ALPHAI, BETA}.
absl::StatusOr<std::tuple<
      Eigen::MatrixXd,   // S
      Eigen::MatrixXd,   // T
      Eigen::MatrixXd,   // Q
      Eigen::MatrixXd,   // Z
      Eigen::VectorXd,   // ALPHAR
      Eigen::VectorXd,   // ALPHAI
      Eigen::VectorXd>>  // BETA
Qz(const Eigen::Ref<const Eigen::MatrixXd>& A,
   const Eigen::Ref<const Eigen::MatrixXd>& B,
   const SelectorName selector = SelectorName::none);

// Reorder the generalized real Schur decomposition of a real matrix pair (A, B) in terms of an
// orthonormal equivalence transformation Qᵀ(A, B)Z), so that a selected cluster of
// eigenvalues appears in the leading diagonal blocks of the upper quasi-triangular matrix A and
// the upper triangular matrix B. The leading columns of Q and Z form orthonormal bases of the
// corresponding left and right eigenspaces (deflating subspaces). (A, B) must be in generalized
// real Schur canonical form (as returned by `DGGES`), i.e., A is block upper triangular with 1-by-1
// and 2-by-2 diagonal blocks and B is upper triangular (not checked). Each eigenvalue j is
// represented by (ALPHAR(j) + ALPHAI(j)) / BETA(j), and the reordered eigenvalues correspond to the
// reordered pair (A', B'). (Wrapper for `DTGSEN` -- Double precision, Triangular matrices,
// Generalized problem, and...?).
// As a more verbose description, consider obtaining S, T, Q, Z by calling the `DGEES` wrapper
// `qz(A, B)`. These matrices satisfy
//   (A, B) = (QSZᵀ, QTZᵀ).
// The pair (S, T) is in generalized real Schur form. Calling `reorderQz(S, T, Q, Z, SELECT)` then
// provides the reordered matrices S', T', Q', Z', and the reordered eigenvalues ALPHAR', ALPHAI',
// BETA'. The reordered matrices correspond the ordering in `SELECT`. Internally, `DTGSEN` computes
// orthogonal U and W such that
//   (S', T') = (UᵀSW, UᵀTW)
// and the selected eigenvalues appear in the top left corners of (S', T'). By orthogonality we have
// UUᵀ = I and WWᵀ = I, and we can rewrite the generalized real Schur form as
//   (A, B) = Q (S, T) Zᵀ
//          = Q UUᵀ (S, T) WWᵀ Zᵀ
//          = (QU) (UᵀSW, UᵀTW) (ZW)ᵀ,  Q' = QU, Z' = ZW
//          = (Q') (S', T') (Z')ᵀ
// Returns {S', T', Z', Q', ALPHAR', ALPHAI', BETA'}.
absl::StatusOr<std::tuple<
      Eigen::MatrixXd,   // S' -- reordered version of the first input (A or S)
      Eigen::MatrixXd,   // T' -- reordered version of the second input (B or T)
      Eigen::MatrixXd,   // Q' -- reordered version of the third input (Q)
      Eigen::MatrixXd,   // Z' -- reordered version of the fourth input (Z)
      Eigen::VectorXd,   // ALPHAR' -- real part of the numerator of the reordered eigenvalues
      Eigen::VectorXd,   // ALPHAI' -- imaginary part of the numerator of the reordered eigenvalues
      Eigen::VectorXd>>  // BETA' -- denominator of the reordered eigenvalues
ReorderQz(
      const Eigen::Ref<const Eigen::MatrixXd>& A,  // A per LAPACK docs, S in the sense of A = QSZᵀ
      const Eigen::Ref<const Eigen::MatrixXd>& B,  // B per LAPACK docs, T in the sense of B = QTZᵀ
      const Eigen::Ref<const Eigen::MatrixXd>& Q,
      const Eigen::Ref<const Eigen::MatrixXd>& Z,
      const std::vector<__LAPACK_bool>& SELECT);

}  // namespace lapack
