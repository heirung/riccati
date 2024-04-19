#pragma once

#include "Eigen/Dense"
#include "absl/status/statusor.h"
#include "lapack_compat.h"

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

namespace schur {
// Simple functions to select eigenvalues that satisfy a specific criterion. Uses non-const
// parameters to match the LAPACK interface. The eigenvalues are of the form eigRe + i * eigIm.
__LAPACK_bool InsideUnitCircle(double* eigRe, double* eigIm);
__LAPACK_bool OutsideUnitCircle(double* eigRe, double* eigIm);
__LAPACK_bool LeftHalfPlane(double* eigRe, [[maybe_unused]] double* eigIm);
__LAPACK_bool RightHalfPlane(double* eigRe, double* eigIm);
}  // namespace schur
}  // namespace internal

// Compute for a N x N real nonsymmetric matrix A the real Schur matrix, the Schur vectors, and the
// eigenvalues (sorted, optionally). (Wrapper for `DGEES` -- Double precision, GEneral matrix,
// Eigenvalues, Schur.) The real Schur decomposition for A is A = ZTZᵀ, where
//   - T is the real Schur form (upper quasi-triangular, with 1x1 and 2x2 blocks), and
//   - Z is an orthogonal matrix containing the Schur vectors (with Zᵀ = Z⁻¹).
// Returns {T, Z, eigenvalues}.
absl::StatusOr<std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXcd>>
Schur(const Eigen::Ref<const Eigen::MatrixXd>& A, const SelectorName selector = SelectorName::none);
}  // namespace lapack
