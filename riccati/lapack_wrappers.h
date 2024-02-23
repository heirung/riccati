#pragma once

#include <Accelerate/Accelerate.h>  // Includes LAPACK interfaces

#include "Eigen/Dense"
#include "absl/status/statusor.h"

namespace lapack {
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

// `dgees` wrapper
absl::StatusOr<std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXcd>>
Schur(const Eigen::Ref<const Eigen::MatrixXd>& A, const SelectorName selector = SelectorName::none);
}  // namespace lapack
