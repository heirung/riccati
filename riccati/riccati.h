#pragma once

#include "Eigen/Dense"
#include "absl/status/statusor.h"

namespace laub {
namespace internal {

absl::StatusOr<Eigen::MatrixXd>
GetZ(const Eigen::Ref<const Eigen::MatrixXd>& A,
     const Eigen::Ref<const Eigen::MatrixXd>& B,
     const Eigen::Ref<const Eigen::MatrixXd>& Q,
     const Eigen::Ref<const Eigen::MatrixXd>& R);

absl::StatusOr<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>>
GetSchurVectors(const Eigen::Ref<const Eigen::MatrixXd>& Z);
}  // namespace internal

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

Eigen::MatrixXd Residual(
      const Eigen::Ref<const Eigen::MatrixXd>& A,
      const Eigen::Ref<const Eigen::MatrixXd>& B,
      const Eigen::Ref<const Eigen::MatrixXd>& Q,
      const Eigen::Ref<const Eigen::MatrixXd>& R,
      const Eigen::Ref<const Eigen::MatrixXd>& X,
      bool alternativeForm = false);
}  // namespace riccati
