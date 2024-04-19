#include "Eigen/Dense"
#include "absl/status/statusor.h"
#include "gtest/gtest.h"
#include "riccati/riccati.h"

namespace {
constexpr double DOUBLE_TOLERANCE = 1.0e-6;

double MaxAbsElem(const Eigen::Ref<const Eigen::MatrixXd>& m) { return m.array().abs().maxCoeff(); }
}  // namespace

TEST(TestSchurLaub, TestGetZBasic) {
    constexpr double a = 3.0;
    constexpr double b = 5.0;
    constexpr double q = 7.0;
    constexpr double r = 8.0;
    const Eigen::Matrix<double, 1, 1> A{{a}};
    const Eigen::Matrix<double, 1, 1> B{{b}};
    const Eigen::Matrix<double, 1, 1> Q{{q}};
    const Eigen::Matrix<double, 1, 1> R{{r}};
    const Eigen::Matrix<double, 2, 2> ZExpected{
          {a + q * b * b / (r * a), -b * b / (r * a)},
          {                 -q / a,          1.0 / a}
    };
    const absl::StatusOr<Eigen::MatrixXd> maybeZ = laub::internal::GetZ(A, B, Q, R);
    ASSERT_TRUE(maybeZ.ok());
    EXPECT_TRUE(maybeZ.value().isApprox(ZExpected, DOUBLE_TOLERANCE));
}

TEST(TestSchurLaub, TestGetZ3x3) {
    const Eigen::MatrixXd A{
          {0.9,   0.0,  0.0},
          {0.0,   0.0,  1.0},
          {0.1, -0.79, 1.78}
    };
    const Eigen::VectorXd B{
          {1.0, 0.0, 0.1}
    };
    Eigen::DiagonalMatrix<double, 3> Q;
    Q.diagonal() << 0.0, 0.0, 1.0;
    Eigen::DiagonalMatrix<double, 1> R;
    R.diagonal() << 1.0;
    const absl::StatusOr<Eigen::MatrixXd> maybeZ =
          laub::internal::GetZ(A, B, Q.toDenseMatrix(), R.toDenseMatrix());
    ASSERT_TRUE(maybeZ.ok());
    const Eigen::MatrixXd ZExpected{
          { 0.9,   0.0,  0.0,  -1.111111111,  -0.01406469761, 7.554488001e-17},
          { 0.0,   0.0,  1.0,          -0.0,             0.0,             0.0},
          { 0.1, -0.79, 1.78, -0.1111111111, -0.001406469761,  1.71147822e-17},
          { 0.0,   0.0,  0.0,   1.111111111,    0.1406469761,             0.0},
          {-0.0,  -0.0, -1.0,           0.0,     2.253164557,             1.0},
          { 0.0,   0.0,  0.0,           0.0,    -1.265822785,             0.0}
    };
    EXPECT_TRUE(maybeZ.value().isApprox(ZExpected, DOUBLE_TOLERANCE));
}

TEST(TestSchurLaub, TestGetZ2x2) {
    const Eigen::MatrixXd A{
          {0.5, -0.2},
          {0.3,  0.7}
    };
    const Eigen::MatrixXd B{
          {1.3, 0.1},
          {0.2, 1.1}
    };
    Eigen::DiagonalMatrix<double, 2> Q;
    Q.diagonal() << 2.0, 4.0;
    Eigen::DiagonalMatrix<double, 2> R;
    R.diagonal() << 3.0, 9.0;
    const absl::StatusOr<Eigen::MatrixXd> maybeZ =
          laub::internal::GetZ(A, B, Q.toDenseMatrix(), R.toDenseMatrix());
    const Eigen::MatrixXd ZExpected{
          {  2.523848, -1.369648,  -1.011924,  0.2924119},
          { 0.7818428,  1.131436, -0.2409214, -0.1078591},
          { -3.414634,  2.926829,   1.707317, -0.7317073},
          {-0.9756098, -4.878049,  0.4878049,   1.219512}
    };
    EXPECT_TRUE(maybeZ.value().isApprox(ZExpected, DOUBLE_TOLERANCE));
}

TEST(TestSchurLaub, TestGetSchurVectors) {
    const Eigen::MatrixXd Z{
          {  2.523848, -1.369648,  -1.011924,  0.2924119},
          { 0.7818428,  1.131436, -0.2409214, -0.1078591},
          { -3.414634,  2.926829,   1.707317, -0.7317073},
          {-0.9756098, -4.878049,  0.4878049,   1.219512}
    };
    const absl::StatusOr<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> maybeSchurVectors =
          laub::internal::GetSchurVectors(Z);
    ASSERT_TRUE(maybeSchurVectors.ok());
    const auto [U_11, U_21] = maybeSchurVectors.value();
    const Eigen::MatrixXd U_11Expected{
          { 0.1811629,  0.3348793},
          {-0.1756442, 0.04315428}
    };
    const Eigen::MatrixXd U_21Expected{
          { 0.3636182, 0.8501869},
          {-0.8967223, 0.4039504}
    };
    EXPECT_TRUE(U_11.isApprox(U_11Expected, DOUBLE_TOLERANCE));
    EXPECT_TRUE(U_21.isApprox(U_21Expected, DOUBLE_TOLERANCE));
}

TEST(TestSchurLaub, TestSolveDareBasic) {
    const Eigen::MatrixXd A{
          {0.4, -0.1},
          {0.5,  0.8}
    };
    const Eigen::MatrixXd B{
          {1.3, 0.1},
          {0.2, 1.1}
    };
    Eigen::DiagonalMatrix<double, 2> Q;
    Q.diagonal() << 2.0, 4.0;
    Eigen::DiagonalMatrix<double, 2> R;
    R.diagonal() << 3.0, 9.0;

    const absl::StatusOr<Eigen::MatrixXd> maybeX =
          riccati::SolveDiscrete(A, B, Q.toDenseMatrix(), R.toDenseMatrix(), riccati::Solver::Laub);
    ASSERT_TRUE(maybeX.ok());
    const Eigen::MatrixXd XExpected{
          {2.897611, 1.152146},
          {1.152146, 5.974003}
    };
    EXPECT_TRUE(maybeX.value().isApprox(XExpected, DOUBLE_TOLERANCE));

    const Eigen::MatrixXd residual =
          riccati::Residual(A, B, Q.toDenseMatrix(), R.toDenseMatrix(), maybeX.value());
    EXPECT_LT(MaxAbsElem(residual), 1.4e-14);
}
