#include <ostream>

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

TEST(TestVanDooren, TestGetPencilBasic) {
    const Eigen::MatrixXd A{
          {2.0, 3.0,  4.0},
          {5.0, 6.0,  7.0},
          {8.0, 9.0, 10.0}
    };
    const Eigen::MatrixXd B{
          {15.0, 27.0},
          {35.0, 47.0},
          {55.0, 67.0}
    };
    const Eigen::MatrixXd Q{
          {1.0, 2.0, 3.0},
          {4.0, 5.0, 6.0},
          {7.0, 8.0, 9.0}
    };
    const Eigen::MatrixXd R{
          {10.0, 11.0},
          {12.0, 13.0}
    };
    const auto [L, M] = van_dooren::internal::GetPencil(A, B, Q, R);
    const int n = B.rows();
    const int m = B.cols();
    const Eigen::MatrixXd E = Eigen::MatrixXd::Identity(n, n);
    const int N = 2 * n + m;

    // The pencil is 𝜆L - M; L and M are both N x N.
    Eigen::MatrixXd LExpected = Eigen::MatrixXd::Zero(N, N);
    LExpected.block(0, 0, n, n) = E;
    LExpected.block(n, n, n, n) = A.transpose();
    LExpected.block(2 * n, n, m, n) = -B.transpose();
    EXPECT_TRUE(L.isApprox(LExpected, 1.0e-30));
    Eigen::MatrixXd MExpected = Eigen::MatrixXd::Zero(N, N);
    MExpected.block(0, 0, n, n) = A;
    MExpected.block(0, 2 * n, n, m) = B;
    MExpected.block(n, 0, n, n) = -Q;
    MExpected.block(n, n, n, n) = E.transpose();
    MExpected.block(2 * n, 2 * n, m, m) = R;
    EXPECT_TRUE(M.isApprox(MExpected, 1.0e-30));
}

TEST(TestVanDooren, TestBasic2x2) {
    const Eigen::MatrixXd A{
          {0.9512, 0.0000},
          {0.0000, 0.9048}
    };
    const Eigen::MatrixXd B{
          { 4.8770, 4.8770},
          {-1.1895, 3.5690}
    };
    const Eigen::MatrixXd Q{
          {0.0050, 0.0000},
          {0.0000, 0.0200}
    };
    const Eigen::MatrixXd R{
          {0.3333, 0.0000},
          {0.0000, 3.0000}
    };
    const absl::StatusOr<Eigen::MatrixXd> maybeX =
          riccati::SolveDiscrete(A, B, Q, R, riccati::Solver::VanDooren);
    ASSERT_TRUE(maybeX.ok());
    Eigen::MatrixXd XExpected{
          { 0.01045886, 0.003224873},
          {0.003224873,  0.05039751}
    };
    EXPECT_TRUE(maybeX.value().isApprox(XExpected, DOUBLE_TOLERANCE));
}

struct RiccatiTestCase {
    std::string name;
    Eigen::MatrixXd A;
    Eigen::MatrixXd B;
    Eigen::MatrixXd Q;
    Eigen::MatrixXd R;
    std::optional<Eigen::MatrixXd> analyticalX = std::nullopt;
    bool failsInLaub = false;

    struct ErrorTolerance {
        double laub = 1.0e-14;
        double vanDooren = 1.0e-14;
    } errorTolerance;
};

class TestVanDoorenRiccati : public ::testing::TestWithParam<RiccatiTestCase> {};

TEST_P(TestVanDoorenRiccati, TestBennerCollection) {
    constexpr double ANALYTICAL_TOLERANCE = 1.0e-9;
    RiccatiTestCase testCase = GetParam();

    const absl::StatusOr<Eigen::MatrixXd> maybeXVanDooren = riccati::SolveDiscrete(
          testCase.A, testCase.B, testCase.Q, testCase.R, riccati::Solver::VanDooren);
    ASSERT_TRUE(maybeXVanDooren.ok());
    const Eigen::MatrixXd residualVanDooren = riccati::Residual(
          testCase.A, testCase.B, testCase.Q, testCase.R, maybeXVanDooren.value());
    EXPECT_LT(MaxAbsElem(residualVanDooren), testCase.errorTolerance.vanDooren);
    if (testCase.analyticalX.has_value()) {
        EXPECT_TRUE(
              maybeXVanDooren.value().isApprox(testCase.analyticalX.value(), ANALYTICAL_TOLERANCE));
    }

    const absl::StatusOr<Eigen::MatrixXd> maybeXLaub = riccati::SolveDiscrete(
          testCase.A, testCase.B, testCase.Q, testCase.R, riccati::Solver::Laub);
    if (testCase.failsInLaub) {
        EXPECT_FALSE(maybeXLaub.ok());
        EXPECT_EQ(maybeXLaub.status().code(), absl::StatusCode::kInvalidArgument);
    } else {
        ASSERT_TRUE(maybeXLaub.ok());
        const Eigen::MatrixXd residualLaub =
              riccati::Residual(testCase.A, testCase.B, testCase.Q, testCase.R, maybeXLaub.value());
        EXPECT_LT(MaxAbsElem(residualLaub), testCase.errorTolerance.laub);
        if (testCase.analyticalX.has_value()) {
            EXPECT_TRUE(
                  maybeXLaub.value().isApprox(testCase.analyticalX.value(), ANALYTICAL_TOLERANCE));
        }
        // Ensure that the two approaches give similar results:
        EXPECT_TRUE(maybeXLaub.value().isApprox(maybeXVanDooren.value(), DOUBLE_TOLERANCE));
    }
}

// After Benner, P., Laub,  A. J., and Mehrmann. V. A collection of benchmark examples for the
// numerical solution of algebraic Riccati equations II: discrete-time case. Technical report SPC
// 95_23, Technische Universität Chemnitz-Zwickau, 1995.
// https://www.tu-chemnitz.de/sfb393/Files/PDF/spc95-23.pdf
std::vector<RiccatiTestCase> GenParams() {
    const double epsilon = 100.0;  // For Example 12
    const int N = 30;              // For Example 15
    return {
          {.name = "2x_1u",  // Example 1
           .A = Eigen::Matrix2d{{4.0, 3.0}, {-9.0 / 2.0, -7.0 / 2.0}},
           .B = Eigen::Vector2d{{1.0}, {-1.0}},
           .Q = Eigen::Matrix2d{{9.0, 6.0}, {6.0, 4.0}},
           .R = Eigen::Matrix<double, 1, 1>{{1.0}},
           .analyticalX = ((1.0 + std::sqrt(5.0)) / 2.0) * Eigen::Matrix2d{{9.0, 6.0}, {6.0, 4.0}},
           .errorTolerance = {.laub = 2.6e-13, .vanDooren = 5.2e-14}},
          {.name = "2x_2u",  // Example 2
           .A = Eigen::Matrix2d{{0.9512, 0.0}, {0.0, 0.9048}},
           .B = Eigen::Matrix2d{{4.877, 4.877}, {-1.1895, 3.569}},
           .Q = Eigen::Matrix2d{{0.005, 0.0}, {0.0, 0.02}},
           .R = Eigen::Matrix2d{{1.0 / 3.0, 0.0}, {0.0, 3.0}},
           .errorTolerance = {.laub = 2.5e-16, .vanDooren = 5.4e-17}},
          {.name = "2x_2u_0R",  // Example 3
           .A = Eigen::Matrix2d{{2.0, -1.0}, {1.0, 0.0}},
           .B = Eigen::Vector2d{{1.0}, {0.0}},
           .Q = Eigen::Matrix2d{{0.0, 0.0}, {0.0, 1.0}},
           .R = Eigen::Matrix<double, 1, 1>::Zero(),
           .analyticalX = Eigen::Matrix2d::Identity(),
           .failsInLaub = true,  // R is singular
           .errorTolerance = {.vanDooren = 1.0e-30}},
          {.name = "2x_2u_singularR",  // Example 4 with S omitted
           .A = Eigen::Matrix2d{{0.0, 1.0}, {0.0, -1.0}},
           .B = Eigen::Matrix2d{{1.0, 0.0}, {2.0, 1.0}},
           .Q = Eigen::Matrix2d{{-4.0, -4.0}, {-4.0, 7.0}} * (1.0 / 11.0),
           .R = Eigen::Matrix2d{{9.0, 3.0}, {3.0, 1.0}},
           .failsInLaub = true,  // A is singular
           .errorTolerance = {.vanDooren = 3.6e-15}},
          {.name = "2x_1u_minimal_A",  // Example 5
           .A = Eigen::Matrix2d{{0.0, 1.0}, {0.0, 0.0}},
           .B = Eigen::Vector2d{{0.0}, {1.0}},
           .Q = Eigen::Matrix2d{{1.0, 2.0}, {2.0, 4.0}},
           .R = Eigen::Matrix<double, 1, 1>{{1.0}},
           .failsInLaub = true,  // A is singular
           .errorTolerance = {.vanDooren = 2.5e-15}},
          {.name = "4x_2u_IR",  // Example 6
           .A =
                 Eigen::Matrix4d{
                       {0.998, 0.067, 0.0, 0.0},
                       {-0.067, 0.998, 0.0, 0.0},
                       {0.0, 0.0, 0.998, 0.153},
                       {0.0, 0.0, -0.153, 0.998}},
           .B =
                 Eigen::Matrix<double, 4, 2>{
                       {0.0033, 0.02}, {0.1, -0.0007}, {0.04, 0.0073}, {-0.0028, 0.1}},
           .Q =
                 Eigen::Matrix4d{
                       {1.87, 0.0, 0.0, -0.244},
                       {0.0, 0.744, 0.205, 0.0},
                       {0.0, 0.205, 0.589, 0.0},
                       {-0.244, 0.0, 0.0, 1.048}},
           .R = Eigen::Matrix2d::Identity(),
           .errorTolerance = {.laub = 5.4e-13, .vanDooren = 1.6e-13}},
          {.name = "4x_2u_tinyQ",  // Example 7
           .A = 1.0e-3 *
                 Eigen::Matrix4d{
                       {984.750, -79.903, 0.9054, -1.0765},
                       {41.588, 998.990, -35.855, 12.684},
                       {-546.620, 44.916, -329.910, 193.180},
                       {2662.400, -100.450, -924.550, -263.250}},
           .B = 1.0e-4 *
                 Eigen::Matrix<double, 4, 2>{
                       {37.112, 7.361},
                       {-870.51, 0.093411},
                       {-11984.40, -4.1378},
                       {-31927.0, 9.2535}},
           .Q = 1e-2 * Eigen::Matrix4d::Identity(),
           .R = Eigen::Matrix2d::Identity(),
           .errorTolerance = {.laub = 5.6e-15, .vanDooren = 2.1e-15}},
          {.name = "4x_4u_IR",  // Example 8
           .A =
                 Eigen::Matrix4d{
                       {-0.6000000, -2.2000000, -3.6000000, -5.4000180},
                       {1.0000000, 0.6000000, 0.8000000, 3.3999820},
                       {0.0000000, 1.0000000, 1.8000000, 3.7999820},
                       {0.0000000, 0.0000000, 0.0000000, -0.9999820}},
           .B =
                 Eigen::Matrix4d{
                       {1.0, -1.0, -1.0, -1.0},
                       {0.0, 1.0, -1.0, -1.0},
                       {0.0, 0.0, 1.0, -1.0},
                       {0.0, 0.0, 0.0, 1.0}},
           .Q =
                 Eigen::Matrix4d{
                       {2.0, 1.0, 3.0, 6.0},
                       {1.0, 2.0, 2.0, 5.0},
                       {3.0, 2.0, 6.0, 11.0},
                       {6.0, 5.0, 11.0, 22.0}},
           .R = Eigen::Matrix4d::Identity(),
           .errorTolerance = {.laub = 5.5e-13, .vanDooren = 3.0e-13}},
          {.name = "5x_2u_IQ_IR",  // Example 9
           .A = 1.0e-4 *
                 Eigen::MatrixXd{
                       {9540.70, 196.43, 35.97, 6.73, 1.90},
                       {4084.90, 4131.70, 1608.40, 446.79, 119.71},
                       {1221.70, 2632.60, 3614.90, 1593.00, 1238.30},
                       {411.18, 1285.80, 2720.90, 2144.20, 4097.60},
                       {13.05, 58.08, 187.50, 361.62, 9428.00}},
           .B = 1.0e-4 *
                 Eigen::MatrixXd{
                       {4.34, -1.22},
                       {266.06, -104.53},
                       {375.30, -551.00},
                       {360.76, -660.00},
                       {46.17, -91.48}},
           .Q = Eigen::MatrixXd::Identity(5, 5),
           .R = Eigen::MatrixXd::Identity(2, 2),
           .errorTolerance = {.laub = 2.2e-12, .vanDooren = 4.0e-13}},
          {.name = "6x_2u_blockQ",  // Example 10
           .A =
                 []() {
                     Eigen::MatrixXd A = Eigen::Matrix<double, 6, 6>::Zero();
                     A.block<2, 2>(0, 1) = Eigen::Matrix2d::Identity();
                     A.block<2, 2>(3, 4) = Eigen::Matrix2d::Identity();
                     return A;
                 }(),
           .B =
                 []() {
                     Eigen::MatrixXd B = Eigen::Matrix<double, 6, 2>::Zero();
                     B(2, 0) = 1.0;
                     B(5, 1) = 1.0;
                     return B;
                 }(),
           .Q =
                 []() {
                     Eigen::MatrixXd Q = Eigen::Matrix<double, 6, 6>::Zero();
                     Q.block<2, 2>(0, 0) = Eigen::Matrix2d::Ones();
                     Q.block<2, 2>(3, 3) = Eigen::Matrix2d{
                           {1.0, -1.0}, {-1.0, 1.0}};
                     return Q;
                 }(),
           .R = Eigen::Matrix2d{{3, 0}, {0, 1}},
           .failsInLaub = true,  // A is singular
           .errorTolerance = {.vanDooren = 2.1e-15}},
          {.name = "9x_3u_IR",  // Example 11
           .A = 1.0e-2 *
                 Eigen::MatrixXd{
                       {87.01, 13.5, 1.159, 0.05014, -3.722, 0.03484, 0.0, 0.4242, 0.7249},
                       {7.655, 89.74, 1.272, 0.05504, -4.016, 0.03743, 0.0, 0.453, 0.7499},
                       {-12.72, 35.75, 81.7, 0.1455, -10.28, 0.0987, 0.0, 1.185, 1.872},
                       {-36.35, 63.39, 7.491, 79.66, -27.35, 0.2653, 0.0, 3.172, 4.882},
                       {-96.0, 164.59, -12.89, -0.5597, 7.142, 0.7108, 0.0, 8.452, 12.59},
                       {-66.44, 11.296, -8.889, -0.3854, 8.447, 1.36, 0.0, 14.43, 10.16},
                       {-41.02, 69.3, -5.471, -0.2371, 6.649, 1.249, 0.01063, 9.997, 6.967},
                       {-17.99, 30.17, -2.393, -0.1035, 6.059, 2.216, 0.0, 21.39, 3.554},
                       {-34.51, 58.04, -4.596, -0.1989, 10.56, 1.986, 0.0, 21.91, 21.52}},
           .B = 1.0e-4 *
                 Eigen::MatrixXd{
                       {4.7600, -0.5701, -83.6800},
                       {0.8790, -4.7730, -2.7300},
                       {1.4820, -13.1200, 8.8760},
                       {3.8920, -35.1300, 24.8000},
                       {10.3400, -92.7500, 66.8000},
                       {7.2030, -61.5900, 38.3400},
                       {4.4540, -36.8300, 20.2900},
                       {1.9710, -15.5400, 6.9370},
                       {3.7730, -30.2800, 14.6900}},
           .Q =
                 []() {
                     Eigen::Matrix<double, 2, 9> C = Eigen::Matrix<double, 2, 9>::Zero();
                     C(0, 0) = 1.0;
                     C(1, 4) = 1.0;
                     Eigen::Matrix<double, 9, 9> Q = 50.0 * C.transpose() * C;
                     return Q;
                 }(),
           .R = Eigen::Matrix3d::Identity(),
           .errorTolerance = {.laub = 2.2e-6, .vanDooren = 3.7e-10}},
          {.name = "2x_1u_parameterized",  // Example 12
           .A = Eigen::MatrixXd{{0.0, epsilon}, {0.0, 0.0}},
           .B = Eigen::MatrixXd{{0.0}, {1.0}},
           .Q = Eigen::MatrixXd::Identity(2, 2),
           .R = Eigen::MatrixXd{{1.0}},
           .analyticalX = Eigen::MatrixXd{{1.0, 0.0}, {0.0, 1.0 + epsilon * epsilon}},
           .failsInLaub = true,  // A is singular
           .errorTolerance = {.vanDooren = 1.7e-8}},
          {.name = "Nx_1u",  // Example 15
           .A =
                 []() {
                     Eigen::MatrixXd A = Eigen::MatrixXd::Zero(N, N);
                     A.diagonal(1).setOnes();  // Set ones on the superdiagonal
                     return A;
                 }(),
           .B =
                 []() {
                     Eigen::MatrixXd B = Eigen::MatrixXd::Zero(N, 1);
                     B.bottomRows(1)(0, 0) = 1.0;
                     return B;
                 }(),
           .Q = Eigen::MatrixXd::Identity(N, N),
           .R = Eigen::MatrixXd{{1.0}},
           .analyticalX =
                 []() {
                     Eigen::VectorXd diagonal = Eigen::VectorXd::LinSpaced(N, 1, N);
                     Eigen::MatrixXd X = diagonal.asDiagonal();
                     return X;
                 }(),
           .failsInLaub = true,  // A is singular
           .errorTolerance = {.vanDooren = 9.4e-13}},
    };
}

std::string RiccatiCaseName(const ::testing::TestParamInfo<RiccatiTestCase>& info) {
    return "case_" + std::to_string(info.index) + "_" + info.param.name;
}

// This is not really used, but defining it addresses a Valgrind error.
void PrintTo(const RiccatiTestCase& tc, std::ostream* os) { *os << tc.name; }

INSTANTIATE_TEST_SUITE_P(
      TestRiccati,
      TestVanDoorenRiccati,
      ::testing::ValuesIn(GenParams()),
      RiccatiCaseName);
