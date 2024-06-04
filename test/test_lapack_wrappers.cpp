#include "Eigen/Dense"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "riccati/lapack_wrappers.h"

namespace {

constexpr double DOUBLE_TOLERANCE = 1.0e-6;

// Check if the (complex) eigenvalue `value` is contained in the complex vector `vector` within a
// given tolerance. (An O(n) search, not for repeated use on large vectors.)
bool ContainsEigenvalue(
      const Eigen::VectorXcd& vec,
      const std::complex<double>& value,
      double tolerance) {
    for (int i = 0; i < vec.size(); ++i) {
        if (std::abs(vec(i).real() - value.real()) <= tolerance &&
            std::abs(vec(i).imag() - value.imag()) <= tolerance) {
            return true;
        }
    }
    return false;
}
}  // namespace

using ::testing::ElementsAre;

TEST(TestSelectors2, TestUnitCircle) {
    using lapack::internal::schur::InsideUnitCircle;
    using lapack::internal::schur::OutsideUnitCircle;
    double re = 0.0;
    double im = 0.0;
    EXPECT_TRUE(InsideUnitCircle(&re, &im));
    EXPECT_FALSE(OutsideUnitCircle(&re, &im));

    re = 0.0;
    im = 1.0;
    EXPECT_TRUE(InsideUnitCircle(&re, &im));
    EXPECT_FALSE(OutsideUnitCircle(&re, &im));

    re = -1.0;
    im = 0.0;
    EXPECT_TRUE(InsideUnitCircle(&re, &im));
    EXPECT_FALSE(OutsideUnitCircle(&re, &im));

    re = 0.5;
    im = 0.5;
    EXPECT_TRUE(InsideUnitCircle(&re, &im));
    EXPECT_FALSE(OutsideUnitCircle(&re, &im));

    re = 1.0;
    im = 1.0;
    EXPECT_FALSE(InsideUnitCircle(&re, &im));
    EXPECT_TRUE(OutsideUnitCircle(&re, &im));
}

TEST(TestSelectors2, TestLeftRightHalfPlane) {
    using lapack::internal::schur::LeftHalfPlane;
    using lapack::internal::schur::RightHalfPlane;
    double re = 0.0;
    double im = 0.0;
    EXPECT_TRUE(LeftHalfPlane(&re, &im));
    EXPECT_FALSE(RightHalfPlane(&re, &im));

    re = 0.0;
    im = 1.0;
    EXPECT_TRUE(LeftHalfPlane(&re, &im));
    EXPECT_FALSE(RightHalfPlane(&re, &im));

    re = -1.0;
    im = 0.0;
    EXPECT_TRUE(LeftHalfPlane(&re, &im));
    EXPECT_FALSE(RightHalfPlane(&re, &im));

    re = 0.5;
    im = 0.5;
    EXPECT_FALSE(LeftHalfPlane(&re, &im));
    EXPECT_TRUE(RightHalfPlane(&re, &im));

    re = 1.0;
    im = 1.0;
    EXPECT_FALSE(LeftHalfPlane(&re, &im));
    EXPECT_TRUE(RightHalfPlane(&re, &im));
}

TEST(TestSchur, Test3x3Iuc) {
    const Eigen::MatrixXd A{
          {1.0, 2.0, 3.0},
          {2.0, 3.0, 4.0},
          {3.0, 4.0, 5.0}
    };
    const auto maybeSchur = lapack::Schur(A, lapack::SelectorName::insideUnitCircle);
    ASSERT_TRUE(maybeSchur.ok());
    const auto [T, Z, lambda] = maybeSchur.value();
    EXPECT_TRUE(A.isApprox(Z * T * Z.transpose(), DOUBLE_TOLERANCE));
    EXPECT_TRUE((Z.transpose() * Z).isApprox(Eigen::MatrixXd::Identity(3, 3), DOUBLE_TOLERANCE));
    // Check that the function doesn't just return the input matrices:
    EXPECT_FALSE(A.isApprox(T, DOUBLE_TOLERANCE));
    EXPECT_FALSE(Z.isApprox(Eigen::MatrixXd::Identity(A.cols(), A.cols()), DOUBLE_TOLERANCE));

    std::vector<__LAPACK_bool> eigenvalueAreInsideUnitCircle;
    eigenvalueAreInsideUnitCircle.reserve(lambda.size());
    std::transform(  // Is the eigenvalue inside the unit circle?
          lambda.cbegin(),
          lambda.cbegin() + lambda.size(),
          std::back_inserter(eigenvalueAreInsideUnitCircle),
          [](const std::complex<double>& z) {
              double re = z.real();
              double im = z.imag();
              return lapack::internal::schur::InsideUnitCircle(&re, &im);
          });
    int numInside = std::count_if(  // Count the number of eigenvalues inside the unit circle.
          eigenvalueAreInsideUnitCircle.cbegin(),
          eigenvalueAreInsideUnitCircle.cend(),
          [](__LAPACK_bool b) { return static_cast<bool>(b); });
    EXPECT_EQ(numInside, 2);
    EXPECT_TRUE(std::all_of(  // The eigenvalues inside the unit circle should appear first.
          eigenvalueAreInsideUnitCircle.begin(),
          eigenvalueAreInsideUnitCircle.begin() + numInside,
          [](__LAPACK_bool b) { return static_cast<bool>(b); }));
    EXPECT_TRUE(std::all_of(  // The eigenvalues outside the unit circle should appear last.
          eigenvalueAreInsideUnitCircle.begin() + numInside,
          eigenvalueAreInsideUnitCircle.end(),
          [](__LAPACK_bool b) { return !static_cast<bool>(b); }));
}

TEST(TestSelectors3, TestUnitCircle) {
    using lapack::internal::qz::InsideUnitCircle;
    using lapack::internal::qz::OutsideUnitCircle;
    // These selectors take eigenvalues of the form (alphar + alphai) / beta.
    double alphar = 0.0;
    double alphai = 0.0;
    double beta = 1.0;
    EXPECT_TRUE(InsideUnitCircle(&alphar, &alphai, &beta));
    EXPECT_FALSE(OutsideUnitCircle(&alphar, &alphai, &beta));

    alphar = 0.0;
    alphai = 1.0;
    EXPECT_TRUE(InsideUnitCircle(&alphar, &alphai, &beta));
    EXPECT_FALSE(OutsideUnitCircle(&alphar, &alphai, &beta));

    alphar = -1.0;
    alphai = 0.0;
    EXPECT_TRUE(InsideUnitCircle(&alphar, &alphai, &beta));
    EXPECT_FALSE(OutsideUnitCircle(&alphar, &alphai, &beta));

    alphar = 0.5;
    alphai = 0.5;
    EXPECT_TRUE(InsideUnitCircle(&alphar, &alphai, &beta));
    EXPECT_FALSE(OutsideUnitCircle(&alphar, &alphai, &beta));

    beta = 0.1;
    EXPECT_FALSE(InsideUnitCircle(&alphar, &alphai, &beta));
    EXPECT_TRUE(OutsideUnitCircle(&alphar, &alphai, &beta));

    alphar = 1.0;
    alphai = 1.0;
    beta = 3.0;
    EXPECT_TRUE(InsideUnitCircle(&alphar, &alphai, &beta));
    EXPECT_FALSE(OutsideUnitCircle(&alphar, &alphai, &beta));

    // Decided to treat eigenvalues with beta = 0 as outside the unit circle.
    alphar = 0.1;
    alphai = 0.1;
    beta = 0.0;
    EXPECT_FALSE(InsideUnitCircle(&alphar, &alphai, &beta));
    EXPECT_TRUE(OutsideUnitCircle(&alphar, &alphai, &beta));

    alphar = 10.0;
    alphai = 10.0;
    beta = 0.0;
    EXPECT_FALSE(InsideUnitCircle(&alphar, &alphai, &beta));
    EXPECT_TRUE(OutsideUnitCircle(&alphar, &alphai, &beta));
}

TEST(TestSelectors3, TestLeftRightHalfPlane) {
    using lapack::internal::qz::LeftHalfPlane;
    using lapack::internal::qz::RightHalfPlane;
    // These selectors take eigenvalues of the form (alphar + alphai) / beta.
    double alphar = 0.0;
    double alphai = 0.0;
    double beta = 1.0;
    EXPECT_TRUE(LeftHalfPlane(&alphar, &alphai, &beta));
    EXPECT_FALSE(RightHalfPlane(&alphar, &alphai, &beta));

    alphar = 0.0;
    alphai = 1.0;
    EXPECT_TRUE(LeftHalfPlane(&alphar, &alphai, &beta));
    EXPECT_FALSE(RightHalfPlane(&alphar, &alphai, &beta));

    alphar = -1.0;
    alphai = 0.0;
    EXPECT_TRUE(LeftHalfPlane(&alphar, &alphai, &beta));
    EXPECT_FALSE(RightHalfPlane(&alphar, &alphai, &beta));

    alphar = 0.5;
    alphai = 0.5;
    EXPECT_FALSE(LeftHalfPlane(&alphar, &alphai, &beta));
    EXPECT_TRUE(RightHalfPlane(&alphar, &alphai, &beta));

    alphar = 1.0;
    alphai = 1.0;
    beta = 0.0;
    EXPECT_FALSE(LeftHalfPlane(&alphar, &alphai, &beta));
    EXPECT_TRUE(RightHalfPlane(&alphar, &alphai, &beta));

    alphar = -1.0;
    alphai = 1.0;
    beta = 0.0;
    EXPECT_TRUE(LeftHalfPlane(&alphar, &alphai, &beta));
    EXPECT_FALSE(RightHalfPlane(&alphar, &alphai, &beta));

    alphar = 0.0;
    alphai = 0.0;
    beta = 0.0;
    EXPECT_TRUE(LeftHalfPlane(&alphar, &alphai, &beta));
    EXPECT_FALSE(RightHalfPlane(&alphar, &alphai, &beta));
}

TEST(TestQz, Test5x5) {
    constexpr int n = 5;
    const Eigen::MatrixXd A{
          { -7.0,   5.0,  11.0, -4.0,  13.0},
          {-11.0,  -3.0,  11.0,  8.0, -19.0},
          { -6.0,   3.0,  -5.0,  0.0, -12.0},
          { -4.0, -12.0, -14.0,  8.0,  -8.0},
          { 11.0,   0.0,   9.0,  6.0,  10.0}
    };
    Eigen::MatrixXd B = Eigen::MatrixXd::Identity(n, n);
    B(4, 4) = 0.0;
    const auto maybeQz = lapack::Qz(A, B);
    ASSERT_TRUE(maybeQz.ok());
    const auto [S, T, Q, Z, ALPHAR, ALPHAI, BETA] = maybeQz.value();
    // Invariant-based checks:
    EXPECT_TRUE((Q.transpose() * A * Z).isApprox(S, DOUBLE_TOLERANCE));
    EXPECT_TRUE((Q.transpose() * B * Z).isApprox(T, DOUBLE_TOLERANCE));
    EXPECT_TRUE((Q.transpose() * Q).isApprox(Eigen::MatrixXd::Identity(n, n), DOUBLE_TOLERANCE));
    EXPECT_TRUE((Z.transpose() * Z).isApprox(Eigen::MatrixXd::Identity(n, n), DOUBLE_TOLERANCE));
    // Check that the function doesn't just return the input matrices:
    EXPECT_FALSE(Q.isApprox(Eigen::MatrixXd::Identity(n, n), DOUBLE_TOLERANCE));
    EXPECT_FALSE(Z.isApprox(Eigen::MatrixXd::Identity(n, n), DOUBLE_TOLERANCE));
    EXPECT_FALSE(S.isApprox(A, DOUBLE_TOLERANCE));
    EXPECT_FALSE(T.isApprox(B, DOUBLE_TOLERANCE));
    // Check the eigenvalues:
    Eigen::VectorXcd qzEigenvalues = lapack::internal::GetEigenvalues(ALPHAR, ALPHAI, BETA);
    // There should be exactly one non-finite, or ill-defined, eigenvalue (with a denominator (BETA)
    // of zero).
    EXPECT_EQ(
          std::count_if(
                BETA.cbegin(), BETA.cend(), [](double b) { return lapack::internal::IsZero(b); }),
          1);
    // The other four eigenvalues should be finite, or well-defined.
    const std::vector<std::complex<double>> expectedFiniteEigenvalues{
          {-18.6931,      0.0},
          {-7.57208,      0.0},
          { 10.2826, -13.7357},
          { 10.2826,  13.7357}
    };
    for (size_t i = 0; i != expectedFiniteEigenvalues.size(); ++i) {
        EXPECT_TRUE(ContainsEigenvalue(qzEigenvalues, expectedFiniteEigenvalues[i], 1.0e-4))
              << "Expected eigenvalue with index " << i << " not found among computed eigenvalues.";
    }
}

TEST(TestQz, TestBasic3x3) {
    constexpr double NEAR_EXACT_TOLERANCE = 1.0e-15;
    const Eigen::MatrixXd A{
          {2.0, -1.0,  0.0},
          {0.0,  3.0, -1.0},
          {0.0,  0.0,  0.5}
    };
    const int n = A.cols();
    const Eigen::MatrixXd B = Eigen::MatrixXd::Identity(n, n);
    const std::vector<std::complex<double>> expectedEigenvalues{
          {2.0, 0.0},
          {3.0, 0.0},
          {0.5, 0.0}
    };  // The last eigenvalue is the only one inside the unit circle.
    {
        const auto maybeQz = lapack::Qz(A, B, lapack::SelectorName::none);
        ASSERT_TRUE(maybeQz.ok());
        const auto [S, T, Q, Z, ALPHAR, ALPHAI, BETA] = maybeQz.value();
        // The matrices and eigenvalues should not be reordered:
        EXPECT_TRUE(A.isApprox(S, NEAR_EXACT_TOLERANCE));
        EXPECT_TRUE(B.isApprox(T, NEAR_EXACT_TOLERANCE));
        EXPECT_TRUE(Q.isApprox(Q, NEAR_EXACT_TOLERANCE));
        EXPECT_TRUE(Z.isApprox(Z, NEAR_EXACT_TOLERANCE));
        // Safe to compare the eigenvalues directly since reordering would be an error:
        const Eigen::VectorXcd qzEigenvalues =
              lapack::internal::GetEigenvalues(ALPHAR, ALPHAI, BETA);
        for (size_t i = 0; i != expectedEigenvalues.size(); ++i) {
            EXPECT_TRUE(ContainsEigenvalue(qzEigenvalues, expectedEigenvalues[i], 1.0e-4))
                  << "Expected eigenvalue with index " << i
                  << " not found among computed eigenvalues.";
        }
    }
    {
        // Reorder such that the stable (third) eigenvalue is first:
        const auto maybeQz = lapack::Qz(A, B, lapack::SelectorName::insideUnitCircle);
        ASSERT_TRUE(maybeQz.ok());
        const auto [S, T, Q, Z, ALPHAR, ALPHAI, BETA] = maybeQz.value();
        // Invariant-based checks:
        EXPECT_TRUE(A.isApprox(Q * S * Z.transpose(), DOUBLE_TOLERANCE));
        EXPECT_TRUE(B.isApprox(Q * T * Z.transpose(), DOUBLE_TOLERANCE));
        EXPECT_TRUE(
              (Q.transpose() * Q).isApprox(Eigen::MatrixXd::Identity(n, n), DOUBLE_TOLERANCE));
        EXPECT_TRUE(
              (Z.transpose() * Z).isApprox(Eigen::MatrixXd::Identity(n, n), DOUBLE_TOLERANCE));
        // Check that the function doesn't just return the input matrices:
        EXPECT_FALSE(Q.isApprox(Eigen::MatrixXd::Identity(n, n), DOUBLE_TOLERANCE));
        EXPECT_FALSE(Z.isApprox(Eigen::MatrixXd::Identity(n, n), DOUBLE_TOLERANCE));
        EXPECT_FALSE(S.isApprox(A, DOUBLE_TOLERANCE));

        const Eigen::VectorXcd qzEigenvalues =
              lapack::internal::GetEigenvalues(ALPHAR, ALPHAI, BETA);
        EXPECT_DOUBLE_EQ(
              qzEigenvalues(0).real(), A(2, 2));  // The stable eigenvalue (0.5, 0.0) is first.
        for (size_t i = 0; i != expectedEigenvalues.size(); ++i) {
            EXPECT_TRUE(ContainsEigenvalue(qzEigenvalues, expectedEigenvalues[i], 1.0e-6))
                  << "Expected eigenvalue with index " << i
                  << " not found among computed eigenvalues.";
        }
    }
}

TEST(TestQz, Test4x4LeftHalfPlane) {
    Eigen::MatrixXd A{
          {3.9, 2.5, -3.5, 2.5},
          {4.3, 1.5, -3.5, 3.5},
          {4.1, 1.5, -2.5, 1.5},
          {4.4, 6.0, -4.0, 3.0}
    };
    Eigen::MatrixXd B{
          {1.0, 1.0, -3.0, 1.0},
          {2.0, 3.0, -5.0, 0.4},
          {1.0, 2.0, -4.0, 1.0},
          {1.2, 3.0, -4.0, 4.0}
    };
    const int n = A.cols();
    // There are two eigenvalues in the left half, which appear last with no ordering.
    const auto maybeQz = lapack::Qz(A, B, lapack::SelectorName::leftHalfPlane);
    ASSERT_TRUE(maybeQz.ok());
    const auto [S, T, Q, Z, ALPHAR, ALPHAI, BETA] = maybeQz.value();
    EXPECT_LT(ALPHAR(0) / BETA(0), 0.0);
    EXPECT_LT(ALPHAR(1) / BETA(1), 0.0);
    EXPECT_GT(ALPHAR(2) / BETA(2), 0.0);
    EXPECT_GT(ALPHAR(3) / BETA(3), 0.0);

    EXPECT_TRUE(A.isApprox(Q * S * Z.transpose(), DOUBLE_TOLERANCE));
    EXPECT_TRUE(B.isApprox(Q * T * Z.transpose(), DOUBLE_TOLERANCE));
    // Check that the function doesn't just return the input matrices:
    EXPECT_FALSE(Q.isApprox(Eigen::MatrixXd::Identity(n, n), DOUBLE_TOLERANCE));
    EXPECT_FALSE(Z.isApprox(Eigen::MatrixXd::Identity(n, n), DOUBLE_TOLERANCE));
    EXPECT_FALSE(S.isApprox(A, DOUBLE_TOLERANCE));
    EXPECT_FALSE(T.isApprox(B, DOUBLE_TOLERANCE));
}

TEST(TestQz, TestSelectors) {
    Eigen::VectorXd ALPHAR{
          {-1.0, 0.0, 1.0, -0.5, 0.5, -1.0, 0.0, 1.0, -0.5, 0.5, -1.0, 0.0, 1.0}
    };
    Eigen::VectorXd ALPHAI{
          {-1.0, -1.0, -1.0, -0.5, -0.5, 0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0}
    };
    Eigen::VectorXd BETA{
          {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
    };
    ASSERT_TRUE(ALPHAR.size() == ALPHAI.size());
    ASSERT_TRUE(ALPHAI.size() == BETA.size());
    // Marginally stable eigenvalues (on the unit circle, on the imaginary axis) treated as inside.
    std::vector<__LAPACK_bool> iuc = lapack::internal::order_qz::EigenvaluesToSelect(
          ALPHAR, ALPHAI, BETA, lapack::SelectorName::insideUnitCircle);
    std::vector<__LAPACK_bool> ouc = lapack::internal::order_qz::EigenvaluesToSelect(
          ALPHAR, ALPHAI, BETA, lapack::SelectorName::outsideUnitCircle);
    std::vector<__LAPACK_bool> lhp = lapack::internal::order_qz::EigenvaluesToSelect(
          ALPHAR, ALPHAI, BETA, lapack::SelectorName::leftHalfPlane);
    std::vector<__LAPACK_bool> rhp = lapack::internal::order_qz::EigenvaluesToSelect(
          ALPHAR, ALPHAI, BETA, lapack::SelectorName::rightHalfPlane);
    const std::vector<__LAPACK_bool> iucExpected{
          false, true, false, true, true, true, true, true, true, true, false, true, false};
    const std::vector<__LAPACK_bool> oucExpected{
          true, false, true, false, false, false, false, false, false, false, true, false, true};
    const std::vector<__LAPACK_bool> lhpExpected{
          true, true, false, true, false, true, true, false, true, false, true, true, false};
    const std::vector<__LAPACK_bool> rhpExpected{
          false, false, true, false, true, false, false, true, false, true, false, false, true};
    EXPECT_EQ(iuc, iucExpected);
    EXPECT_EQ(ouc, oucExpected);
    EXPECT_EQ(lhp, lhpExpected);
    EXPECT_EQ(rhp, rhpExpected);
}

TEST(TestReorderQz, Test4x4LeftHalfPlane) {
    Eigen::MatrixXd A{
          {3.9, 2.5, -3.5, 2.5},
          {4.3, 1.5, -3.5, 3.5},
          {4.1, 1.5, -2.5, 1.5},
          {4.4, 6.0, -4.0, 3.0}
    };
    Eigen::MatrixXd B{
          {1.0, 1.0, -3.0, 1.0},
          {2.0, 3.0, -5.0, 0.4},
          {1.0, 2.0, -4.0, 1.0},
          {1.2, 3.0, -4.0, 4.0}
    };
    // There are two eigenvalues in the left half plane, which appear last with no ordering.
    const auto maybeQz = lapack::Qz(A, B, lapack::SelectorName::none);
    ASSERT_TRUE(maybeQz.ok());
    const auto [S, T, Q, Z, ALPHAR, ALPHAI, BETA] = maybeQz.value();
    // The first two eigenvalues are in the right half plane, the other two are in the left half
    // plane.
    EXPECT_GT(ALPHAR(0) / BETA(0), 0.0);  // RHP
    EXPECT_GT(ALPHAR(1) / BETA(1), 0.0);  // RHP
    EXPECT_LT(ALPHAR(2) / BETA(2), 0.0);  // LHP
    EXPECT_LT(ALPHAR(3) / BETA(3), 0.0);  // LHP
    EXPECT_TRUE(A.isApprox(Q * S * Z.transpose(), DOUBLE_TOLERANCE));
    EXPECT_TRUE(B.isApprox(Q * T * Z.transpose(), DOUBLE_TOLERANCE));

    // Reorder such that the two stable eigenvalues appear first:
    std::vector<__LAPACK_bool> isLhp = lapack::internal::order_qz::EigenvaluesToSelect(
          ALPHAR, ALPHAI, BETA, lapack::SelectorName::leftHalfPlane);
    ASSERT_THAT(isLhp, ElementsAre(false, false, true, true));
    const auto maybeReorderQz = lapack::ReorderQz(S, T, Q, Z, isLhp);
    ASSERT_TRUE(maybeReorderQz.ok());
    const auto [orderedS, orderedT, orderedQ, orderedZ, orderedALPHAR, orderedALPHAI, orderedBETA] =
          maybeReorderQz.value();
    EXPECT_TRUE(A.isApprox(orderedQ * orderedS * orderedZ.transpose(), DOUBLE_TOLERANCE));
    EXPECT_TRUE(B.isApprox(orderedQ * orderedT * orderedZ.transpose(), DOUBLE_TOLERANCE));
    EXPECT_LT(orderedALPHAR(0) / orderedBETA(0), 0.0);  // LHP
    EXPECT_LT(orderedALPHAR(1) / orderedBETA(1), 0.0);  // LHP
    EXPECT_GT(orderedALPHAR(2) / orderedBETA(2), 0.0);  // RHP
    EXPECT_GT(orderedALPHAR(3) / orderedBETA(3), 0.0);  // RHP
}
