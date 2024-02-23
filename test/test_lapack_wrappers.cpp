#include "Eigen/Dense"
#include "gtest/gtest.h"
#include "riccati/lapack_wrappers.h"

namespace {
constexpr double DOUBLE_TOLERANCE = 1.0e-6;
}  // namespace

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
