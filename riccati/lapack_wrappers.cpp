#include "lapack_wrappers.h"

#include "Eigen/Dense"
#include "lapack_compat.h"

namespace lapack {

namespace internal {
namespace {
schurSelectorFunc GetSchurSelector(SelectorName name) {
    static const std::unordered_map<SelectorName, schurSelectorFunc> selectorMap{
          {             SelectorName::none,                  nullptr},
          { SelectorName::insideUnitCircle,  schur::InsideUnitCircle},
          {SelectorName::outsideUnitCircle, schur::OutsideUnitCircle},
          {    SelectorName::leftHalfPlane,     schur::LeftHalfPlane},
          {   SelectorName::rightHalfPlane,    schur::RightHalfPlane},
    };
    return selectorMap.at(name);
}
}  // namespace

namespace schur {
__LAPACK_bool InsideUnitCircle(double* eigRe, double* eigIm) {
    return std::pow(*eigRe, 2) + std::pow(*eigIm, 2) <= 1.0;  // No reason to do the square root.
}

__LAPACK_bool OutsideUnitCircle(double* eigRe, double* eigIm) {
    return !InsideUnitCircle(eigRe, eigIm);
}

__LAPACK_bool LeftHalfPlane(double* eigRe, [[maybe_unused]] double* eigIm) { return *eigRe <= 0.0; }

__LAPACK_bool RightHalfPlane(double* eigRe, double* eigIm) { return !LeftHalfPlane(eigRe, eigIm); }
}  // namespace schur

}  // namespace internal

absl::StatusOr<std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXcd>>
Schur(const Eigen::Ref<const Eigen::MatrixXd>& A, const SelectorName selector) {
    const char JOBVS = 'V';  // 'V' ('N') to (not) compute Schur vectors
    internal::schurSelectorFunc SELECT = internal::GetSchurSelector(selector);
    const char SORT =
          (selector == SelectorName::none) ? 'N' : 'S';  // 'S' ('N') to (not) sort the eigenvalues
    const int N = A.cols();                              // A is `LDA` x `N`
    const int LDA = A.outerStride();  // Leading dimension of A (the number of rows)
    const int LDVS = LDA;  // Leading dimension of `VS` (same as `A`; `VS` is for the matrix Z)
    // Outputs: eigenvalues, Schur vectors, and other workspace.
    Eigen::MatrixXd T = A;     // `dgees` modifies the input matrix, so pass a non-const copy of A
    int SDIM;                  // Number of selected eigenvalues (for sorting)
    Eigen::VectorXd eigRe(N);  // Real part of the eigenvalues (passed as `WR`)
    Eigen::VectorXd eigIm(N);  // Imaginary part of the eigenvalues (passed as `WI`)
    Eigen::MatrixXd Z(N, N);   // Matrix of Schur vectors (passed as `VS`; not used if `JOBVS = N`;
                               // using an N x N matrix is more convenient than an N^2 x 1 vector)
    double WORK_QUERY;
    int LWORK = -1;  // -1 to query for optimal workspace size, stored in `WORK_QUERY`
    std::vector<__LAPACK_bool> BWORK(N);
    int INFO;  // Exit code

    // Query for optimal workspace size.
    dgees_(
          &JOBVS,
          &SORT,
          SELECT,
          &N,
          T.data(),
          &LDA,
          &SDIM,
          eigRe.data(),
          eigIm.data(),
          Z.data(),
          &LDVS,
          &WORK_QUERY,
          &LWORK,
          BWORK.data(),
          &INFO);
    if (INFO != 0) {
        return absl::InternalError(
              std::string(__func__) + ", DGEES error querying workspace size. N = " +
              std::to_string(N) + ", INFO = " + std::to_string(INFO));
    }

    LWORK = int(WORK_QUERY);
    std::vector<double> WORK(LWORK);  // Allocate optimally sized workspace.
    // Call `dgees` again, this time to get the Schur form, Schur vectors, and eigenvalues.
    dgees_(
          &JOBVS,
          &SORT,
          SELECT,
          &N,
          T.data(),
          &LDA,
          &SDIM,
          eigRe.data(),
          eigIm.data(),
          Z.data(),
          &LDVS,
          WORK.data(),
          &LWORK,
          BWORK.data(),
          &INFO);
    if (INFO != 0) {
        return absl::InternalError(
              std::string(__func__) + ", DGEES in final call. N = " + std::to_string(N) +
              ", INFO = " + std::to_string(INFO));
    }
    const Eigen::VectorXcd eigenvalues = eigRe + eigIm * std::complex<double>(0.0, 1.0);

    return std::make_tuple(T, Z, eigenvalues);
}
}  // namespace lapack
