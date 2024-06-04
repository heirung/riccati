#include "lapack_wrappers.h"

#include <cmath>
#include <complex>
#include <functional>
#include <unordered_map>

#include "Eigen/Dense"
#include "absl/status/statusor.h"
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

qzSelectorFunc GetQzSelector(SelectorName name) {
    static const std::unordered_map<SelectorName, qzSelectorFunc> selectorMap{
          {             SelectorName::none,               nullptr},
          { SelectorName::insideUnitCircle,  qz::InsideUnitCircle},
          {SelectorName::outsideUnitCircle, qz::OutsideUnitCircle},
          {    SelectorName::leftHalfPlane,     qz::LeftHalfPlane},
          {   SelectorName::rightHalfPlane,    qz::RightHalfPlane},
    };
    return selectorMap.at(name);
}
}  // namespace

bool IsZero(double value) { return std::abs(value) < ZERO_TOLERANCE; }

Eigen::VectorXcd GetEigenvalues(
      const Eigen::Ref<const Eigen::VectorXd>& ALPHAR,
      const Eigen::Ref<const Eigen::VectorXd>& ALPHAI,
      const Eigen::Ref<const Eigen::VectorXd>& BETA) {
    return (ALPHAR.array() + ALPHAI.array() * std::complex<double>(0.0, 1.0)) / BETA.array();
}

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

namespace qz {

__LAPACK_bool InsideUnitCircle(double* ALPHAR, double* ALPHAI, double* BETA) {
    return std::pow(*ALPHAR, 2.0) + std::pow(*ALPHAI, 2.0) <= std::pow(*BETA, 2.0);
}

__LAPACK_bool OutsideUnitCircle(double* ALPHAR, double* ALPHAI, double* BETA) {
    return !InsideUnitCircle(ALPHAR, ALPHAI, BETA);
}

__LAPACK_bool LeftHalfPlane(double* ALPHAR, [[maybe_unused]] double* ALPHAI, double* BETA) {
    // Ensures a reasonably well-defined result when `BETA` is zero.
    return IsZero(*BETA) ? *ALPHAR <= 0.0 : (*ALPHAR / *BETA) <= 0.0;
}

__LAPACK_bool RightHalfPlane(double* ALPHAR, [[maybe_unused]] double* ALPHAI, double* BETA) {
    return !LeftHalfPlane(ALPHAR, ALPHAI, BETA);
}
}  // namespace qz

namespace order_qz {

std::vector<__LAPACK_bool> EigenvaluesToSelect(
      const Eigen::Ref<const Eigen::VectorXd>& alphaRe,
      const Eigen::Ref<const Eigen::VectorXd>& alphaIm,
      const Eigen::Ref<const Eigen::VectorXd>& beta,
      const SelectorName selectorName) {
    const int N = alphaRe.size();
    std::vector<__LAPACK_bool> result(N);
    if (selectorName == SelectorName::none) {
        return result;
    }
    qzSelectorFunc selector = internal::GetQzSelector(selectorName);
    for (int i = 0; i < N; ++i) {
        // Arguably better to make copies than to cast away the const, or to have two separate
        // implementations (`double*` args for passing to LAPACK, `const double*` for use here).
        double alpharLapack = alphaRe[i];
        double alphaiLapack = alphaIm[i];
        double betaLapack = beta[i];
        result[i] = std::invoke(selector, &alpharLapack, &alphaiLapack, &betaLapack);
    }
    return result;
}
}  // namespace order_qz
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

absl::StatusOr<std::tuple<
      Eigen::MatrixXd,
      Eigen::MatrixXd,
      Eigen::MatrixXd,
      Eigen::MatrixXd,
      Eigen::VectorXd,
      Eigen::VectorXd,
      Eigen::VectorXd>>
Qz(const Eigen::Ref<const Eigen::MatrixXd>& A,
   const Eigen::Ref<const Eigen::MatrixXd>& B,
   const SelectorName selector) {
    const char JOBVSL = 'V';  // 'V' ('N') to (not) compute left Schur vectors (in Q)
    const char JOBVSR = 'V';  // 'V' ('N') to (not) compute right Schur vectors (in Z)
    internal::qzSelectorFunc SELECT = internal::GetQzSelector(selector);
    const char SORT =
          (selector == SelectorName::none) ? 'N' : 'S';  // 'S' ('N') to (not) sort the eigenvalues
    const int LDA = A.outerStride();  // Leading dimension of A (the number of rows)
    const int LDB = B.outerStride();  // Leading dimension of B (the number of rows)
    const int N = A.cols();
    Eigen::MatrixXd S = A;  // `dgges` modifies the input matrices, so pass non-const copies of A, B
    Eigen::MatrixXd T = B;
    // Outputs: eigenvalues, Schur vectors, and other workspace.
    // Each eigenvalue j is represented by (ALPHAR(j) + ALPHAI(j)) / BETA(j).
    Eigen::VectorXd ALPHAR(N);  // Real parts of the eigenvalue numerators
    Eigen::VectorXd ALPHAI(N);  // Imaginary parts of the eigenvalue numerators
    Eigen::VectorXd BETA(N);    // Eigenvalue denominators
    Eigen::MatrixXd Q(N, N);    // Q contains the left Schur vectors (passed as `VSL`)
    Eigen::MatrixXd Z(N, N);    // Z contains the right Schur vectors (passed as `VSR`)
    const int LDQ = N;
    const int LDZ = N;
    int SDIM;           // Number of selected eigenvalues (for sorting)
    int LWORK = -1;     // -1 to query for optimal workspace size, stored in `WORK_QUERY`
    double WORK_QUERY;  // Optimal work size
    std::vector<__LAPACK_bool> BWORK(N);  // Whether (pre-sorting) an eigenvalue satisfies `SELECT`
    int INFO;                             // Exit code

    // Query for optimal workspace size.
    dgges_(
          &JOBVSL,
          &JOBVSR,
          &SORT,
          SELECT,
          &N,
          S.data(),
          &LDA,
          T.data(),
          &LDB,
          &SDIM,
          ALPHAR.data(),
          ALPHAI.data(),
          BETA.data(),
          Q.data(),
          &LDQ,
          Z.data(),
          &LDZ,
          &WORK_QUERY,
          &LWORK,
          BWORK.data(),
          &INFO);
    if (INFO != 0) {
        return absl::InternalError(
              std::string(__func__) + ", DGGES error querying workspace size. N = " +
              std::to_string(N) + ", INFO = " + std::to_string(INFO));
    }

    LWORK = int(WORK_QUERY);
    std::vector<double> WORK(LWORK);  // Allocate optimally sized workspace.
    // Call `dgges` again, now to compute the generalized eigenvalues and left/right Schur vectors.
    dgges_(
          &JOBVSL,
          &JOBVSR,
          &SORT,
          SELECT,
          &N,
          S.data(),
          &LDA,
          T.data(),
          &LDB,
          &SDIM,
          ALPHAR.data(),
          ALPHAI.data(),
          BETA.data(),
          Q.data(),
          &LDQ,
          Z.data(),
          &LDZ,
          WORK.data(),
          &LWORK,
          BWORK.data(),
          &INFO);
    if (INFO != 0) {
        return absl::InternalError(
              std::string(__func__) + ", DGGES error in final call. N = " + std::to_string(N) +
              ", INFO = " + std::to_string(INFO));
    }

    return std::make_tuple(S, T, Q, Z, ALPHAR, ALPHAI, BETA);
}

absl::StatusOr<std::tuple<
      Eigen::MatrixXd,
      Eigen::MatrixXd,
      Eigen::MatrixXd,
      Eigen::MatrixXd,
      Eigen::VectorXd,
      Eigen::VectorXd,
      Eigen::VectorXd>>
ReorderQz(
      const Eigen::Ref<const Eigen::MatrixXd>& A,
      const Eigen::Ref<const Eigen::MatrixXd>& B,
      const Eigen::Ref<const Eigen::MatrixXd>& Q,
      const Eigen::Ref<const Eigen::MatrixXd>& Z,
      const std::vector<__LAPACK_bool>& SELECT) {
    const int IJOB = 0;  // Specifies whether condition numbers are required for the cluster of
                         // eigenvalues (PL and PR) or the deflating subspaces (Difu and Difl).
    const __LAPACK_bool WANTQ = true;
    const __LAPACK_bool WANTZ = true;
    const int N = A.cols();
    const int LDA = A.outerStride();  // Leading dimension of A (the number of rows)
    const int LDB = B.outerStride();  // Leading dimension of B (the number of rows)
    const int LDQ = N;                // Q is N x N
    const int LDZ = N;                // Z is N x N
    // Outputs
    Eigen::MatrixXd orderedA = A;  // `dtgsen` modifies the input matrices, so pass non-const copies
    Eigen::MatrixXd orderedB = B;  // of A, B
    Eigen::VectorXd ALPHAR(N);     // Real parts of the eigenvalue numerators
    Eigen::VectorXd ALPHAI(N);     // Imaginary parts of the eigenvalue numerators
    Eigen::VectorXd BETA(N);       // Eigenvalue denominators
    Eigen::MatrixXd orderedQ = Q;  // Left Schur vectors, will be reordered by `dtgsen`
    Eigen::MatrixXd orderedZ = Z;  // Right Schur vectors, will be reordered by `dtgsen`
    int M;      // The dimension of the pair of left and right eigenspaces (deflating subspaces)
    double PL;  // If IJOB = 1, 4 or 5, PL, PR are lower bounds on the reciprocal of the norm of the
    double PR;  // norm of "projections" onto left and right eigenspaces wrt the selected cluster.
    std::vector<double> DIF(2);  // Holds the estimates of Difu and Difl (reciprocal cond. numbers)
    int LWORK = -1;              // -1 for workspace query
    double WORK_QUERY;           // Returns optimal `LWORK` if `LWORK` = -1 and `INFO` = 0
    int LIWORK = -1;             // -1 for workspace query
    int IWORK_QUERY;             // Returns optimal `LIWORK` if `LIWORK` = -1 and `INFO` = 0
    int INFO;                    // Exit code

    // Query for optimal workspace size.
    dtgsen_(
          &IJOB,
          &WANTQ,
          &WANTZ,
          SELECT.data(),
          &N,
          orderedA.data(),
          &LDA,
          orderedB.data(),
          &LDB,
          ALPHAR.data(),
          ALPHAI.data(),
          BETA.data(),
          orderedQ.data(),
          &LDQ,
          orderedZ.data(),
          &LDZ,
          &M,
          &PL,
          &PR,
          DIF.data(),
          &WORK_QUERY,
          &LWORK,
          &IWORK_QUERY,
          &LIWORK,
          &INFO);
    if (INFO != 0) {
        return absl::InternalError(
              std::string(__func__) + ", DTGSEN error querying workspace size. N = " +
              std::to_string(N) + ", INFO = " + std::to_string(INFO));
    }

    LWORK = int(WORK_QUERY);
    LIWORK = IWORK_QUERY;
    std::vector<double> WORK(LWORK);  // Allocate optimally sized workspace.
    std::vector<int> IWORK(LIWORK);   // Allocate optimally sized workspace.
    // Call `dtgsen`, now to do the actual reordering.
    dtgsen_(
          &IJOB,
          &WANTQ,
          &WANTZ,
          SELECT.data(),
          &N,
          orderedA.data(),
          &LDA,
          orderedB.data(),
          &LDB,
          ALPHAR.data(),
          ALPHAI.data(),
          BETA.data(),
          orderedQ.data(),
          &LDQ,
          orderedZ.data(),
          &LDZ,
          &M,
          &PL,
          &PR,
          DIF.data(),
          WORK.data(),
          &LWORK,
          IWORK.data(),
          &LIWORK,
          &INFO);
    if (INFO != 0) {
        return absl::InternalError(
              std::string(__func__) + ", DTGSEN error in final call. N = " + std::to_string(N) +
              ", INFO = " + std::to_string(INFO));
    }

    return std::make_tuple(orderedA, orderedB, orderedQ, orderedZ, ALPHAR, ALPHAI, BETA);
}
}  // namespace lapack
