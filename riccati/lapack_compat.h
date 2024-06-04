#pragma once

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>  // Includes LAPACK interfaces on macOS
#else

typedef int __LAPACK_int;
typedef int __LAPACK_bool;
typedef __LAPACK_bool (*__LAPACK_dgees_func_ptr)(double*, double*);
typedef __LAPACK_bool (*__LAPACK_dgges_func_ptr)(double*, double*, double*);

extern "C" {

// LAPACK: Schur decomposition of a real nonsymmetric matrix
void dgees_(
      const char* JOBVS,
      const char* SORT,
      __LAPACK_dgees_func_ptr SELECT,
      const int* N,
      double* A,
      const int* LDA,
      int* SDIM,
      double* WR,
      double* WI,
      double* VS,
      const int* LDVS,
      double* WORK,
      const int* LWORK,
      __LAPACK_bool* BWORK,
      int* INFO);

// LAPACK: Generalized real Schur decomposition (QZ)
void dgges_(
      const char* JOBVSL,
      const char* JOBVSR,
      const char* SORT,
      __LAPACK_dgges_func_ptr SELCTG,
      const int* N,
      double* A,
      const int* LDA,
      double* B,
      const int* LDB,
      int* SDIM,
      double* ALPHAR,
      double* ALPHAI,
      double* BETA,
      double* VSL,
      const int* LDVSL,
      double* VSR,
      const int* LDVSR,
      double* WORK,
      const int* LWORK,
      __LAPACK_bool* BWORK,
      int* INFO);

// LAPACK: Reordering of generalized Schur form
void dtgsen_(
      const int* IJOB,
      const __LAPACK_bool* WANTQ,
      const __LAPACK_bool* WANTZ,
      const __LAPACK_bool* SELECT,
      const int* N,
      double* A,
      const int* LDA,
      double* B,
      const int* LDB,
      double* ALPHAR,
      double* ALPHAI,
      double* BETA,
      double* Q,
      const int* LDQ,
      double* Z,
      const int* LDZ,
      int* M,
      double* PL,
      double* PR,
      double* DIF,
      double* WORK,
      const int* LWORK,
      int* IWORK,
      const int* LIWORK,
      int* INFO);
}

#endif  // __APPLE__
