#pragma once

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>  // Includes LAPACK interfaces on macOS
#else

typedef int __LAPACK_int;
typedef int __LAPACK_bool;
typedef __LAPACK_bool (*__LAPACK_dgees_func_ptr)(double*, double*);

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
}

#endif  // __APPLE__
