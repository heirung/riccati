# riccati

This repo contains basic implementations of the two approaches by [Laub (1979)](#references) and [Van Dooren (1981)](#references) to solving the discrete-time algebraic Riccati equation (DARE),

```math
X = A^\top X A − A^\top X B (R + B^\top X B)^{-1} B^\top X A + Q
```

The method by Laub (1979) requires that both $A$ and $R$ be nonsingular, whereas the method by Van Dooren (1981) does not.
The code uses [Eigen](https://eigen.tuxfamily.org) and [LAPACK](https://www.netlib.org/lapack/) for linear algebra not implemented in Eigen.
I've tested this on both
Linux (Ubuntu 22.04 in a Docker container, with LAPACK/BLAS from `apt-get install liblapack-dev libblas-dev`) and on
macOS (Sonoma, on Apple silicon, with the LAPACK distribution that ships with macOS as part of Accelerate).
I had to relax some of the numerical tolerances in the tests when running them on Linux, so they might need further adjustment for a different setup.
There are some mildly surprising differences in LAPACK output between the two Linux and macOS versions I've used, so I'm not sure exactly how robust this is to various combinations of hardware, OS, and LAPACK installation.
My attempt at multi-platform build support in Bazel is janky, but it seems to do the job.
All of the interfacing with LAPACK is contained in [lapack_wrappers.h](riccati/lapack_wrappers.h) and [lapack_wrappers.cpp](riccati/lapack_wrappers.cpp).

There's very modest error checking and handling – the implementation will happily return `nan`s and `inf`s without warning.

### Dependencies

- C++20
- [LAPACK](https://www.netlib.org/lapack/) (`apt-get install liblapack-dev libblas-dev` on Linux, comes preinstalled on macOS)
- [Bazel](https://bazel.build)
- [Eigen](https://eigen.tuxfamily.org) (managed through Bazel)
- [GoogleTest](https://github.com/google/googletest) (managed through Bazel)

### [example.cpp](example.cpp)

```C++
#include <iostream>

#include "Eigen/Dense"
#include "absl/status/statusor.h"
#include "riccati/riccati.h"

// Run with one of
// `bazel run --config=macos //:example`
// `bazel run --config=linux //:example`
int main() {
    // Triple integrator sampled at 10 Hz (zero-order hold) with unity weights:
    static constexpr double dt = 0.1;
    const Eigen::Matrix3d Ac{
          {0.0, 1.0, 0.0},
          {0.0, 0.0, 1.0},
          {0.0, 0.0, 0.0}
    };
    const Eigen::Matrix3d A =
          Eigen::Matrix3d::Identity() + dt * Ac + (1.0 / 2.0) * dt * dt * Ac * Ac;
    const Eigen::Vector3d B{
          {(1.0 / 6.0) * dt * dt * dt, (1.0 / 2.0) * dt * dt, dt}
    };
    const Eigen::DiagonalMatrix<double, 3> Q = Eigen::DiagonalMatrix<double, 3>::Identity();
    const Eigen::DiagonalMatrix<double, 1> R = Eigen::DiagonalMatrix<double, 1>::Identity();

    // Solve the discrete-time algebraic Riccati equation with the approach from Van Dooren (1981):
    const absl::StatusOr<Eigen::MatrixXd> maybeX = riccati::SolveDiscrete(
          A, B, Q.toDenseMatrix(), R.toDenseMatrix(), riccati::Solver::VanDooren);
    if (!maybeX.ok()) {
        std::cout << "Error -- " << maybeX.status().message() << std::endl;
        return 1;
    }
    // Check numerical error:
    const Eigen::Matrix3d residual =
          riccati::Residual(A, B, Q.toDenseMatrix(), R.toDenseMatrix(), maybeX.value());

    std::cout << "X =\n" << maybeX.value() << std::endl;
    std::cout << "residual:\n" << residual << std::endl;
    std::cout << "residual.norm() = " << residual.norm() << std::endl;

    return 0;
}
```

Running this example (with `bazel run --config=macos //:example`, or `--config=linux`) produces the output

```Text
X =
24.6614 24.1761 10.0326
24.1761 48.8802 24.2117
10.0326 24.2117 24.7147
residual:
-2.44471e-13  -3.0731e-13 -1.33227e-13
 -3.0731e-13 -3.18856e-13 -1.26121e-13
-1.33227e-13 -1.26121e-13 -4.61853e-14
residual.norm() = 6.47888e-13
```

### References

Laub, A. J. A Schur method for solving algebraic Riccati equations. _IEEE Transactions on Automatic Control_, 24(6):913–921, 1979. <https://doi.org/10.1109/TAC.1979.1102178>

Van Dooren, P. M. A generalized eigenvalue approach for solving Riccati equations. _SIAM Journal on Scientific and Statistical Computing_, 2(2):121–135, 1981. <https://doi.org/10.1137/0902010>

Arnold, N. III and Laub, A. J. Generalized eigenproblem algorithms and software for algebraic Riccati equations. _Proceedings of the IEEE_, 72(12):1746–1754, 1984. <https://doi.org/10.1109/PROC.1984.13083>

Benner, P., Laub,  A. J., and Mehrmann. V. A collection of benchmark examples for the
numerical solution of algebraic Riccati equations II: Discrete-time case. Technical report SPC
95_23, Technische Universität Chemnitz-Zwickau, 1995. <https://www.tu-chemnitz.de/sfb393/Files/PDF/spc95-23.pdf>
