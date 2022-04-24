#include "eigenproblem.h"

#include <iostream>

using namespace std;

Eigenproblem::Eigenproblem() {
  MPI_Comm_size(MPI_COMM_WORLD, &_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &_rank);
}

void Eigenproblem::Initialize(PetscInt p_KS_size, PetscInt p_dimension,
                              PetscBool p_complex) {
  _KS_size = p_KS_size;
  _dimension = p_dimension;
  _complex = p_complex;

  PetscMalloc(_dimension * (_KS_size + 1) * sizeof(PetscScalar), &_mat_Q);
  PetscMalloc((_KS_size + 1) * (_KS_size + 1) * sizeof(PetscScalar), &_mat_H);
  PetscMalloc(_KS_size * sizeof(PetscScalar), &_eigenvalue);

  for (PetscInt r = 0; r < p_dimension; r++)
    for (PetscInt c = 0; c < _KS_size + 1; c++)
      _mat_Q[(_KS_size + 1) * r + c] = 0.0;

  for (PetscInt r = 0; r < _KS_size + 1; r++)
    for (PetscInt c = 0; c < _KS_size + 1; c++)
      _mat_H[(_KS_size + 1) * r + c] = 0.0;
}

void Eigenproblem::Finalize() {
  PetscFree(_mat_Q);
  PetscFree(_mat_H);
  PetscFree(_eigenvalue);
}

void Eigenproblem::ArnoldiUpdate(PetscInt p_col, PetscBool p_zero_pressure,
                                 PetscBool p_weighted, PetscReal *p_weights,
                                 PetscScalar *r_vec_to_orthogonalize) {
  PetscReal eps = 1e-8;

  // Scale vector
  if (p_zero_pressure == PETSC_TRUE) {
    for (PetscInt i = 0; i < _dimension; i++) {
      if (i % NVAR == 0) r_vec_to_orthogonalize[i] = 0.0;
    }
  }

  if (p_weighted == PETSC_TRUE) {
    for (PetscInt i = 0; i < _dimension; i++)
      r_vec_to_orthogonalize[i] = p_weights[i] * r_vec_to_orthogonalize[i];
  }

  if (p_col == 0) {
    // Initial guess

    //  q_0 = Z_0
    for (PetscInt row = 0; row < _dimension; row++)
      _mat_Q[(_KS_size + 1) * row + p_col] = r_vec_to_orthogonalize[row];

    // norm = ||q_0||
    PetscReal norm, locNorm = 0.0;
    for (PetscInt row = 0; row < _dimension; row++)
      locNorm += PetscRealPart(_mat_Q[(_KS_size + 1) * row + p_col]) *
                     PetscRealPart(_mat_Q[(_KS_size + 1) * row + p_col]) +
                 PetscImaginaryPart(_mat_Q[(_KS_size + 1) * row + p_col]) *
                     PetscImaginaryPart(_mat_Q[(_KS_size + 1) * row + p_col]);

    MPI_Allreduce(&locNorm, &norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    norm = sqrt(norm);

    for (PetscInt row = 0; row < _dimension; row++)
      _mat_Q[(_KS_size + 1) * row + p_col] =
          _mat_Q[(_KS_size + 1) * row + p_col] / norm;
  } else {
    //  q_k = Z_k
    for (PetscInt row = 0; row < _dimension; row++)
      _mat_Q[(_KS_size + 1) * row + p_col] = r_vec_to_orthogonalize[row];

    // Projection
    PetscReal locVecMult_re, locVecMult_im;
    PetscReal VecMult_re, VecMult_im;
    for (PetscInt j = 0; j < p_col; j++) {
      // H_{j,k-1} = (q_j, Z_k)
      locVecMult_re = 0.0;
      locVecMult_im = 0.0;
      for (PetscInt row = 0; row < _dimension; row++) {
        locVecMult_re +=
            PetscRealPart(PetscConjComplex(_mat_Q[(_KS_size + 1) * row + j]) *
                          r_vec_to_orthogonalize[row]);
        locVecMult_im += PetscImaginaryPart(
            PetscConjComplex(_mat_Q[(_KS_size + 1) * row + j]) *
            r_vec_to_orthogonalize[row]);
      }

      MPI_Allreduce(&locVecMult_re, &VecMult_re, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
      MPI_Allreduce(&locVecMult_im, &VecMult_im, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);

      _mat_H[(_KS_size + 1) * j + p_col - 1] =
          VecMult_re + PETSC_i * VecMult_im;

      //  q_k = q_k - H_{j,k-1}q_j
      for (PetscInt row = 0; row < _dimension; row++)
        _mat_Q[(_KS_size + 1) * row + p_col] -=
            _mat_H[(_KS_size + 1) * j + p_col - 1] *
            _mat_Q[(_KS_size + 1) * row + j];
    }

    // H_{k,k-1} = ||q_k||
    PetscReal locVecMult = 0.0;
    PetscReal VecMult;
    for (PetscInt row = 0; row < _dimension; row++)
      locVecMult +=
          PetscRealPart(PetscConjComplex(_mat_Q[(_KS_size + 1) * row + p_col]) *
                        _mat_Q[(_KS_size + 1) * row + p_col]);

    MPI_Allreduce(&locVecMult, &VecMult, 1, MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
    _mat_H[(_KS_size + 1) * p_col + p_col - 1] = VecMult;
    _mat_H[(_KS_size + 1) * p_col + p_col - 1] =
        sqrt(_mat_H[(_KS_size + 1) * p_col + p_col - 1]);

    // End criteria
    if (PetscRealPart(_mat_H[(_KS_size + 1) * p_col + p_col - 1]) < eps) {
      if (_rank == 0)
        printf("Orthogonalization: stopped at k=%ld (h_{k,k-1}=%.16lf)\n",
               p_col,
               PetscRealPart(_mat_H[(_KS_size + 1) * p_col + p_col - 1]));
    } else {
      for (PetscInt row = 0; row < _dimension; row++)
        _mat_Q[(_KS_size + 1) * row + p_col] =
            _mat_Q[(_KS_size + 1) * row + p_col] /
            _mat_H[(_KS_size + 1) * p_col + p_col - 1];
    }
  }

  // Scale
  if (p_weighted == PETSC_TRUE) {
    for (PetscInt i = 0; i < _dimension; i++)
      r_vec_to_orthogonalize[i] =
          _mat_Q[(_KS_size + 1) * i + p_col] / p_weights[i];
  } else {
    for (PetscInt i = 0; i < _dimension; i++)
      r_vec_to_orthogonalize[i] = _mat_Q[(_KS_size + 1) * i + p_col];
  }
}

void Eigenproblem::Solve() {
  if (_complex == PETSC_TRUE)
    SolveComplex();
  else
    SolveReal();
}

void Eigenproblem::SolveReal() {
  if (_rank == 0) printf("Compute Eigenvalues and Eigenvectors ...\n");

  MatrixXd sH(_KS_size, _KS_size);

  for (PetscInt row = 0; row < _KS_size; row++)
    for (PetscInt col = 0; col < _KS_size; col++)
      sH(row, col) = PetscRealPart(_mat_H[(_KS_size + 1) * row + col]);

  _eigen_solver.computeFromTridiagonal(sH.diagonal(), sH.diagonal<1>());

  for (PetscInt i = 0; i < _KS_size; i++)
    _eigenvalue[i] = _eigen_solver.eigenvalues()(i);
}

void Eigenproblem::SolveComplex() {
  if (_rank == 0) printf("Compute Eigenvalues and Eigenvectors ...\n");

  MatrixXcd sH(_KS_size, _KS_size);

  for (PetscInt row = 0; row < _KS_size; row++)
    for (PetscInt col = 0; col < _KS_size; col++)
      sH(row, col) = _mat_H[(_KS_size + 1) * row + col];

  _eigen_solver_complex.compute(sH, true);

  for (PetscInt i = 0; i < _KS_size; i++)
    _eigenvalue[i] = _eigen_solver_complex.eigenvalues()(i);
}

void Eigenproblem::GetEigenVector(PetscInt p_idx, PetscBool p_weighted,
                                  PetscReal *p_weights,
                                  PetscScalar *r_eigenvector) {
  // eigenvector = Q*eigenvector_H
  if (_complex == PETSC_TRUE) {
    for (PetscInt row = 0; row < _dimension; row++) {
      r_eigenvector[row] = 0.0;
      for (PetscInt col = 0; col < _KS_size; col++)
        r_eigenvector[row] +=
            (PetscRealPart(_mat_Q[(_KS_size + 1) * row + col]) *
                 _eigen_solver_complex.eigenvectors().col(p_idx)(col).real() -
             PetscImaginaryPart(_mat_Q[(_KS_size + 1) * row + col]) *
                 _eigen_solver_complex.eigenvectors().col(p_idx)(col).imag()) +
            PETSC_i * (PetscRealPart(_mat_Q[(_KS_size + 1) * row + col]) *
                           _eigen_solver_complex.eigenvectors()
                               .col(p_idx)(col)
                               .imag() +
                       PetscImaginaryPart(_mat_Q[(_KS_size + 1) * row + col]) *
                           _eigen_solver_complex.eigenvectors()
                               .col(p_idx)(col)
                               .real());

      if (p_weighted == PETSC_TRUE)
        r_eigenvector[row] = r_eigenvector[row] / p_weights[row];
    }
  } else {
    for (PetscInt row = 0; row < _dimension; row++) {
      r_eigenvector[row] = 0.0;
      for (PetscInt col = 0; col < _KS_size; col++)
        r_eigenvector[row] += _mat_Q[(_KS_size + 1) * row + col] *
                              _eigen_solver.eigenvectors().col(p_idx)(col);

      if (p_weighted == PETSC_TRUE)
        r_eigenvector[row] = r_eigenvector[row] / p_weights[row];
    }
  }
}

void Eigenproblem::PrintEigenValues() {
  if (_rank == 0) {
    if (_complex == PETSC_TRUE) {
      cout << "The eigenvalues of H are: "
           << _eigen_solver_complex.eigenvalues().transpose() << endl;
      // cout << "Maximum eigenvalue of H is: " <<
      // _eigen_solver_complex.eigenvalues().maxCoeff(&idx) << endl; cout <<
      // "Corresponding eigenvector: " <<
      // _eigen_solver_complex.eigenvectors().col(idx).transpose() << endl;
    } else {
      PetscInt idx;
      cout << "The eigenvalues of H are: "
           << _eigen_solver.eigenvalues().transpose() << endl;
      cout << "Maximum eigenvalue of H is: "
           << _eigen_solver.eigenvalues().maxCoeff(&idx) << endl;
      cout << "Corresponding eigenvector: "
           << _eigen_solver.eigenvectors().col(idx).transpose() << endl;
    }
  }
}

void Eigenproblem::SaveEigenValues(char *p_output_folder) {
  if (_rank == 0) {
    char file[512];
    sprintf(file, "%s/Eigenvalues.txt", p_output_folder);
    FILE *fptr;
    fptr = fopen(file, "w");
    if (_complex == PETSC_TRUE) {
      for (PetscInt evIdx = 0; evIdx < _KS_size; evIdx++)
        fprintf(fptr, "%.20lf\t%.20lf\n",
                PetscRealPart(_eigen_solver_complex.eigenvalues()(evIdx)),
                PetscImaginaryPart(_eigen_solver_complex.eigenvalues()(evIdx)));
    } else {
      for (PetscInt evIdx = 0; evIdx < _KS_size; evIdx++)
        fprintf(fptr, "%.20lf\t", _eigen_solver.eigenvalues()(evIdx));
    }

    fprintf(fptr, "\n");
    fclose(fptr);
  }
}

void Eigenproblem::SaveMatrixHQMATLAB(char *p_output_folder, PetscInt p_mode) {
  SaveMatrixMATLAB(p_output_folder,
                   ((string)("H-" + to_string(p_mode))).c_str(), _KS_size + 1,
                   _KS_size + 1, _mat_H);
  SaveMatrixMATLAB(p_output_folder,
                   ((string)("Q-" + to_string(p_mode))).c_str(), _dimension,
                   (_KS_size + 1), _mat_Q);
}
void Eigenproblem::SaveMatrixTXT(char *p_output_folder, char *p_matrix_name,
                                 PetscInt p_rows, PetscInt p_columns,
                                 PetscReal *p_matrix) {
  if (_rank == 0) {
    char file[512];
    sprintf(file, "%s/%s.txt", p_output_folder, p_matrix_name);
    FILE *fptr;
    fptr = fopen(file, "w");

    for (int row = 0; row < p_rows; row++) {
      for (int col = 0; col < p_columns; col++)
        fprintf(fptr, "%.10lf\t", p_matrix[p_columns * row + col]);
      fprintf(fptr, "\n");
    }

    fclose(fptr);

    if (_rank == 0) printf("Matrix '%s' saved\n", p_matrix_name);
  }
}

void Eigenproblem::SaveMatrixMATLAB(char *p_output_folder,
                                    const char *p_matrix_name, PetscInt p_rows,
                                    PetscInt p_columns, PetscReal *p_matrix) {
  if (_rank == 0) {
    char file[512];
    sprintf(file, "%s/%s.m", p_output_folder, p_matrix_name);
    FILE *fptr;
    fptr = fopen(file, "w");

    fprintf(fptr, "%s=...\n[", p_matrix_name);
    for (int row = 0; row < p_rows; row++) {
      for (int col = 0; col < p_columns; col++)
        fprintf(fptr, "%.10lf\t", p_matrix[p_columns * row + col]);
      fprintf(fptr, ";\n");
    }
    fprintf(fptr, "];\n");

    fclose(fptr);

    if (_rank == 0) printf("Matrix '%s' saved\n", p_matrix_name);
  }
}
void Eigenproblem::SaveMatrixMATLAB(char *p_output_folder,
                                    const char *p_matrix_name, PetscInt p_rows,
                                    PetscInt p_columns, PetscScalar *p_matrix) {
  FILE *fptr;
  char file[512];
  sprintf(file, "%s/%s.m", p_output_folder, p_matrix_name);
  for (PetscInt t = 0; t < _size; t++) {
    if (_rank == t) {
      if (_rank == 0) {
        fptr = fopen(file, "w");
      } else {
        fptr = fopen(file, "a");
      }

      fprintf(fptr, "%s=...\n[", p_matrix_name);
      for (int row = 0; row < p_rows; row++) {
        for (int col = 0; col < p_columns; col++)
          fprintf(fptr, "%.20lf\t",
                  PetscRealPart(p_matrix[p_columns * row + col]));
        fprintf(fptr, ";\n");
      }
      fprintf(fptr, "];\n");

      fclose(fptr);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  if (_rank == 0) printf("Matrix '%s' saved\n", p_matrix_name);
}
