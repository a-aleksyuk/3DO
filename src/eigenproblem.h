#ifndef EIGENPROBLEM_H
#define EIGENPROBLEM_H

#include "structures.h"

using namespace Eigen;

class Eigenproblem {
 public:
  Eigenproblem();

  void Initialize(PetscInt p_KS_size, PetscInt p_dimension,
                  PetscBool p_complex);
  void Finalize();

  void ArnoldiUpdate(PetscInt p_col, PetscBool p_zero_pressure,
                     PetscBool p_weighted, PetscReal *p_weights,
                     PetscScalar *r_vec_to_orthogonalize);

  void Solve();
  void SolveReal();
  void SolveComplex();

  void GetEigenVector(PetscInt p_idx, PetscBool p_weighted,
                      PetscReal *p_weights, PetscScalar *r_eigenvector);

  void PrintEigenValues();
  void SaveEigenValues(char *p_output_folder);
  void SaveMatrixHQMATLAB(char *p_output_folder, PetscInt p_mode);

  void SaveMatrixTXT(char *p_output_folder, char *p_file_name, PetscInt p_rows,
                     PetscInt p_columns, PetscReal *p_matrix);
  void SaveMatrixMATLAB(char *p_output_folder, const char *p_file_name,
                        PetscInt p_rows, PetscInt p_columns,
                        PetscReal *p_matrix);
  void SaveMatrixMATLAB(char *p_output_folder, const char *p_file_name,
                        PetscInt p_rows, PetscInt p_columns,
                        PetscScalar *p_matrix);

 private:
  int _rank, _size;
  PetscBool _complex;
  PetscInt _dimension;
  PetscInt _KS_size;
  PetscScalar *_mat_H;
  PetscScalar *_mat_Q;

  SelfAdjointEigenSolver<MatrixXd> _eigen_solver;
  ComplexEigenSolver<MatrixXcd> _eigen_solver_complex;

 public:
  PetscScalar *_eigenvalue;
};

#endif  // EIGENPROBLEM_H
