#include "navierstokes.h"

#include <petsctime.h>

#include <iostream>

using namespace std;
using namespace O3D;

NavierStokes::NavierStokes(Input *p_input, Monitor *p_monitor) {
  MPI_Comm_rank(MPI_COMM_WORLD, &_Rank);

  _input = p_input;
  _monitor = p_monitor;
  _jacobi_matrix_alias_enabled = PETSC_FALSE;
}

NavierStokes::~NavierStokes() {}

PetscErrorCode NavierStokes::UpdateBaseFlow(void *p_user, PetscReal pT) {
  AppCtx *user = (AppCtx *)p_user;

  PetscReal nu, cos_nu_t, sin_nu_t;
  for (int i = 0; i < user->lN; i++) user->lX[i] = 0.0;

  for (PetscInt k = 0; k < _input->_base_flow_modes_count; k++) {
    nu = 2.0 * PETSC_PI * k / _input->_base_flow_period;
    if (k == 0) {
      cos_nu_t = cos(nu * pT);
      sin_nu_t = sin(nu * pT);
    } else {
      cos_nu_t = 2.0 * cos(nu * pT);
      sin_nu_t = 2.0 * sin(nu * pT);
    }

    for (int i = 0; i < user->lN; i++) {
      user->lX[i] +=
          _input->_base_flow_FS[_input->_base_flow_modes_count * i + k][0] *
              cos_nu_t -
          _input->_base_flow_FS[_input->_base_flow_modes_count * i + k][1] *
              sin_nu_t;
    }
  }

  for (int i = 0; i < user->lN; i++) user->lX0[i] = user->lX[i];

  return 0;
}

void NavierStokes::InitializeJacobianInterpolation(void *p_user) {
  // In and out vectors dimensions are equal
  int n[1];
  n[0] = _input->_base_flow_times_count;

  if (_Rank == 0) printf("InitializeMatrixInterpolation-Start\n");

  AppCtx *user = (AppCtx *)p_user;
  // ---------------------------------
  // Allocate memory
  PetscReal sizeMB = 0.0;

  PetscMalloc(n[0] * sizeof(PetscScalar), &_exp_t);
  if (_Rank == 0) {
    sizeMB += n[0] * sizeof(PetscScalar) / 1000000.0;
  }

  PetscInt N =
      (user->Nvlocal + user->NvGhost) * NVAR * NVAR * (user->NMaxAdj + 1);

  fftw_complex *ps_J =
      (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N * n[0]);
  if (_Rank == 0) {
    sizeMB += N * n[0] * sizeof(fftw_complex) / 1000000.0;
  }

  for (PetscInt i = 0; i < N * n[0]; i++) {
    ps_J[i][0] = 0.0;
    ps_J[i][1] = 0.0;
  }

  PetscMalloc(N * sizeof(PetscScalar), &(_jacobi_matrix_alias));
  if (_Rank == 0) {
    sizeMB += N * sizeof(PetscReal) / 1000000.0;
  }

  _fs_J = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N * n[0]);
  if (_Rank == 0) {
    sizeMB += N * n[0] * sizeof(fftw_complex) / 1000000.0;
  }

  _fs_J_backward = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * N * n[0]);
  if (_Rank == 0) {
    sizeMB += N * n[0] * sizeof(fftw_complex) / 1000000.0;
  }

  if (_Rank == 0) printf(" # MatrixInterpolation costs: %.2lfMB\n", sizeMB);

  int howmany, idist, odist;
  howmany = N;
  idist = n[0];
  odist = n[0];
  fftw_plan FFTW_Plan_P2F_J =
      fftw_plan_many_dft(1, n, howmany, ps_J, NULL, 1, idist, _fs_J, NULL, 1,
                         odist, FFTW_FORWARD, FFTW_ESTIMATE);
  fftw_plan FFTW_Plan_P2F_RHS =
      fftw_plan_many_dft(1, n, howmany, ps_J, NULL, 1, idist, _fs_J_backward,
                         NULL, 1, odist, FFTW_FORWARD, FFTW_ESTIMATE);

  // -------------------------------------------------------------
  // Set matrices in physical space
  user->FirstTimeStep = PETSC_FALSE;
  user->CurFM = 1;
  for (PetscInt t = 0; t < n[0]; t++) {
    if (_Rank == 0)
      printf("Set matrices J (primal) in physical space: %ld/%d\n", t + 1,
             n[0]);

    for (PetscInt i = 0; i < N; i++) _jacobi_matrix_alias[i] = 0.0;

    user->T = t * _input->_base_flow_period / n[0];
    UpdateBaseFlow(user, user->T);

    user->FreezeJac = PETSC_FALSE;
    _jacobi_matrix_alias_enabled = PETSC_TRUE;
    Function(NULL, user->GlobalX[user->CurFM], user->GlobalR, user);
    JacobianPrimal(NULL, user->GlobalX[user->CurFM], user->_mat_J_aux,
                   user->_mat_J_aux, user);
    _jacobi_matrix_alias_enabled = PETSC_FALSE;

    for (PetscInt c = 0; c < (user->Nvlocal + user->NvGhost); c++) {
      for (PetscInt c1 = 0; c1 < NVAR; c1++) {
        for (PetscInt r = 0; r < NVAR * (user->ITot[c] + 1); r++) {
          ps_J[n[0] * (NVAR * (user->Nvlocal + user->NvGhost) * r + NVAR * c +
                       c1) +
               t][0] =
              PetscRealPart(
                  _jacobi_matrix_alias[NVAR * (user->Nvlocal + user->NvGhost) *
                                           r +
                                       NVAR * c + c1]);
          ps_J[n[0] * (NVAR * (user->Nvlocal + user->NvGhost) * r + NVAR * c +
                       c1) +
               t][1] =
              PetscImaginaryPart(
                  _jacobi_matrix_alias[NVAR * (user->Nvlocal + user->NvGhost) *
                                           r +
                                       NVAR * c + c1]);
        }
      }
    }
  }

  // ---------------------------------
  // Fourier transform
  fftw_execute(FFTW_Plan_P2F_J);

  // normalisation
  for (PetscInt i = 0; i < N * n[0]; i++) {
    _fs_J[i][0] = _fs_J[i][0] / n[0];
    _fs_J[i][1] = _fs_J[i][1] / n[0];

    if (!isfinite(_fs_J[i][0]) || !isfinite(_fs_J[i][1])) {
      printf("NAN: _fs_J[%ld]\n", i);
      PressEnterToContinue();
    }
  }

  // -------------------------------------------------------------
  // Set matrices in physical space
  user->CurFM = 1;
  for (PetscInt t = 0; t < n[0]; t++) {
    if (_Rank == 0)
      printf("Set matrices J (adjoint) in physical space: %ld/%d\n", t + 1,
             n[0]);

    for (PetscInt i = 0; i < N; i++) _jacobi_matrix_alias[i] = 0.0;

    user->T = t * _input->_base_flow_period / n[0];

    UpdateBaseFlow(user, user->T);
    Function(NULL, user->GlobalX[user->CurFM], user->GlobalR, user);
    _jacobi_matrix_alias_enabled = PETSC_TRUE;
    user->BackwardsInTime = PETSC_TRUE;
    user->FreezeJac = PETSC_FALSE;
    JacobianPrimal(NULL, user->GlobalX[user->CurFM], user->_mat_J_aux,
                   user->_mat_J_aux, user);
    user->BackwardsInTime = PETSC_FALSE;
    _jacobi_matrix_alias_enabled = PETSC_FALSE;

    for (PetscInt c = 0; c < (user->Nvlocal + user->NvGhost); c++) {
      for (PetscInt c1 = 0; c1 < NVAR; c1++) {
        for (PetscInt r = 0; r < NVAR * (user->ITot[c] + 1); r++) {
          ps_J[n[0] * (NVAR * (user->Nvlocal + user->NvGhost) * r + NVAR * c +
                       c1) +
               t][0] =
              PetscRealPart(
                  _jacobi_matrix_alias[NVAR * (user->Nvlocal + user->NvGhost) *
                                           r +
                                       NVAR * c + c1]);
          ps_J[n[0] * (NVAR * (user->Nvlocal + user->NvGhost) * r + NVAR * c +
                       c1) +
               t][1] =
              PetscImaginaryPart(
                  _jacobi_matrix_alias[NVAR * (user->Nvlocal + user->NvGhost) *
                                           r +
                                       NVAR * c + c1]);
        }
      }
    }
  }

  // ---------------------------------
  // Fourier transform
  fftw_execute(FFTW_Plan_P2F_RHS);

  // normalisation
  for (PetscInt i = 0; i < N * n[0]; i++) {
    _fs_J_backward[i][0] = _fs_J_backward[i][0] / n[0];
    _fs_J_backward[i][1] = _fs_J_backward[i][1] / n[0];

    if (!isfinite(_fs_J_backward[i][0]) || !isfinite(_fs_J_backward[i][1])) {
      printf("NAN: _fs_J_backward[%ld]\n", i);
      PressEnterToContinue();
    }
  }

  // ---------------------------------
  // Free memory
  PetscFree(_jacobi_matrix_alias);

  fftw_destroy_plan(FFTW_Plan_P2F_J);
  fftw_destroy_plan(FFTW_Plan_P2F_RHS);
  fftw_free(ps_J);

  if (_Rank == 0) printf("InitializeMatrixInterpolation-End\n");
}

void NavierStokes::FinalizeJacobianInterpolation() {
  PetscFree(_exp_t);
  fftw_free(_fs_J);
  fftw_free(_fs_J_backward);
}

PetscErrorCode NavierStokes::JacobianNonZeros(SNES snes, Vec X, Mat J, Mat B,
                                              void *ptr) {
  AppCtx *user = (AppCtx *)ptr;
  PetscErrorCode ierr;
  PetscInt Nvlocal;

  Mat jac = B;

  Nvlocal = user->Nvlocal;

  PetscInt rows[user->NMaxAdj + 1];
  PetscScalar value1[NVAR * NVAR * (user->NMaxAdj + 1)];

  PetscInt c, r, c1, r1, nrows = 0;
  for (c = 0; c < Nvlocal; c++) {
    rows[0] = c;
    for (c1 = 0; c1 < NVAR; c1++) {
      // -------------------------------------------------------------------------
      // Блок с координатами (с, с)
      for (r1 = 0; r1 < NVAR; r1++) {
        if (_input->GetBCType(user->GloInd[c], r1, user->CurFM) != Dirichlet ||
            c1 == r1)
          value1[c1 + r1 * NVAR] = 1.0;
        else
          value1[c1 + r1 * NVAR] = 0.0;
      }
      // -------------------------------------------------------------------------
      // Блок с координатами (rows[nrows], с)
      nrows = 1;
      for (r = 0; r < user->ITot[c]; r++) {
        rows[nrows] = user->AdjM[c][r];

        for (r1 = 0; r1 < NVAR; r1++) {
          if (_input->GetBCType(user->GloInd[rows[nrows]], r1, user->CurFM) !=
              Dirichlet)
            value1[c1 + (r1 + nrows * NVAR) * NVAR] = 1.0;
          else
            value1[c1 + (r1 + nrows * NVAR) * NVAR] = 0.0;
        }
        nrows++;
      }
    }

    ierr = MatSetValuesBlockedLocal(jac, nrows, rows, 1, &c,
                                    (const PetscScalar *)value1, INSERT_VALUES);
    CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY);
  CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY);
  CHKERRQ(ierr);
  ierr = MatSetOption(jac, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE);
  CHKERRQ(ierr);
  return 0;
}

PetscErrorCode NavierStokes::Jacobian(SNES pSNES, Vec pX, Mat pJ, Mat pB,
                                      void *ptr) {
  if (_input->_jacobi_matrix_interpolation == true &&
      (((AppCtx *)ptr)->FirstTimeStep == PETSC_FALSE ||
       _input->_time_scheme_acceleration[((AppCtx *)ptr)->CurFM] == Euler))
    JacobianPrimalPeriodic(pSNES, pX, pJ, pB, ptr);
  else
    JacobianPrimal(pSNES, pX, pJ, pB, ptr);

  return 0;
}

PetscErrorCode NavierStokes::JacobianPrimalPeriodic(SNES pSNES, Vec pX, Mat pJ,
                                                    Mat pB, void *ptr) {
  _monitor->StartTimeMeasurement(EqJacobian);

  AppCtx *user = (AppCtx *)ptr;

  if (user->FreezeJac == PETSC_FALSE) {
    PetscInt idx, idx1;

    PetscReal nu;
    PetscBool BlockCC;

    PetscInt row_current;

    for (PetscInt k = 0; k < _input->_base_flow_times_count; k++) {
      nu = 2.0 * PETSC_PI * k / _input->_base_flow_period;
      _exp_t[k] = cos(nu * user->T) + PETSC_i * sin(nu * user->T);
    }

    _monitor->StartTimeMeasurement(EqJacobianLoop);
    for (PetscInt c = 0; c < user->Nvlocal; c++) {
      for (idx = 0; idx < (NVAR * NVAR * (user->NMaxAdj + 1)); idx++)
        _jValues[idx] = 0.0;

      for (PetscInt c1 = 0; c1 < NVAR; c1++) {
        row_current = 0;
        BlockCC = PETSC_FALSE;
        for (PetscInt r = 0; r < user->ITot[c]; r++) {
          // ----------------------------------------------------------------------------
          // Диагональный блок с координатами (с, с)
          if (BlockCC == PETSC_FALSE &&
              c < user->AdjM[c][r])  // Для хранения блоков по возрастанию
                                     // индекса строки
          {
            _jRows[row_current] = c;
            for (PetscInt r1 = 0; r1 < NVAR; r1++) {
              idx = c1 + (r1 + row_current * NVAR) * NVAR;
              idx1 = _input->_base_flow_times_count *
                     (NVAR * (user->Nvlocal + user->NvGhost) *
                          (r1 + row_current * NVAR) +
                      NVAR * c + c1);
              _jValues[idx] = 0.0;
              if (user->BackwardsInTime == PETSC_FALSE)
                for (PetscInt k = 0; k < _input->_base_flow_times_count; k++)
                  _jValues[idx] += ComplexMult(_fs_J[idx1 + k], _exp_t[k]);
              else
                for (PetscInt k = 0; k < _input->_base_flow_times_count; k++)
                  _jValues[idx] +=
                      ComplexMult(_fs_J_backward[idx1 + k], _exp_t[k]);
            }
            row_current++;
            BlockCC = PETSC_TRUE;
          }
          // ----------------------------------------------------------------------------
          // Недиагональный блок с координатами (_jRows[nrows], с)
          if (user->AdjM[c][r] < user->Nvlocal) {
            _jRows[row_current] = user->AdjM[c][r];
            for (PetscInt r1 = 0; r1 < NVAR; r1++) {
              idx = c1 + (r1 + row_current * NVAR) * NVAR;
              idx1 = _input->_base_flow_times_count *
                     (NVAR * (user->Nvlocal + user->NvGhost) *
                          (r1 + row_current * NVAR) +
                      NVAR * c + c1);
              _jValues[idx] = 0.0;
              if (user->BackwardsInTime == PETSC_FALSE)
                for (PetscInt k = 0; k < _input->_base_flow_times_count; k++)
                  _jValues[idx] += ComplexMult(_fs_J[idx1 + k], _exp_t[k]);
              else
                for (PetscInt k = 0; k < _input->_base_flow_times_count; k++)
                  _jValues[idx] +=
                      ComplexMult(_fs_J_backward[idx1 + k], _exp_t[k]);
            }
            row_current++;
          }
        }
        // ----------------------------------------------------------------------------
        // Диагональный блок с координатами (с, с)
        if (BlockCC ==
            PETSC_FALSE)  // Для хранения блоков по возрастанию индекса строки
        {
          _jRows[row_current] = c;
          for (PetscInt r1 = 0; r1 < NVAR; r1++) {
            idx = c1 + (r1 + row_current * NVAR) * NVAR;
            idx1 = _input->_base_flow_times_count *
                   (NVAR * (user->Nvlocal + user->NvGhost) *
                        (r1 + row_current * NVAR) +
                    NVAR * c + c1);
            _jValues[idx] = 0.0;
            if (user->BackwardsInTime == PETSC_FALSE)
              for (PetscInt k = 0; k < _input->_base_flow_times_count; k++)
                _jValues[idx] += ComplexMult(_fs_J[idx1 + k], _exp_t[k]);
            else
              for (PetscInt k = 0; k < _input->_base_flow_times_count; k++)
                _jValues[idx] +=
                    ComplexMult(_fs_J_backward[idx1 + k], _exp_t[k]);
          }
          row_current++;
        }
      }
      MatSetValuesBlockedLocal(pB, row_current, _jRows, 1, &c,
                               (const PetscScalar *)_jValues, INSERT_VALUES);
    }
    for (PetscInt c = user->Nvlocal; c < user->Nvlocal + user->NvGhost; c++) {
      for (PetscInt c1 = 0; c1 < NVAR; c1++) {
        row_current = 0;
        for (PetscInt r = 0; r < user->ITot[c]; r++) {
          // ----------------------------------------------------------------------------
          // Недиагональный блок с координатами (_jRows[nrows], с)
          if (user->AdjM[c][r] < user->Nvlocal) {
            _jRows[row_current] = user->AdjM[c][r];
            for (PetscInt r1 = 0; r1 < NVAR; r1++) {
              idx = c1 + (r1 + row_current * NVAR) * NVAR;
              idx1 = _input->_base_flow_times_count *
                     (NVAR * (user->Nvlocal + user->NvGhost) *
                          (r1 + row_current * NVAR) +
                      NVAR * c + c1);
              _jValues[idx] = 0.0;
              if (user->BackwardsInTime == PETSC_FALSE)
                for (PetscInt k = 0; k < _input->_base_flow_times_count; k++)
                  _jValues[idx] += ComplexMult(_fs_J[idx1 + k], _exp_t[k]);
              else
                for (PetscInt k = 0; k < _input->_base_flow_times_count; k++)
                  _jValues[idx] +=
                      ComplexMult(_fs_J_backward[idx1 + k], _exp_t[k]);
            }
            row_current++;
          }
        }
      }
      MatSetValuesBlockedLocal(pB, row_current, _jRows, 1, &c,
                               (const PetscScalar *)_jValues, INSERT_VALUES);
    }

    _monitor->EndTimeMeasurement(EqJacobianLoop);

    MatAssemblyBegin(pB, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(pB, MAT_FINAL_ASSEMBLY);
    MatSetOption(pB, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE);

    user->FreezeJac = PETSC_TRUE;

    if (_input->_debug_jacobi_save == PETSC_TRUE) {
      char fJ[512];
      sprintf(fJ, "%s/J-periodic-%g.m", _input->_Output, user->T);
      PetscViewer viewerJ;

      PetscViewerCreate(PETSC_COMM_WORLD, &viewerJ);
      PetscViewerSetType(viewerJ, PETSCVIEWERASCII);
      PetscViewerBinarySetSkipHeader(viewerJ, PETSC_TRUE);
      PetscViewerBinarySetSkipOptions(viewerJ, PETSC_TRUE);
      PetscViewerBinarySkipInfo(viewerJ);
      PetscViewerFileSetMode(viewerJ, FILE_MODE_WRITE);
      PetscViewerFileSetName(viewerJ, fJ);

      PetscViewerPushFormat(viewerJ, PETSC_VIEWER_ASCII_MATLAB);
      PetscObjectSetName((PetscObject)pB, "J");
      MatView(pB, viewerJ);
      PetscViewerPopFormat(viewerJ);

      PetscViewerDestroy(&viewerJ);

      if (_Rank == 0) PressEnterToContinue();
    }
  }

  _monitor->EndTimeMeasurement(EqJacobian);
  return 0;
}

PetscErrorCode NavierStokes::JacobianPrimal(SNES pSNES, Vec pX, Mat pJ, Mat pB,
                                            void *ptr) {
  _monitor->StartTimeMeasurement(EqJacobian);

  if (_input->_debug_enabled == PETSC_TRUE &&
      (_input->_debug_thread == _Rank || _input->_debug_thread == -1))
    printf("[%d]NavierStokes::Jacobian-Start\n", _Rank);

  AppCtx *user = (AppCtx *)ptr;

  PetscReal time_scheme_k;
  if (user->FirstTimeStep == PETSC_TRUE ||
      _input->_time_scheme_acceleration[user->CurFM] == Euler)
    time_scheme_k = user->BackwardsInTime == PETSC_FALSE ? 1.0 : -1.0;
  else
    time_scheme_k = user->BackwardsInTime == PETSC_FALSE ? 1.5 : -2.0;

  PetscScalar *x_t_ptr = user->lX_t;
  PetscScalar *x_ptr =
      user->BackwardsInTime == PETSC_FALSE ? user->lX : user->lX0;
  PetscErrorCode ierr;
  Mat jac = pB;
  if (_input->_debug_test != 0) {
    PetscInt c, r, c1, r1, k, nrows = 0;
    PetscBool BlockCC;
    for (c = 0; c < user->Nvlocal; c++) {
      // ====================================================================================
      // Расчет элементов матрицы для блок-столбца с
      for (c1 = 0; c1 < NVAR; c1++) {
        k = c * NVAR + c1;  // номер столбца
        nrows = 0;
        BlockCC = PETSC_FALSE;

        for (r = 0; r < user->ITot[c]; r++) {
          // ----------------------------------------------------------------------------
          // Диагональный блок с координатами (с, с)
          if (BlockCC == PETSC_FALSE && c < user->Nvlocal &&
              c < user->AdjM[c][r])  // Для хранения блоков по возрастанию
                                     // индекса строки
          {
            _jRows[nrows] = c;
            for (r1 = 0; r1 < NVAR; r1++) {
              if (user->BackwardsInTime == PETSC_TRUE) {
                if (c1 == r1 && c1 == 1 && c == 0)
                  _jValues[c1 + (r1 + nrows * NVAR) * NVAR] = -2.0;
                else if (c1 == r1)
                  _jValues[c1 + (r1 + nrows * NVAR) * NVAR] = -1.0;
                else
                  _jValues[c1 + (r1 + nrows * NVAR) * NVAR] = 0.0;
              } else if (c1 == r1) {
                _jValues[c1 + (r1 + nrows * NVAR) * NVAR] = 1.0;
              } else {
                _jValues[c1 + (r1 + nrows * NVAR) * NVAR] = 0.0;
              }
            }
            nrows++;
            BlockCC = PETSC_TRUE;
          }
          // ----------------------------------------------------------------------------
          // Недиагональный блок с координатами (_jRows[nrows], с)
          if (user->AdjM[c][r] < user->Nvlocal) {
            _jRows[nrows] = user->AdjM[c][r];
            for (r1 = 0; r1 < NVAR; r1++)
              _jValues[c1 + (r1 + nrows * NVAR) * NVAR] = 0.0;
            nrows++;
          }
        }
        // ----------------------------------------------------------------------------
        // Диагональный блок с координатами (с, с)
        if (BlockCC == PETSC_FALSE &&
            c < user->Nvlocal)  // Для хранения блоков по возрастанию индекса
                                // строки
        {
          _jRows[nrows] = c;
          for (r1 = 0; r1 < NVAR; r1++) {
            if (user->BackwardsInTime == PETSC_TRUE) {
              if (c1 == r1 && c1 == 1 && c == 0)
                _jValues[c1 + (r1 + nrows * NVAR) * NVAR] = -2.0;
              else if (c1 == r1)
                _jValues[c1 + (r1 + nrows * NVAR) * NVAR] = -1.0;
              else
                _jValues[c1 + (r1 + nrows * NVAR) * NVAR] = 0.0;
            } else if (c1 == r1) {
              _jValues[c1 + (r1 + nrows * NVAR) * NVAR] = 1.0;
            } else {
              _jValues[c1 + (r1 + nrows * NVAR) * NVAR] = 0.0;
            }
          }
          nrows++;
          BlockCC = PETSC_TRUE;
        }
      }
      // ====================================================================================
      // Задание блока матрицы Якоби
      ierr = MatSetValuesBlockedLocal(jac, nrows, _jRows, 1, &c,
                                      (const PetscScalar *)_jValues,
                                      INSERT_VALUES);
      CHKERRQ(ierr);
      if (_jacobi_matrix_alias_enabled == PETSC_TRUE) {
        for (c1 = 0; c1 < NVAR; c1++)
          for (int r = 0; r < nrows * NVAR; r++)
            _jacobi_matrix_alias[NVAR * (user->Nvlocal + user->NvGhost) * r +
                                 NVAR * c + c1] = _jValues[c1 + r * NVAR];
      }
    }

    for (c = user->Nvlocal; c < user->Nvlocal + user->NvGhost; c++) {
      // ====================================================================================
      // Расчет элементов матрицы для блок-столбца с
      for (c1 = 0; c1 < NVAR; c1++) {
        k = c * NVAR + c1;  // номер столбца
        nrows = 0;

        // ----------------------------------------------------------------------------
        // Если столбец соответствует граничному условию Дирихле, то в столбце
        // одна единица
        for (r = 0; r < user->ITot[c]; r++) {
          // ----------------------------------------------------------------------------
          // Недиагональный блок с координатами (_jRows[nrows], с)
          if (user->AdjM[c][r] < user->Nvlocal) {
            _jRows[nrows] = user->AdjM[c][r];
            for (r1 = 0; r1 < NVAR; r1++)
              _jValues[c1 + (r1 + nrows * NVAR) * NVAR] = 0.0;
            nrows++;
          }
        }
      }
      // ====================================================================================
      // Задание блока матрицы Якоби
      ierr = MatSetValuesBlockedLocal(jac, nrows, _jRows, 1, &c,
                                      (const PetscScalar *)_jValues,
                                      INSERT_VALUES);
      CHKERRQ(ierr);
      if (_jacobi_matrix_alias_enabled == PETSC_TRUE) {
        for (c1 = 0; c1 < NVAR; c1++)
          for (int r = 0; r < nrows * NVAR; r++)
            _jacobi_matrix_alias[NVAR * (user->Nvlocal + user->NvGhost) * r +
                                 NVAR * c + c1] = _jValues[c1 + r * NVAR];
      }
    }
    ierr = MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
    ierr = MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
    ierr = MatSetOption(jac, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE);
    CHKERRQ(ierr);
  } else if (user->FreezeJac == PETSC_FALSE) {
    PetscInt i, k, offset = user->lN * user->CurFM;

    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    // For linear system of equations
    PetscReal h = 10;  // 100*EPS_F_DERIVATIVE;
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    _monitor->StartTimeMeasurement(EqJacobianLoop);

    PetscInt c, r, c1, r1, nrows = 0;
    PetscBool BlockCC;
    for (c = 0; c < user->Nvlocal; c++) {
      // ====================================================================================
      // Save quadratures and aux parameters before adding deviation

      SaveAux(c, user);

      // ====================================================================================
      // Расчет элементов матрицы для блок-столбца с
      for (c1 = 0; c1 < NVAR; c1++) {
        k = c * NVAR + c1;  // номер столбца

        nrows = 0;
        BlockCC = PETSC_FALSE;
        // ----------------------------------------------------------------------------
        // Если столбец соответствует граничному условию Дирихле, то в столбце
        // одна единица
        if (_input->_solver_snes_dirichlet_conditions_new[user->CurFM] ==
                PETSC_TRUE &&
            (_input->_pressure_constant_node != c || c1 != 0) &&
            _input->GetBCType(user->GloInd[c], c1, user->CurFM) == Dirichlet) {
          for (r = 0; r < user->ITot[c]; r++) {
            // ----------------------------------------------------------------------------
            // Диагональный блок с координатами (с, с)
            if (BlockCC == PETSC_FALSE && c < user->Nvlocal &&
                c < user->AdjM[c][r])  // Для хранения блоков по возрастанию
                                       // индекса строки
            {
              _jRows[nrows] = c;
              for (r1 = 0; r1 < NVAR; r1++) {
                if (user->BackwardsInTime == PETSC_TRUE) {
                  _jValues[c1 + (r1 + nrows * NVAR) * NVAR] = 0.0;
                } else if (c1 == r1) {
                  _jValues[c1 + (r1 + nrows * NVAR) * NVAR] = 1.0;
                } else {
                  _jValues[c1 + (r1 + nrows * NVAR) * NVAR] = 0.0;
                }
              }
              nrows++;
              BlockCC = PETSC_TRUE;
            }
            // ----------------------------------------------------------------------------
            // Недиагональный блок с координатами (_jRows[nrows], с)
            if (user->AdjM[c][r] < user->Nvlocal) {
              _jRows[nrows] = user->AdjM[c][r];
              for (r1 = 0; r1 < NVAR; r1++)
                _jValues[c1 + (r1 + nrows * NVAR) * NVAR] = 0.0;
              nrows++;
            }
          }
          // ----------------------------------------------------------------------------
          // Диагональный блок с координатами (с, с)
          if (BlockCC == PETSC_FALSE &&
              c < user->Nvlocal)  // Для хранения блоков по возрастанию индекса
                                  // строки
          {
            _jRows[nrows] = c;
            for (r1 = 0; r1 < NVAR; r1++) {
              if (user->BackwardsInTime == PETSC_TRUE) {
                _jValues[c1 + (r1 + nrows * NVAR) * NVAR] = 0.0;
              } else if (c1 == r1) {
                _jValues[c1 + (r1 + nrows * NVAR) * NVAR] = 1.0;
              } else {
                _jValues[c1 + (r1 + nrows * NVAR) * NVAR] = 0.0;
              }
            }
            nrows++;
            BlockCC = PETSC_TRUE;
          }
        } else {
          // Argument variation
          x_t_ptr[offset + k] += time_scheme_k * h / _input->_dT;
          x_ptr[offset + k] += h;

          UpdateAuxVars(user, c);

          for (r = 0; r < user->ITot[c]; r++) {
            // ----------------------------------------------------------------------------
            // Диагональный блок с координатами (с, с)
            if (BlockCC == PETSC_FALSE && c < user->Nvlocal &&
                c < user->AdjM[c][r])  // Для хранения блоков по возрастанию
                                       // индекса строки
            {
              _jRows[nrows] = c;
              for (r1 = 0; r1 < NVAR; r1++) {
                if (_input->GetBCType(user->GloInd[c], r1, user->CurFM) !=
                    Dirichlet) {
                  i = _jRows[nrows] * NVAR + r1;
                  _aDF[i] =
                      WeakFormMain(i, user) - WeakFormDomainBorder(i, user);
                  _jValues[c1 + (r1 + nrows * NVAR) * NVAR] =
                      (_aDF[i] - _aF[i]) / h;

                  _jValues[c1 + (r1 + nrows * NVAR) * NVAR] *= _pScaleRows[i];
                } else if (user->BackwardsInTime == PETSC_TRUE) {
                  _jValues[c1 + (r1 + nrows * NVAR) * NVAR] = 0.0;
                } else if (c1 == r1)
                  _jValues[c1 + (r1 + nrows * NVAR) * NVAR] = 1.0;
                else
                  _jValues[c1 + (r1 + nrows * NVAR) * NVAR] = 0.0;
              }
              nrows++;
              BlockCC = PETSC_TRUE;
            }

            // ----------------------------------------------------------------------------
            // Недиагональный блок с координатами (_jRows[nrows], с)
            if (user->AdjM[c][r] < user->Nvlocal) {
              _jRows[nrows] = user->AdjM[c][r];
              for (r1 = 0; r1 < NVAR; r1++) {
                if (_input->GetBCType(user->GloInd[_jRows[nrows]], r1,
                                      user->CurFM) != Dirichlet) {
                  i = _jRows[nrows] * NVAR + r1;
                  _aDF[i] =
                      WeakFormMain(i, user) - WeakFormDomainBorder(i, user);
                  _jValues[c1 + (r1 + nrows * NVAR) * NVAR] =
                      (_aDF[i] - _aF[i]) / h;

                  _jValues[c1 + (r1 + nrows * NVAR) * NVAR] *= _pScaleRows[i];

                } else
                  _jValues[c1 + (r1 + nrows * NVAR) * NVAR] = 0.0;
              }
              nrows++;
            }
          }

          // ----------------------------------------------------------------------------
          // Диагональный блок с координатами (с, с)
          if (BlockCC == PETSC_FALSE &&
              c < user->Nvlocal)  // Для хранения блоков по возрастанию индекса
                                  // строки
          {
            _jRows[nrows] = c;
            for (r1 = 0; r1 < NVAR; r1++) {
              if (_input->GetBCType(user->GloInd[c], r1, user->CurFM) !=
                  Dirichlet) {
                i = _jRows[nrows] * NVAR + r1;
                _aDF[i] = WeakFormMain(i, user) - WeakFormDomainBorder(i, user);
                _jValues[c1 + (r1 + nrows * NVAR) * NVAR] =
                    (_aDF[i] - _aF[i]) / h;

                _jValues[c1 + (r1 + nrows * NVAR) * NVAR] *= _pScaleRows[i];
              } else if (user->BackwardsInTime == PETSC_TRUE) {
                _jValues[c1 + (r1 + nrows * NVAR) * NVAR] = 0.0;
              } else if (c1 == r1)
                _jValues[c1 + (r1 + nrows * NVAR) * NVAR] = 1.0;
              else
                _jValues[c1 + (r1 + nrows * NVAR) * NVAR] = 0.0;
            }
            nrows++;
            BlockCC = PETSC_TRUE;
          }

          x_t_ptr[offset + k] -= time_scheme_k * h / _input->_dT;
          x_ptr[offset + k] -= h;
        }
      }
      // ====================================================================================
      // Restore quadratures and aux parameters before adding deviation

      RestoreAux(c, user);
      // ====================================================================================
      // Задание блока матрицы Якоби
      ierr = MatSetValuesBlockedLocal(jac, nrows, _jRows, 1, &c,
                                      (const PetscScalar *)_jValues,
                                      INSERT_VALUES);
      CHKERRQ(ierr);

      if (_jacobi_matrix_alias_enabled == PETSC_TRUE) {
        for (c1 = 0; c1 < NVAR; c1++)
          for (int r = 0; r < nrows * NVAR; r++)
            _jacobi_matrix_alias[NVAR * (user->Nvlocal + user->NvGhost) * r +
                                 NVAR * c + c1] = _jValues[c1 + r * NVAR];
      }
    }
    for (c = user->Nvlocal; c < user->Nvlocal + user->NvGhost; c++) {
      // ====================================================================================
      // Save quadratures and aux parameters before adding deviation
      SaveAux(c, user);
      // ====================================================================================
      // Расчет элементов матрицы для блок-столбца с
      for (c1 = 0; c1 < NVAR; c1++) {
        k = c * NVAR + c1;  // номер столбца
        nrows = 0;

        // ----------------------------------------------------------------------------
        // Если столбец соответствует граничному условию Дирихле, то в столбце
        // одна единица
        if (_input->_solver_snes_dirichlet_conditions_new[user->CurFM] ==
                PETSC_TRUE &&
            (_input->_pressure_constant_node != c || c1 != 0) &&
            _input->GetBCType(user->GloInd[c], c1, user->CurFM) == Dirichlet) {
          for (r = 0; r < user->ITot[c]; r++) {
            // ----------------------------------------------------------------------------
            // Недиагональный блок с координатами (_jRows[nrows], с)
            if (user->AdjM[c][r] < user->Nvlocal) {
              _jRows[nrows] = user->AdjM[c][r];
              for (r1 = 0; r1 < NVAR; r1++)
                _jValues[c1 + (r1 + nrows * NVAR) * NVAR] = 0.0;
              nrows++;
            }
          }
        } else {
          // Вычисление шага h для расчета производной F
          x_t_ptr[offset + k] += time_scheme_k * h / _input->_dT;
          x_ptr[offset + k] += h;

          UpdateAuxVars(user, c);

          for (r = 0; r < user->ITot[c]; r++) {
            // ----------------------------------------------------------------------------
            // Недиагональный блок с координатами (_jRows[nrows], с)
            if (user->AdjM[c][r] < user->Nvlocal) {
              _jRows[nrows] = user->AdjM[c][r];
              for (r1 = 0; r1 < NVAR; r1++) {
                if (_input->GetBCType(user->GloInd[_jRows[nrows]], r1,
                                      user->CurFM) != Dirichlet) {
                  i = _jRows[nrows] * NVAR + r1;
                  _aDF[i] =
                      WeakFormMain(i, user) - WeakFormDomainBorder(i, user);
                  _jValues[c1 + (r1 + nrows * NVAR) * NVAR] =
                      (_aDF[i] - _aF[i]) / h;

                  _jValues[c1 + (r1 + nrows * NVAR) * NVAR] *= _pScaleRows[i];
                } else
                  _jValues[c1 + (r1 + nrows * NVAR) * NVAR] = 0.0;
              }
              nrows++;
            }
          }

          x_t_ptr[offset + k] -= time_scheme_k * h / _input->_dT;
          x_ptr[offset + k] -= h;
        }
      }

      // ====================================================================================
      // Restore quadratures and aux parameters before adding deviation
      RestoreAux(c, user);
      // ====================================================================================
      // Задание блока матрицы Якоби
      ierr = MatSetValuesBlockedLocal(jac, nrows, _jRows, 1, &c,
                                      (const PetscScalar *)_jValues,
                                      INSERT_VALUES);
      CHKERRQ(ierr);

      if (_jacobi_matrix_alias_enabled == PETSC_TRUE) {
        for (c1 = 0; c1 < NVAR; c1++)
          for (int r = 0; r < nrows * NVAR; r++)
            _jacobi_matrix_alias[NVAR * (user->Nvlocal + user->NvGhost) * r +
                                 NVAR * c + c1] = _jValues[c1 + r * NVAR];
      }
    }

    _monitor->EndTimeMeasurement(EqJacobianLoop);

    ierr = MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
    ierr = MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
    ierr = MatSetOption(jac, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE);
    CHKERRQ(ierr);

    user->FreezeJac = PETSC_TRUE;
  }

  if (_input->_debug_jacobi_save == PETSC_TRUE) {
    char fJ[512];
    char fX[512];
    sprintf(fJ, "%s/J-primal-%g.m", _input->_Output, user->T);
    sprintf(fX, "%s/X%g.m", _input->_Output, user->T);
    PetscViewer viewerJ, viewerX;

    PetscViewerCreate(PETSC_COMM_WORLD, &viewerJ);
    PetscViewerSetType(viewerJ, PETSCVIEWERASCII);
    PetscViewerBinarySetSkipHeader(viewerJ, PETSC_TRUE);
    PetscViewerBinarySetSkipOptions(viewerJ, PETSC_TRUE);
    PetscViewerBinarySkipInfo(viewerJ);
    PetscViewerFileSetMode(viewerJ, FILE_MODE_WRITE);
    PetscViewerFileSetName(viewerJ, fJ);

    PetscViewerCreate(PETSC_COMM_WORLD, &viewerX);
    PetscViewerSetType(viewerX, PETSCVIEWERASCII);
    PetscViewerBinarySetSkipHeader(viewerX, PETSC_TRUE);
    PetscViewerBinarySetSkipOptions(viewerX, PETSC_TRUE);
    PetscViewerBinarySkipInfo(viewerX);
    PetscViewerFileSetMode(viewerX, FILE_MODE_WRITE);
    PetscViewerFileSetName(viewerX, fX);

    PetscViewerPushFormat(viewerX, PETSC_VIEWER_ASCII_MATLAB);
    PetscObjectSetName((PetscObject)pX, "X");
    ierr = VecView(pX, viewerX);
    CHKERRQ(ierr);
    PetscViewerPopFormat(viewerX);

    PetscViewerPushFormat(viewerJ, PETSC_VIEWER_ASCII_MATLAB);
    PetscObjectSetName((PetscObject)jac, "J");
    ierr = MatView(jac, viewerJ);
    CHKERRQ(ierr);
    PetscViewerPopFormat(viewerJ);

    PetscViewerDestroy(&viewerJ);
    PetscViewerDestroy(&viewerX);

    if (_Rank == 0) PressEnterToContinue();
  }

  _monitor->EndTimeMeasurement(EqJacobian);

  if (_input->_debug_enabled == PETSC_TRUE &&
      (_input->_debug_thread == _Rank || _input->_debug_thread == -1))
    printf("[%d]NavierStokes::Jacobian-End\n", _Rank);

  return 0;
}

PetscErrorCode NavierStokes::JacobianAdjoint(SNES pSNES, Vec pX, Mat pJ, Mat pB,
                                             void *ptr) {
  _monitor->StartTimeMeasurement(EqJacobian);

  AppCtx *user = (AppCtx *)ptr;

  MatCopy(user->_matAH, pB, SAME_NONZERO_PATTERN);

  if (_input->_debug_jacobi_save == PETSC_TRUE) {
    char fJ[512];
    sprintf(fJ, "%s/JA%g.m", _input->_Output, user->T);
    PetscViewer viewerJ;

    PetscViewerCreate(PETSC_COMM_WORLD, &viewerJ);
    PetscViewerSetType(viewerJ, PETSCVIEWERASCII);
    PetscViewerBinarySetSkipHeader(viewerJ, PETSC_TRUE);
    PetscViewerBinarySetSkipOptions(viewerJ, PETSC_TRUE);
    PetscViewerBinarySkipInfo(viewerJ);
    PetscViewerFileSetMode(viewerJ, FILE_MODE_WRITE);
    PetscViewerFileSetName(viewerJ, fJ);

    PetscViewerPushFormat(viewerJ, PETSC_VIEWER_ASCII_MATLAB);
    PetscObjectSetName((PetscObject)user->_matAH, "A");
    MatView(user->_matAH, viewerJ);
    PetscViewerPopFormat(viewerJ);

    PetscViewerDestroy(&viewerJ);

    if (_Rank == 0) PressEnterToContinue();
  }

  _monitor->EndTimeMeasurement(EqJacobian);
  return 0;
}

PetscErrorCode NavierStokes::SaveAux(PetscInt pVIdx, void *ptr) {
  AppCtx *user = (AppCtx *)ptr;

  PetscInt ElIdx;

  for (PetscInt e = 0; e < user->NEByV[pVIdx]; e++) {
    ElIdx = user->EByV[pVIdx][e];
    for (PetscInt i = 0; i < NVAR; i++) {
      for (PetscInt k = 0; k < D1; k++)
        mW[D1_N * e + D1 * i + k] = _mW[D1_N * ElIdx + D1 * i + k];

      for (PetscInt d = 0; d < DIM; d++)
        quadWi[DIM * NVAR * e + DIM * i + d] =
            _quadWi[DIM * NVAR * ElIdx + DIM * i + d];
    }
  }
  for (PetscInt i = 0; i < NVAR; i++) {
    aC1[i] = _C1[NVAR * pVIdx + i];
  }

  return 0;
}

PetscErrorCode NavierStokes::RestoreAux(PetscInt pVIdx, void *ptr) {
  AppCtx *user = (AppCtx *)ptr;

  PetscInt ElIdx;

  for (PetscInt e = 0; e < user->NEByV[pVIdx]; e++) {
    ElIdx = user->EByV[pVIdx][e];
    for (PetscInt i = 0; i < NVAR; i++) {
      for (PetscInt k = 0; k < D1; k++)
        _mW[D1_N * ElIdx + D1 * i + k] = mW[D1_N * e + D1 * i + k];

      for (PetscInt d = 0; d < DIM; d++)
        _quadWi[DIM * NVAR * ElIdx + DIM * i + d] =
            quadWi[DIM * NVAR * e + DIM * i + d];
    }
  }
  for (PetscInt i = 0; i < NVAR; i++) {
    _C1[NVAR * pVIdx + i] = aC1[i];
  }
  return 0;
}

PetscErrorCode NavierStokes::Function(SNES snes, Vec X, Vec F, void *ptr) {
  _monitor->StartTimeMeasurement(EqFunction);

  if (_input->_debug_enabled == PETSC_TRUE &&
      (_input->_debug_thread == _Rank || _input->_debug_thread == -1))
    printf("[%d]NavierStokes::Function-Start\n", _Rank);

  AppCtx *user = (AppCtx *)ptr;
  PetscErrorCode ierr;
  PetscInt i;
  const PetscScalar *x;
  PetscScalar *f;

  // ===================================================================================================
  // Update variables of new layer
  // ---------------------------------------------------------------------------------------------------
  ierr = VecScatterBegin(user->Scatter, X, user->LocalX, INSERT_VALUES,
                         SCATTER_FORWARD);
  CHKERRQ(ierr);
  ierr = VecScatterEnd(user->Scatter, X, user->LocalX, INSERT_VALUES,
                       SCATTER_FORWARD);
  CHKERRQ(ierr);
  // ---------------------------------------------------------------------------------------------------

  ierr = VecGetArrayRead(user->LocalX, &x);
  CHKERRQ(ierr);

  for (int i = 0; i < user->lN; i++)
    user->lX[user->lN * user->CurFM + i] = x[i];

  if (_input->_time_scheme_acceleration[user->CurFM] == Euler ||
      user->FirstTimeStep == PETSC_TRUE) {
    for (int i = 0; i < user->lN; i++)
      user->lX_t[user->lN * user->CurFM + i] =
          (user->lX[user->lN * user->CurFM + i] -
           user->lX0[user->lN * user->CurFM + i]) /
          _input->_dT;
  } else if (_input->_time_scheme_acceleration[user->CurFM] == BDF2)  // Default
  {
    for (int i = 0; i < user->lN; i++)
      user->lX_t[user->lN * user->CurFM + i] =
          (3.0 * user->lX[user->lN * user->CurFM + i] -
           4.0 * user->lX0[user->lN * user->CurFM + i] +
           user->lX00[user->lN * user->CurFM + i]) /
          (2.0 * _input->_dT);
  }

  UpdateAuxVars(user, -1);

  ierr = VecGetArray(user->LocalF, &f);
  CHKERRQ(ierr);

  for (i = 0; i < NVAR * user->Nvlocal; i++) {
    if (_input->_debug_test == 0) {
      if (_input->GetBCType(user->GloInd[i / NVAR], i % NVAR, user->CurFM) !=
          Dirichlet) {
        _aF[i] = WeakFormMain(i, user) - WeakFormDomainBorder(i, user);
        f[i] = _pScaleRows[i] * _aF[i];
      } else {
        _aF[i] =
            x[i] - _input->GetBCValue(user->GloInd[i / NVAR], i % NVAR,
                                      user->T + _input->_dT, user->X1[i / NVAR],
                                      user->X2[i / NVAR], user->CurFM);
        f[i] = _aF[i];
      }
    } else if (_input->_debug_test == 1) {
      f[i] = x[i] - (i % 4) * user->lX0[user->lN * user->CurFM + i];
    } else if (_input->_debug_test == 2) {
      PetscReal a =
          0.1 + 2.0 * PETSC_PI / _input->_base_flow_period *
                    cos(2.0 * PETSC_PI * user->T / _input->_base_flow_period);
      f[i] = user->lX_t[user->lN * user->CurFM + i] -
             a * user->lX[user->lN * user->CurFM + i] -
             (i % 4) * user->lX0[user->lN * user->CurFM + i];
    }

    if (!isfinite(PetscRealPart(f[i])) || !isfinite(PetscImaginaryPart(f[i]))) {
      printf("NAN\n");
      PressEnterToContinue();
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }

  ierr = VecRestoreArrayRead(user->LocalX, &x);
  CHKERRQ(ierr);
  // ierr = VecRestoreArray(user->LocalX,&x);CHKERRQ(ierr);
  // ierr = VecRestoreArrayRead(user->LocalX0[user->CurFM],&x0);CHKERRQ(ierr);

  // ierr = VecRestoreArray(F, &f);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->LocalF, &f);
  CHKERRQ(ierr);
  ierr = VecScatterBegin(user->ScatterWithoutGhosts, user->LocalF, F,
                         INSERT_VALUES, SCATTER_REVERSE);
  CHKERRQ(ierr);
  ierr = VecScatterEnd(user->ScatterWithoutGhosts, user->LocalF, F,
                       INSERT_VALUES, SCATTER_REVERSE);
  CHKERRQ(ierr);

  // VecDestroy(&_LocalX_t);
  // VecDestroy(&_LocalY);
  // ierr = VecView(user->LocalX,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  // ierr = VecView(F,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  if (_input->_debug_jacobi_save == PETSC_TRUE) {
    if (_Rank == 0) printf("Monitor F\n");
    char fF[512];
    sprintf(fF, "%s/F-%g.m", _input->_Output, user->T);
    PetscViewer viewerF;

    PetscViewerCreate(PETSC_COMM_WORLD, &viewerF);
    PetscViewerSetType(viewerF, PETSCVIEWERASCII);
    PetscViewerBinarySetSkipHeader(viewerF, PETSC_TRUE);
    PetscViewerBinarySetSkipOptions(viewerF, PETSC_TRUE);
    PetscViewerBinarySkipInfo(viewerF);
    PetscViewerFileSetMode(viewerF, FILE_MODE_WRITE);
    PetscViewerFileSetName(viewerF, fF);

    PetscViewerPushFormat(viewerF, PETSC_VIEWER_ASCII_MATLAB);
    PetscObjectSetName((PetscObject)F, "F");
    ierr = VecView(F, viewerF);
    CHKERRQ(ierr);
    PetscViewerPopFormat(viewerF);

    PetscViewerDestroy(&viewerF);
    PressEnterToContinue();
  }

  if (_input->_debug_enabled == PETSC_TRUE &&
      (_input->_debug_thread == _Rank || _input->_debug_thread == -1))
    printf("[%d]NavierStokes::Function-End\n", _Rank);

  _monitor->EndTimeMeasurement(EqFunction);
  return 0;
}

PetscErrorCode NavierStokes::FunctionAdjoint(SNES snes, Vec X, Vec F,
                                             void *ptr) {
  PetscErrorCode ierr;

  _monitor->StartTimeMeasurement(EqFunction);

  if (_input->_debug_enabled == PETSC_TRUE &&
      (_input->_debug_thread == _Rank || _input->_debug_thread == -1))
    printf("[%d]NavierStokes::Function-Start\n", _Rank);

  AppCtx *user = (AppCtx *)ptr;

  if (user->InitialConditionsY == PETSC_TRUE) {
    // printf("F - initial conditions\n");
    // Initial conditions
    // F = A^H Y + R = 0
    MatMultAdd(user->_matAH, X, user->GlobalAux, F);
  } else {
    // F = A^H Y + B^H Y0 = 0 - Euler scheme
    // F = A^H Y + B^H Y0 + C^H = 0 - BDF2

    MatMult(user->_matAH, X, user->GlobalAux);
    MatMultAdd(user->_matBH, user->GlobalY0, user->GlobalAux, F);

    if (_input->_time_scheme_acceleration[1] == BDF2)
      VecAXPY(F, -0.25, user->GlobalBY0[user->CurFM]);  // C = -0.25B
  }

  if (_input->_debug_jacobi_save == PETSC_TRUE) {
    if (_Rank == 0) printf("Monitor F-adjoint\n");
    char fF[512];
    PetscViewer viewerF;

    sprintf(fF, "%s/F-adjoint-%g.m", _input->_Output, user->T);

    PetscViewerCreate(PETSC_COMM_WORLD, &viewerF);
    PetscViewerSetType(viewerF, PETSCVIEWERASCII);
    PetscViewerBinarySetSkipHeader(viewerF, PETSC_TRUE);
    PetscViewerBinarySetSkipOptions(viewerF, PETSC_TRUE);
    PetscViewerBinarySkipInfo(viewerF);
    PetscViewerFileSetMode(viewerF, FILE_MODE_WRITE);
    PetscViewerFileSetName(viewerF, fF);

    PetscViewerPushFormat(viewerF, PETSC_VIEWER_ASCII_MATLAB);
    PetscObjectSetName((PetscObject)F, "F");
    ierr = VecView(F, viewerF);
    CHKERRQ(ierr);
    PetscViewerPopFormat(viewerF);
    PetscViewerDestroy(&viewerF);

    // ------------------------------------------------------

    sprintf(fF, "%s/GlobalAux-adjoint-%g.m", _input->_Output, user->T);
    PetscViewerCreate(PETSC_COMM_WORLD, &viewerF);
    PetscViewerSetType(viewerF, PETSCVIEWERASCII);
    PetscViewerBinarySetSkipHeader(viewerF, PETSC_TRUE);
    PetscViewerBinarySetSkipOptions(viewerF, PETSC_TRUE);
    PetscViewerBinarySkipInfo(viewerF);
    PetscViewerFileSetMode(viewerF, FILE_MODE_WRITE);
    PetscViewerFileSetName(viewerF, fF);

    PetscViewerPushFormat(viewerF, PETSC_VIEWER_ASCII_MATLAB);
    PetscObjectSetName((PetscObject)user->GlobalAux, "GlobalAux");
    ierr = VecView(user->GlobalAux, viewerF);
    CHKERRQ(ierr);
    PetscViewerPopFormat(viewerF);
    PetscViewerDestroy(&viewerF);

    // ------------------------------------------------------
    if (user->InitialConditionsY == PETSC_FALSE) {
      sprintf(fF, "%s/GlobalY0-adjoint-%g.m", _input->_Output, user->T);
      PetscViewerCreate(PETSC_COMM_WORLD, &viewerF);
      PetscViewerSetType(viewerF, PETSCVIEWERASCII);
      PetscViewerBinarySetSkipHeader(viewerF, PETSC_TRUE);
      PetscViewerBinarySetSkipOptions(viewerF, PETSC_TRUE);
      PetscViewerBinarySkipInfo(viewerF);
      PetscViewerFileSetMode(viewerF, FILE_MODE_WRITE);
      PetscViewerFileSetName(viewerF, fF);

      PetscViewerPushFormat(viewerF, PETSC_VIEWER_ASCII_MATLAB);
      PetscObjectSetName((PetscObject)user->GlobalY0, "GlobalY0");
      ierr = VecView(user->GlobalY0, viewerF);
      CHKERRQ(ierr);
      PetscViewerPopFormat(viewerF);
      PetscViewerDestroy(&viewerF);
    }

    // ------------------------------------------------------

    sprintf(fF, "%s/_matAH-adjoint-%g.m", _input->_Output, user->T);
    PetscViewerCreate(PETSC_COMM_WORLD, &viewerF);
    PetscViewerSetType(viewerF, PETSCVIEWERASCII);
    PetscViewerBinarySetSkipHeader(viewerF, PETSC_TRUE);
    PetscViewerBinarySetSkipOptions(viewerF, PETSC_TRUE);
    PetscViewerBinarySkipInfo(viewerF);
    PetscViewerFileSetMode(viewerF, FILE_MODE_WRITE);
    PetscViewerFileSetName(viewerF, fF);

    PetscViewerPushFormat(viewerF, PETSC_VIEWER_ASCII_MATLAB);
    PetscObjectSetName((PetscObject)user->_matAH, "J");
    ierr = MatView(user->_matAH, viewerF);
    CHKERRQ(ierr);
    PetscViewerPopFormat(viewerF);
    PetscViewerDestroy(&viewerF);

    // ------------------------------------------------------
    if (user->InitialConditionsY == PETSC_FALSE) {
      sprintf(fF, "%s/_matBH-adjoint-%g.m", _input->_Output, user->T);
      PetscViewerCreate(PETSC_COMM_WORLD, &viewerF);
      PetscViewerSetType(viewerF, PETSCVIEWERASCII);
      PetscViewerBinarySetSkipHeader(viewerF, PETSC_TRUE);
      PetscViewerBinarySetSkipOptions(viewerF, PETSC_TRUE);
      PetscViewerBinarySkipInfo(viewerF);
      PetscViewerFileSetMode(viewerF, FILE_MODE_WRITE);
      PetscViewerFileSetName(viewerF, fF);

      PetscViewerPushFormat(viewerF, PETSC_VIEWER_ASCII_MATLAB);
      PetscObjectSetName((PetscObject)user->_matBH, "J");
      ierr = MatView(user->_matBH, viewerF);
      CHKERRQ(ierr);
      PetscViewerPopFormat(viewerF);
      PetscViewerDestroy(&viewerF);
    }

    PressEnterToContinue();
  }

  if (_input->_debug_enabled == PETSC_TRUE &&
      (_input->_debug_thread == _Rank || _input->_debug_thread == -1))
    printf("[%d]NavierStokes::Function-End\n", _Rank);

  _monitor->EndTimeMeasurement(EqFunction);
  return 0;
}

PetscErrorCode NavierStokes::InitialConditions(PetscInt modeIdx, Vec X,
                                               void *ptr) {
  AppCtx *user = (AppCtx *)ptr;
  PetscInt i, Nvlocal, ierr;
  PetscScalar *x;

  Nvlocal = user->Nvlocal;

  ierr = VecGetArray(X, &x);
  CHKERRQ(ierr);

  srand(time(NULL));
  if (_input->_debug_test == 0) {
    for (i = 0; i < NVAR * Nvlocal; i++) {
      if (_input->GetBCType(user->GloInd[i / NVAR], i % NVAR, modeIdx) ==
          Dirichlet)
        x[i] = _input->GetBCValue(user->GloInd[i / NVAR], i % NVAR, user->T,
                                  user->X1[i / NVAR], user->X2[i / NVAR],
                                  modeIdx);  // Учет граничных условий
      else
        x[i] = _input->GetXInitial(user->GloInd[i / NVAR], i % NVAR,
                                   user->X1[i / NVAR], user->X2[i / NVAR],
                                   modeIdx);

      if (i % NVAR == 0) x[i] = _pScale * x[i];
    }
  } else {
    for (i = 0; i < NVAR * Nvlocal; i++)
      x[i] = 2.0 * ((PetscReal)rand()) / ((PetscReal)RAND_MAX) - 1.0;  // 1;
  }

  ierr = VecRestoreArray(X, &x);
  CHKERRQ(ierr);

  return 0;
}

PetscErrorCode NavierStokes::InitAuxVars(void *ptr) {
  if (_input->_debug_enabled == PETSC_TRUE &&
      (_input->_debug_thread == _Rank || _input->_debug_thread == -1))
    printf("[%d]NavierStokes::InitAuxVars-Start\n", _Rank);

  AppCtx *user = (AppCtx *)ptr;

  PetscMPIInt rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  PetscErrorCode ierr;
  PetscInt Nvg = (user->Nvlocal + user->NvGhost);
  PetscReal sizeMB = 0.0;

  ierr = PetscMalloc(NVAR * user->Nvlocal * sizeof(PetscScalar), &_aF);
  CHKERRQ(ierr);
  if (rank == 0) {
    sizeMB += NVAR * user->Nvlocal * sizeof(PetscScalar) / 1000000.0;
  }
  ierr = PetscMalloc(NVAR * user->Nvlocal * sizeof(PetscScalar), &_aDF);
  CHKERRQ(ierr);
  if (rank == 0) {
    sizeMB += NVAR * user->Nvlocal * sizeof(PetscScalar) / 1000000.0;
  }
  ierr = PetscMalloc((user->NMaxAdj + 1) * sizeof(PetscInt), &_jRows);
  CHKERRQ(ierr);
  if (rank == 0) {
    sizeMB += (user->NMaxAdj + 1) * sizeof(PetscInt) / 1000000.0;
  }
  ierr = PetscMalloc((NVAR * NVAR * (user->NMaxAdj + 1)) * sizeof(PetscScalar),
                     &_jValues);
  CHKERRQ(ierr);
  if (rank == 0) {
    sizeMB +=
        (NVAR * NVAR * (user->NMaxAdj + 1)) * sizeof(PetscScalar) / 1000000.0;
  }
  ierr = PetscMalloc(Nvg * NVAR * sizeof(PetscScalar), &(_C1));
  CHKERRQ(ierr);
  if (rank == 0) {
    sizeMB += Nvg * NVAR * sizeof(PetscScalar) / 1000000.0;
  }
  ierr = PetscMalloc(NVAR * sizeof(PetscScalar), &(aC1));
  CHKERRQ(ierr);
  if (rank == 0) {
    sizeMB += NVAR * sizeof(PetscScalar) / 1000000.0;
  }

  // Функции на элементах
  ierr = PetscMalloc(user->Nelocal * D1_N * sizeof(PetscScalar), &(_mW));
  CHKERRQ(ierr);
  if (rank == 0) {
    sizeMB += user->Nelocal * D1_N * sizeof(PetscScalar) / 1000000.0;
  }
  ierr = PetscMalloc(D1_N * DIM * sizeof(PetscScalar), &(_mWi));
  CHKERRQ(ierr);
  if (rank == 0) {
    sizeMB += D1_N * DIM * sizeof(PetscScalar) / 1000000.0;
  }
  ierr = PetscMalloc(D1_N * sizeof(PetscScalar), &(_mWGLS));
  CHKERRQ(ierr);
  if (rank == 0) {
    sizeMB += D1_N * sizeof(PetscScalar) / 1000000.0;
  }
  ierr = PetscMalloc(D1_N * sizeof(PetscScalar), &(_mY_baseflow));
  CHKERRQ(ierr);
  if (rank == 0) {
    sizeMB += D1_N * sizeof(PetscScalar) / 1000000.0;
  }
  ierr =
      PetscMalloc(_input->_Mesh._MaxEAdj * D1_N * sizeof(PetscScalar), &(mW));
  CHKERRQ(ierr);
  if (rank == 0) {
    sizeMB += _input->_Mesh._MaxEAdj * D1_N * sizeof(PetscScalar) / 1000000.0;
  }
  ierr = PetscMalloc(_input->_Mesh._MaxEAdj * D1_N * DIM * sizeof(PetscScalar),
                     &(mWi));
  CHKERRQ(ierr);
  if (rank == 0) {
    sizeMB +=
        _input->_Mesh._MaxEAdj * D1_N * DIM * sizeof(PetscScalar) / 1000000.0;
  }
  ierr = PetscMalloc(_input->_Mesh._MaxEAdj * D1_N * sizeof(PetscScalar),
                     &(mWGLS));
  CHKERRQ(ierr);
  if (rank == 0) {
    sizeMB += _input->_Mesh._MaxEAdj * D1_N * sizeof(PetscScalar) / 1000000.0;
  }

  ierr = PetscMalloc(user->Nelocal * sizeof(PetscReal), &(_eH));
  CHKERRQ(ierr);
  if (rank == 0) {
    sizeMB += user->Nelocal * sizeof(PetscReal) / 1000000.0;
  }

  for (PetscInt e = 0; e < user->Nelocal; e++)
    _eH[e] = PetscSqrtReal(2.0 * user->cell_volume[e]);

  // Квадратуры
  ierr =
      PetscMalloc(user->Nelocal * NVAR * DIM * sizeof(PetscScalar), &(_quadWi));
  CHKERRQ(ierr);
  if (rank == 0) {
    sizeMB += user->Nelocal * NVAR * DIM * sizeof(PetscScalar) / 1000000.0;
  }
  ierr = PetscMalloc(_input->_Mesh._MaxEAdj * NVAR * DIM * sizeof(PetscScalar),
                     &(quadWi));
  CHKERRQ(ierr);
  if (rank == 0) {
    sizeMB +=
        _input->_Mesh._MaxEAdj * NVAR * DIM * sizeof(PetscScalar) / 1000000.0;
  }
  ierr = PetscMalloc(D1 * sizeof(PetscScalar *), &(_uY_x));
  CHKERRQ(ierr);
  if (rank == 0) {
    sizeMB += D1 * NVAR * sizeof(PetscScalar) / 1000000.0;
  }
  ierr = PetscMalloc(D1 * sizeof(PetscScalar *), &(_uY_x_base));
  CHKERRQ(ierr);
  if (rank == 0) {
    sizeMB += D1 * NVAR * sizeof(PetscScalar) / 1000000.0;
  }
  for (int i = 0; i < D1; i++) {
    ierr = PetscMalloc(NVAR * sizeof(PetscScalar), &(_uY_x[i]));
    CHKERRQ(ierr);
    ierr = PetscMalloc(NVAR * sizeof(PetscScalar), &(_uY_x_base[i]));
    CHKERRQ(ierr);
  }

  // ===================================================================
  // Initialize aux arrays for nonlinear term calculations

  _fs_U = (fftw_complex *)fftw_malloc(
      sizeof(fftw_complex) * (user->Nvlocal + user->NvGhost) * D1 * user->FM);
  if (rank == 0) {
    sizeMB += (user->Nvlocal + user->NvGhost) * D1 * user->FM *
              sizeof(PetscScalar) / 1000000.0;
  }
  _fs_Uz = (fftw_complex *)fftw_malloc(
      sizeof(fftw_complex) * (user->Nvlocal + user->NvGhost) * D1 * user->FM);
  if (rank == 0) {
    sizeMB += (user->Nvlocal + user->NvGhost) * D1 * user->FM *
              sizeof(PetscScalar) / 1000000.0;
  }
  ierr = PetscMalloc(
      (user->Nvlocal + user->NvGhost) * D1 * user->NZ * sizeof(PetscReal),
      &(_ps_U));
  CHKERRQ(ierr);
  if (rank == 0) {
    sizeMB += (user->Nvlocal + user->NvGhost) * D1 * user->NZ *
              sizeof(PetscReal) / 1000000.0;
  }
  ierr = PetscMalloc(
      (user->Nvlocal + user->NvGhost) * D1 * user->NZ * sizeof(PetscReal),
      &(_ps_Uz));
  CHKERRQ(ierr);
  if (rank == 0) {
    sizeMB += (user->Nvlocal + user->NvGhost) * D1 * user->NZ *
              sizeof(PetscReal) / 1000000.0;
  }
  for (int i = 0; i < DIM; i++) {
    ierr = PetscMalloc(user->Nelocal * D1 * user->NZ * sizeof(PetscReal),
                       &(_ps_U_x[i]));
    CHKERRQ(ierr);
    if (rank == 0) {
      sizeMB += user->Nelocal * D1 * user->NZ * sizeof(PetscReal) / 1000000.0;
    }
  }

  if (_input->_time_scheme_nonlinear_term_extrapolation_f == PETSC_TRUE) {
    _fs_N0 = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * user->Nelocal *
                                         D1_D1 * user->FM);
    if (rank == 0) {
      sizeMB +=
          user->Nelocal * D1_D1 * user->FM * sizeof(PetscScalar) / 1000000.0;
    }
    _fs_N0_initialized = PETSC_FALSE;
  }

  _fs_N = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * user->Nelocal *
                                      D1_D1 * user->FM);
  if (rank == 0) {
    sizeMB +=
        user->Nelocal * D1_D1 * user->FM * sizeof(PetscScalar) / 1000000.0;
  }
  ierr = PetscMalloc(user->Nelocal * D1_D1 * user->NZ * sizeof(PetscReal),
                     &(_ps_N));
  CHKERRQ(ierr);
  if (rank == 0) {
    sizeMB += user->Nelocal * D1_D1 * user->NZ * sizeof(PetscReal) / 1000000.0;
  }

  // ===================================================================
  // FFTW parameters

  int *inembed = NULL;
  int *onembed = NULL;
  int istride = 1;
  int ostride = 1;
  int n[1];
  n[0] = user->NZ;
  int howmany;

  int idist, odist;

  howmany = (user->Nvlocal + user->NvGhost) * D1;
  idist = user->FM;
  odist = user->NZ;
  _FFTW_PlanC2R_U =
      fftw_plan_many_dft_c2r(1, n, howmany, _fs_U, inembed, istride, idist,
                             _ps_U, onembed, ostride, odist, FFTW_ESTIMATE);

  _FFTW_PlanC2R_Uz =
      fftw_plan_many_dft_c2r(1, n, howmany, _fs_Uz, inembed, istride, idist,
                             _ps_Uz, onembed, ostride, odist, FFTW_ESTIMATE);

  howmany = user->Nelocal * D1_D1;
  idist = user->NZ;
  odist = user->FM;
  _FFTW_PlanR2C_N =
      fftw_plan_many_dft_r2c(1, n, howmany, (double *)_ps_N, NULL, 1, idist,
                             _fs_N, NULL, 1, odist, FFTW_ESTIMATE);

  if (rank == 0) {
    printf(" # InitAuxVars: %.2lfMB\n", sizeMB);
  }

  ierr = PetscMalloc((user->Nvlocal + user->NvGhost) * NVAR * sizeof(PetscReal),
                     &(_pScaleRows));
  CHKERRQ(ierr);
  if (rank == 0) {
    sizeMB += (user->Nvlocal + user->NvGhost) * sizeof(PetscReal) / 1000000.0;
  }

  PetscReal nodeH;
  PetscReal tau0, tau1, tau2;

  PetscReal sumH_local = 0.0;
  PetscReal eps = 0.000001;
  for (PetscInt vtx = 0; vtx < user->Nvlocal + user->NvGhost; vtx++) {
    nodeH = 0.0;
    for (PetscInt e = 0; e < user->NEByV[vtx]; e++)
      nodeH += PetscSqrtReal(2.0 * user->cell_volume[user->EByV[vtx][e]]);

    nodeH = nodeH / user->NEByV[vtx];

    if (vtx < user->Nvlocal) sumH_local += nodeH;

    if (_input->_solver_fem_scale_momentum_type == "C") {
      for (PetscInt eq = 1; eq < NVAR; eq++)
        _pScaleRows[NVAR * vtx + eq] = _input->_solver_fem_scale_momentum;
    } else if (_input->_solver_fem_scale_momentum_type == "C*dT/h^2") {
      for (PetscInt eq = 1; eq < NVAR; eq++)
        _pScaleRows[NVAR * vtx + eq] =
            _input->_solver_fem_scale_momentum * _input->_dT / (nodeH * nodeH);
    } else {
      for (PetscInt eq = 1; eq < NVAR; eq++) _pScaleRows[NVAR * vtx + eq] = 1.0;
    }

    if (_input->_solver_fem_scale_mass_type == "C")
      _pScaleRows[NVAR * vtx + 0] = _input->_solver_fem_scale_mass;
    else if (_input->_solver_fem_scale_mass_type == "C/h")
      _pScaleRows[NVAR * vtx + 0] = _input->_solver_fem_scale_mass / nodeH;
    else if (_input->_solver_fem_scale_mass_type == "C/dT")
      _pScaleRows[NVAR * vtx + 0] =
          _input->_solver_fem_scale_mass / _input->_dT;
    else if (_input->_solver_fem_scale_mass_type == "C*tau") {
      tau0 = 2.0 / _input->_dT;
      tau0 = tau0 * tau0;
      tau1 = 2.0 * 1.0 / nodeH;
      tau1 = tau1 * tau1;
      tau2 = 4.0 / (_input->_Re * nodeH * nodeH);
      tau2 = tau2 * tau2;

      _pScaleRows[NVAR * vtx + 0] =
          _input->_solver_fem_scale_mass * PetscSqrtReal(tau0 + tau1 + tau2);
    } else
      _pScaleRows[NVAR * vtx + 0] = 1.0;
  }
  PetscReal avgH;
  MPI_Allreduce(&sumH_local, &avgH, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  avgH = avgH / user->Nvglobal;

  // printf("\tavgH: %lf\n", avgH);
  if (_input->_solver_fem_scale_pressure_type == "C")
    _pScale = _input->_solver_fem_scale_pressure;
  else if (_input->_solver_fem_scale_pressure_type == "C/dT")
    _pScale = _input->_solver_fem_scale_pressure / _input->_dT;
  else if (_input->_solver_fem_scale_pressure_type == "C*h/dT")
    _pScale = _input->_solver_fem_scale_pressure * avgH / _input->_dT;
  else if (_input->_solver_fem_scale_pressure_type == "C/dT")
    _pScale = _input->_solver_fem_scale_pressure / _input->_dT;
  else
    _pScale = 1.0;

  if (_Rank == 0) {
    printf("\tP scale: %lf\n", _pScale);
  }

  if (_input->_debug_enabled == PETSC_TRUE &&
      (_input->_debug_thread == _Rank || _input->_debug_thread == -1))
    printf("[%d]NavierStokes::InitAuxVars-End\n", _Rank);
  return (0);
}

PetscErrorCode NavierStokes::UpdateAuxVars(AppCtx *user, PetscInt pVrtx) {
  if (_input->_problem_type == "TSA" || _input->_problem_type == "LSA" ||
      _input->_problem_type == "Floquet")
    UpdateAuxVars_Real(user, pVrtx);
  else if (_input->_problem_type == "DNS")
    UpdateAuxVars_Complex(user, pVrtx);

  return (0);
}

PetscErrorCode NavierStokes::UpdateAuxVars_Complex(AppCtx *user,
                                                   PetscInt pVrtx) {
  if (_input->_debug_enabled == PETSC_TRUE && pVrtx == -1 &&
      (_input->_debug_thread == _Rank || _input->_debug_thread == -1))
    printf("[%d]NavierStokes::UpdateAuxVars(%ld)-Start\n", _Rank, pVrtx);

  _monitor->StartTimeMeasurement(EqUpdateAuxVars);

  bool test = false;

  PetscInt idx0, idx1;
  PetscInt ElIdx;

  PetscInt N;
  PetscInt vtx1, vtx2;
  PetscInt offset = user->lN * user->CurFM;

  PetscScalar auxQuad;
  PetscReal tauPSPG, tauSUPG, tauLSIC;

  PetscScalar beta = 2.0 * PETSC_PI * user->CurFM / _input->_L;

  // Update all data (-1) or neighbourhood of one vertex (with number pVrtx)
  if (pVrtx == -1) {
    N = user->Nelocal;
    vtx1 = 0;
    vtx2 = user->Nvlocal + user->NvGhost;
  } else {
    N = user->NEByV[pVrtx];
    vtx1 = pVrtx;
    vtx2 = pVrtx + 1;
  }

  //==============================================================
  // Every node
  for (PetscInt vtx = vtx1; vtx < vtx2; vtx++) {
    idx0 = NVAR * vtx;
    // ---------------------------------------------------------
    // Time derrivative + Forces
    // \vec{C1}=\vec{Y}_{,t}-\vec{R}
    _C1[idx0] = 0.0;

    for (PetscInt eq = 1; eq < NVAR; eq++)
      _C1[idx0 + eq] =
          user->lX_t[offset + idx0 + eq] -
          _input->GetForceValue(eq - 1, user->T + 0.5 * _input->_dT,
                                user->X1[vtx], user->X2[vtx], user->CurFM) /
              _input->_Fr;
  }

  //==============================================================
  // Every element
  for (PetscInt e = 0; e < N; e++) {
    if (pVrtx != -1)
      ElIdx = user->EByV[pVrtx][e];
    else
      ElIdx = e;

    // --------------------------------------------------------------------
    // _uY_x
    // x-, y- derivatives \vec{Y}_{,x} and \vec{Y}_{,y}
    for (PetscInt dir = 0; dir < DIM; dir++)
      Element::Y_x(dir, ElIdx, user->Hx, &(user->lX[offset]), user->VByE[ElIdx],
                   _uY_x[dir]);

    if (_input->_problem_type == "Floquet") {
      for (PetscInt dir = 0; dir < DIM; dir++)
        Element::Y_x(dir, ElIdx, user->Hx, &(user->lX[0]), user->VByE[ElIdx],
                     _uY_x_base[dir]);
    }

    // ---------------------------------------------------------
    // Stabilisation parameters (depends only on base flow)
    if (_input->_time_scheme_stabilisation[user->CurFM] ==
        ArgumentExtrapolation)
      GetTau(&(user->lX1_e[0]), ElIdx, user, &tauPSPG, &tauSUPG, &tauLSIC);
    else if (_input->_time_scheme_stabilisation[user->CurFM] == Implicit)
      GetTau(&(user->lX[0]), ElIdx, user, &tauPSPG, &tauSUPG, &tauLSIC);
    else  // Explicit
      GetTau(&(user->lX0[0]), ElIdx, user, &tauPSPG, &tauSUPG, &tauLSIC);

    tauLSIC = -tauLSIC;  // '-' before tauLSIC due to negative term
                         // \nabla\cdot\vec{u} (see below)
    tauPSPG = -tauPSPG;  // '-' before tauLSIC due to negative term
                         // \nabla\cdot\vec{u} (see below)

    // ---------------------------------------------------------
    for (PetscInt k = 0; k < D1; k++) {
      idx0 = NVAR * user->VByE[ElIdx][k];
      idx1 = D1_N * ElIdx + k;

      // z-derivative \vec{Y}_{,z} - depends on k
      for (PetscInt var = 0; var < NVAR; var++)
        _uY_x[2][var] = PETSC_i * beta * user->lX[offset + idx0 + var];

      for (PetscInt i = 0; i < NVAR; i++) {
        if (_input->_time_scheme_stabilisation[user->CurFM] ==
            ArgumentExtrapolation)
          _mY_baseflow[D1 * i + k] = user->lX1_e[0 + idx0 + i];
        else if (_input->_time_scheme_stabilisation[user->CurFM] == Implicit)
          _mY_baseflow[D1 * i + k] = user->lX[0 + idx0 + i];
        else  // Explicit
          _mY_baseflow[D1 * i + k] = user->lX[0 + idx0 + i];
      }

      // ---------------------------------------------------------
      // _mW=\vec{Y}_{,t}+A_i\vec{Y}_{,i}-\vec{R}

      // _mW = \vec{Y}_{,t}-\vec{R}
      for (PetscInt i = 0; i < NVAR; i++) _mW[idx1 + D1 * i] = _C1[idx0 + i];

      // Conservation of mass: -(u_{,x}+v_{,y}+w_{,z}) '-' - for symmetry
      _mW[idx1 + D1 * 0] += -(_uY_x[0][1] + _uY_x[1][2] + _uY_x[2][3]);

      // Conservation of momentum. Nonlinear term:
      // u\vec{u}_{,x}+v\vec{u}_{,y}+w\vec{u}_{,z}
      if (_input->_problem_type == "Floquet") {
        // Linearised equations
        for (PetscInt var = 1; var < NVAR; var++)
          _mW[idx1 + D1 * var] +=
              user->lX[user->lN * 0 + idx0 + 1] * _uY_x[0][var] +
              user->lX[user->lN * 0 + idx0 + 2] * _uY_x[1][var] +
              user->lX[offset + idx0 + 1] * _uY_x_base[0][var] +
              user->lX[offset + idx0 + 2] * _uY_x_base[1][var];
      } else if (_input->_time_scheme_nonlinear_term[user->CurFM] ==
                 FunctionExtrapolation)  // Default for 3D simulations
      {
        for (PetscInt var = 1; var < NVAR; var++)
          _mW[idx1 + D1 * var] +=
              2.0 * (_fs_N[user->FM * (D1_D1 * ElIdx + D1 * (var - 1) + k) +
                           user->CurFM][0] +
                     PETSC_i *
                         _fs_N[user->FM * (D1_D1 * ElIdx + D1 * (var - 1) + k) +
                               user->CurFM][1]) -
              (_fs_N0[user->FM * (D1_D1 * ElIdx + D1 * (var - 1) + k) +
                      user->CurFM][0] +
               PETSC_i *
                   _fs_N0[user->FM * (D1_D1 * ElIdx + D1 * (var - 1) + k) +
                          user->CurFM][1]);
      } else  // Explicit or argument extrapolation
      {
        for (PetscInt var = 1; var < NVAR; var++)
          _mW[idx1 + D1 * var] +=
              _fs_N[user->FM * (D1_D1 * ElIdx + D1 * (var - 1) + k) +
                    user->CurFM][0] +
              PETSC_i * _fs_N[user->FM * (D1_D1 * ElIdx + D1 * (var - 1) + k) +
                              user->CurFM][1];
      }
      // part of _mWGLS coincides with _mW =
      // \vec{Y}_{,t}+A_i\vec{Y}_{,i}-\vec{R}
      for (PetscInt i = 0; i < NVAR; i++)
        _mWGLS[D1 * i + k] = _mW[idx1 + D1 * i];

      // ---------------------------------------------------------
      // _mWi: diffusion and pressure
      for (PetscInt d = 0; d < DIM; d++)  // multiplier: \psi_{,d}
      {
        _mWi[D1_D * 0 + D1 * d + k] = 0.0;  // eq: 0
        if (_input->weak_form_type == 0) {
          for (PetscInt eq = 1; eq < NVAR; eq++)  // equation: 1-3
            _mWi[D1_D * eq + D1 * d + k] =
                (_uY_x[d][eq] + _uY_x[eq - 1][d + 1]) / _input->_Re;
        } else {
          for (PetscInt eq = 1; eq < NVAR; eq++)  // equation: 1-3
            _mWi[D1_D * eq + D1 * d + k] = _uY_x[d][eq] / _input->_Re;
        }
      }

      for (PetscInt d = 0; d < DIM; d++)  // multiplier: \psi{,d}; equation: d+1
        _mWi[D1_D * (d + 1) + D1 * d + k] +=
            -_pScale * user->lX[offset + idx0];  // user->lX[idx0];//

      // _mW:  diffusion and pressure
      // multiplier: [-i\beta]\psi
      if (_input->weak_form_type == 0) {
        for (PetscInt eq = 1; eq < NVAR; eq++)  // equation: 1-3
          _mW[idx1 + D1 * eq] +=
              -PETSC_i * beta * (_uY_x[2][eq] + _uY_x[eq - 1][3]) / _input->_Re;
      } else {
        for (PetscInt eq = 1; eq < NVAR; eq++)  // equation: 1-3
          _mW[idx1 + D1 * eq] += -PETSC_i * beta * _uY_x[2][eq] / _input->_Re;
      }

      // multiplier: [-i\beta]\psi; equation: 3
      _mW[idx1 + D1 * 3] += PETSC_i * beta * _pScale *
                            user->lX[offset + idx0];  //*user->lX[idx0];//

      // =========================================================
      // Stabilisation part

      // ---------------------------------------------------------
      // _mWGLS=L(Y)=\vec{Y}_{,t}+A_i\vec{Y}_{,i}-\vec{R}+\nabla P

      // Conservation of momentum. Pressure gradient: p_{,i}
      for (PetscInt eq = 1; eq < NVAR; eq++) {
        _mWGLS[D1 * eq + k] += _pScale * _uY_x[eq - 1][0];
      }

      // Conservation of momentum zz-diffusion: \beta^2\vec{u} (xx- and yy-part
      // of diffusion is zero - second order derrivatives of linear functions)
      for (PetscInt eq = 1; eq < NVAR - 1; eq++) {
        _mWGLS[D1 * eq + k] +=
            beta * beta * user->lX[offset + idx0 + eq] / _input->_Re;
      }

      _mWGLS[D1 * 3 + k] +=
          2.0 * beta * beta * user->lX[offset + idx0 + 3] / _input->_Re;

      // ---------------------------------------------------------
      // multiplier: [-i\beta]\psi
      // tauPSPG. Equation: 0
      if (test == false) {
        _mW[idx1 + D1 * 0] += (-PETSC_i * beta) * tauPSPG * _mWGLS[D1 * 3 + k];

        // tauSUPG. Equation: 1-3
        // for(PetscInt eq=1; eq<NVAR; eq++)
        //    _mW[idx1+D1*eq] += (-PETSC_i*beta) * tauSUPG * _mWGLS[D1*eq+k] *
        //    _mY_baseflow[D1*3+k];

        // tauLSIC. Equation: 1-3
        _mW[idx1 + D1 * 3] +=
            (-PETSC_i * beta) *
            (tauLSIC)*_mWGLS[k];  // '-' before tauLSIC due to negative term
                                  // \nabla\cdot\vec{u} (see above)
      }
    }

    // =================================================================
    // INTEGRALS: Wi_{,d}*\int(...)d\Omega

    // ---------------------------------------------------------
    // Classic part (diffusion and pressure)
    for (PetscInt d = 0; d < DIM; d++)  // multiplier: \psi{,d}
    {
      for (PetscInt eq = 1; eq < NVAR; eq++)
        _quadWi[DIM * NVAR * ElIdx + DIM * eq + d] = Element::Quad(
            &(_mWi[D1_D * eq + D1 * d]), user->cell_volume[ElIdx]);
    }

    // ---------------------------------------------------------
    // Stabilisation part

    // tauPSPG. Equation: 0
    for (PetscInt d = 0; d < DIM; d++)  // multiplier: \psi{,d}
      _quadWi[DIM * NVAR * ElIdx + DIM * 0 + d] =
          tauPSPG *
          Element::Quad(&(_mWGLS[D1 * (d + 1)]), user->cell_volume[ElIdx]);

    // tauSUPG. Equation: 1-3
    for (PetscInt eq = 1; eq < NVAR; eq++)
      for (PetscInt d = 0; d < DIM; d++)  // multiplier: \psi{,d}
        _quadWi[DIM * NVAR * ElIdx + DIM * eq + d] +=
            tauSUPG * Element::Quad(&(_mWGLS[D1 * eq]),
                                    &(_mY_baseflow[D1 * (d + 1)]),
                                    user->cell_volume[ElIdx]);

    // tauLSIC. Equation: 1-3
    auxQuad = tauLSIC * Element::Quad(&(_mWGLS[0]), user->cell_volume[ElIdx]);
    for (PetscInt d = 0; d < DIM; d++)  // multiplier: \psi{,d}
      _quadWi[DIM * NVAR * ElIdx + DIM * (d + 1) + d] += auxQuad; /**/
  }

  _monitor->EndTimeMeasurement(EqUpdateAuxVars);

  if (_input->_debug_enabled == PETSC_TRUE && pVrtx == -1 &&
      (_input->_debug_thread == _Rank || _input->_debug_thread == -1))
    printf("[%d]NavierStokes::UpdateAuxVars(%ld)-End\n", _Rank, pVrtx);
  return (0);
}

PetscErrorCode NavierStokes::UpdateAuxVars_Real(AppCtx *user, PetscInt pVrtx) {
  if (_input->_debug_enabled == PETSC_TRUE && pVrtx == -1 &&
      (_input->_debug_thread == _Rank || _input->_debug_thread == -1))
    printf("[%d]NavierStokes::UpdateAuxVars(%ld)-Start\n", _Rank, pVrtx);

  _monitor->StartTimeMeasurement(EqUpdateAuxVars);

  if (user->CurFM == 0)
    printf(
        "Error (UpdateAuxVars): base flow mode is not evaluated in TSA/LSA\n");

  bool test = false;

  PetscInt idx0, idx1;
  PetscInt ElIdx;

  PetscInt N;
  PetscInt vtx1, vtx2;
  PetscInt offset = user->lN * user->CurFM;

  PetscScalar auxQuad;
  PetscReal tauPSPG, tauSUPG, tauLSIC;

  PetscScalar beta = 2.0 * PETSC_PI * user->CurFM / _input->_L;

  PetscReal sign = 1.0;
  // if(user->BackwardsInTime == PETSC_FALSE) { sign = 1.0; }
  // else { sign = -1.0; }

  // Update all data (-1) or neighbourhood of one vertex (with number pVrtx)
  if (pVrtx == -1) {
    N = user->Nelocal;
    vtx1 = 0;
    vtx2 = user->Nvlocal + user->NvGhost;
  } else {
    N = user->NEByV[pVrtx];
    vtx1 = pVrtx;
    vtx2 = pVrtx + 1;
  }

  //==============================================================
  // Every node
  for (PetscInt vtx = vtx1; vtx < vtx2; vtx++) {
    idx0 = NVAR * vtx;
    // ---------------------------------------------------------
    // Time derrivative + Forces
    // \vec{C1}=\vec{Y}_{,t}-\vec{R}
    _C1[idx0] = 0.0;

    for (PetscInt eq = 1; eq < NVAR; eq++)
      _C1[idx0 + eq] =
          user->lX_t[offset + idx0 + eq] -
          _input->GetForceValue(eq - 1, user->T + sign * _input->_dT,
                                user->X1[vtx], user->X2[vtx], user->CurFM) /
              _input->_Fr;
  }

  //==============================================================
  // Every element
  for (PetscInt e = 0; e < N; e++) {
    if (pVrtx != -1)
      ElIdx = user->EByV[pVrtx][e];
    else
      ElIdx = e;

    // --------------------------------------------------------------------
    // _uY_x
    // x-, y- derivatives \vec{Y}_{,x} and \vec{Y}_{,y}
    for (PetscInt dir = 0; dir < DIM; dir++)
      Element::Y_x(dir, ElIdx, user->Hx, &(user->lX[offset]), user->VByE[ElIdx],
                   _uY_x[dir]);

    for (PetscInt dir = 0; dir < DIM; dir++)
      Element::Y_x(dir, ElIdx, user->Hx, &(user->lX[0]), user->VByE[ElIdx],
                   _uY_x_base[dir]);

    // ---------------------------------------------------------
    // Stabilisation parameters (using just base flow)
    if (_input->_time_scheme_stabilisation[user->CurFM] ==
        ArgumentExtrapolation)
      GetTau(&(user->lX1_e[0]), ElIdx, user, &tauPSPG, &tauSUPG, &tauLSIC);
    else if (_input->_time_scheme_stabilisation[user->CurFM] == Implicit)
      GetTau(&(user->lX[0]), ElIdx, user, &tauPSPG, &tauSUPG, &tauLSIC);
    else  // Explicit
      GetTau(&(user->lX0[0]), ElIdx, user, &tauPSPG, &tauSUPG, &tauLSIC);

    tauLSIC = -tauLSIC;  // '-' before tauLSIC due to negative term
                         // \nabla\cdot\vec{u} (see below)
    tauPSPG = -tauPSPG;  // '-' before tauLSIC due to negative term
                         // \nabla\cdot\vec{u} (see below)

    // ---------------------------------------------------------
    for (PetscInt k = 0; k < D1; k++) {
      idx0 = NVAR * user->VByE[ElIdx][k];
      idx1 = D1_N * ElIdx + k;

      // z-derivative \vec{Y}_{,z} - depends on k
      for (PetscInt var = 0; var < NVAR; var++)
        _uY_x[2][var] = beta * user->lX[offset + idx0 + var];

      _uY_x[2][3] = -_uY_x[2][3];

      for (PetscInt i = 0; i < NVAR; i++) {
        if (_input->_time_scheme_stabilisation[user->CurFM] ==
            ArgumentExtrapolation)
          _mY_baseflow[D1 * i + k] = user->lX1_e[0 + idx0 + i];
        else if (_input->_time_scheme_stabilisation[user->CurFM] == Implicit)
          _mY_baseflow[D1 * i + k] = user->lX[0 + idx0 + i];
        else  // Explicit
          _mY_baseflow[D1 * i + k] = user->lX0[0 + idx0 + i];
      }

      // ---------------------------------------------------------
      // _mW=\vec{Y}_{,t}+A_i\vec{Y}_{,i}-\vec{R}

      // _mW = \vec{Y}_{,t}-\vec{R}
      for (PetscInt i = 0; i < NVAR; i++) _mW[idx1 + D1 * i] = _C1[idx0 + i];

      // Conservation of mass: -(u_{,x}+v_{,y}+w_{,z}) '-' - for symmetry
      _mW[idx1 + D1 * 0] += -(_uY_x[0][1] + _uY_x[1][2] + _uY_x[2][3]);

      // Conservation of momentum. Nonlinear term:
      // u\vec{u}_{,x}+v\vec{u}_{,y}+w\vec{u}_{,z}
      for (PetscInt var = 1; var < NVAR; var++)
        _mW[idx1 + D1 * var] +=
            user->lX[user->lN * 0 + idx0 + 1] * _uY_x[0][var] +
            user->lX[user->lN * 0 + idx0 + 2] * _uY_x[1][var] +
            user->lX[offset + idx0 + 1] * _uY_x_base[0][var] +
            user->lX[offset + idx0 + 2] * _uY_x_base[1][var];

      // part of _mWGLS coincides with _mW =
      // \vec{Y}_{,t}+A_i\vec{Y}_{,i}-\vec{R}
      for (PetscInt i = 0; i < NVAR; i++)
        _mWGLS[D1 * i + k] = _mW[idx1 + D1 * i];

      // ---------------------------------------------------------
      // _mWi: diffusion and pressure

      for (PetscInt d = 0; d < DIM; d++)  // multiplier: \psi_{,d}
      {
        _mWi[D1_D * 0 + D1 * d + k] = 0.0;  // eq: 0
        if (_input->weak_form_type == 0) {
          for (PetscInt eq = 1; eq < NVAR; eq++)  // equation: 1-3
            _mWi[D1_D * eq + D1 * d + k] =
                (_uY_x[d][eq] + _uY_x[eq - 1][d + 1]) / _input->_Re;
        } else {
          for (PetscInt eq = 1; eq < NVAR; eq++)  // equation: 1-3
            _mWi[D1_D * eq + D1 * d + k] = _uY_x[d][eq] / _input->_Re;
        }
      }

      for (PetscInt d = 0; d < DIM; d++)  // multiplier: \psi{,d}; equation: d+1
        _mWi[D1_D * (d + 1) + D1 * d + k] +=
            -_pScale * user->lX[offset + idx0];  // user->lX[idx0];//

      // _mW:  diffusion and pressure
      // multiplier: [-i\beta]\psi
      if (_input->weak_form_type == 0) {
        for (PetscInt eq = 1; eq < NVAR; eq++)  // equation: 1-3
        {
          if (eq == 3)
            _mW[idx1 + D1 * eq] += -2.0 * beta * _uY_x[2][3] / _input->_Re;
          else
            _mW[idx1 + D1 * eq] +=
                beta * (_uY_x[2][eq] + _uY_x[eq - 1][3]) / _input->_Re;
        }
      } else {
        for (PetscInt eq = 1; eq < NVAR; eq++)  // equation: 1-3
        {
          if (eq == 3)
            _mW[idx1 + D1 * eq] += -beta * _uY_x[2][3] / _input->_Re;
          else
            _mW[idx1 + D1 * eq] += beta * _uY_x[2][eq] / _input->_Re;
        }
      }

      // multiplier: [-i\beta]\psi; equation: 3
      _mW[idx1 + D1 * 3] += beta * _pScale * user->lX[offset + idx0];

      // =========================================================
      // Stabilisation part

      // ---------------------------------------------------------
      // _mWGLS=L(Y)=\vec{Y}_{,t}+A_i\vec{Y}_{,i}-\vec{R}+\nabla P

      // Conservation of momentum. Pressure gradient: p_{,i}
      for (PetscInt eq = 1; eq < NVAR; eq++) {
        _mWGLS[D1 * eq + k] += _pScale * _uY_x[eq - 1][0];
      }

      // Conservation of momentum zz-diffusion: \beta^2\vec{u} (xx- and yy-part
      // of diffusion is zero - second order derrivatives of linear functions)

      for (PetscInt eq = 1; eq < NVAR - 1; eq++) {
        _mWGLS[D1 * eq + k] +=
            beta * beta * user->lX[offset + idx0 + eq] / _input->_Re;
      }

      _mWGLS[D1 * 3 + k] +=
          2.0 * beta * beta * user->lX[offset + idx0 + 3] / _input->_Re;

      // ---------------------------------------------------------
      // multiplier: [-i\beta]\psi
      if (test == false) {
        // tauPSPG. Equation: 0
        _mW[idx1 + D1 * 0] += beta * tauPSPG * _mWGLS[D1 * 3 + k];

        // tauSUPG. Equation: 1-3
        // for(PetscInt eq=1; eq<NVAR; eq++)
        //    _mW[idx1+D1*eq] += (beta) * tauSUPG * _mWGLS[D1*eq+k] *
        //    _mY_baseflow[D1*3+k];

        // tauLSIC. Equation: 1-3
        _mW[idx1 + D1 * 3] +=
            (-beta) *
            (tauLSIC)*_mWGLS[k];  // '-' before tauLSIC due to negative term
                                  // \nabla\cdot\vec{u} (see above)
      }
    }

    // =================================================================
    // INTEGRALS: Wi_{,d}*\int(...)d\Omega

    // ---------------------------------------------------------
    // Classic part (diffusion and pressure)
    for (PetscInt d = 0; d < DIM; d++)  // multiplier: \psi{,d}
    {
      for (PetscInt eq = 1; eq < NVAR; eq++)
        _quadWi[DIM * NVAR * ElIdx + DIM * eq + d] = Element::Quad(
            &(_mWi[D1_D * eq + D1 * d]), user->cell_volume[ElIdx]);
    }

    // ---------------------------------------------------------
    // Stabilisation part

    // tauPSPG. Equation: 0
    for (PetscInt d = 0; d < DIM; d++)  // multiplier: \psi{,d}
      _quadWi[DIM * NVAR * ElIdx + DIM * 0 + d] =
          tauPSPG *
          Element::Quad(&(_mWGLS[D1 * (d + 1)]), user->cell_volume[ElIdx]);

    // tauSUPG. Equation: 1-3
    for (PetscInt eq = 1; eq < NVAR; eq++)
      for (PetscInt d = 0; d < DIM; d++)  // multiplier: \psi{,d}
        _quadWi[DIM * NVAR * ElIdx + DIM * eq + d] +=
            tauSUPG * Element::Quad(&(_mWGLS[D1 * eq]),
                                    &(_mY_baseflow[D1 * (d + 1)]),
                                    user->cell_volume[ElIdx]);

    // tauLSIC. Equation: 1-3
    auxQuad = tauLSIC * Element::Quad(&(_mWGLS[0]), user->cell_volume[ElIdx]);
    for (PetscInt d = 0; d < DIM; d++)  // multiplier: \psi{,d}
      _quadWi[DIM * NVAR * ElIdx + DIM * (d + 1) + d] += auxQuad; /**/
  }

  _monitor->EndTimeMeasurement(EqUpdateAuxVars);

  if (_input->_debug_enabled == PETSC_TRUE && pVrtx == -1 &&
      (_input->_debug_thread == _Rank || _input->_debug_thread == -1))
    printf("[%d]NavierStokes::UpdateAuxVars(%ld)-End\n", _Rank, pVrtx);
  return (0);
}

PetscErrorCode NavierStokes::NonlinearTermFFT(void *ptr) {
  _monitor->StartTimeMeasurement(EqNonlinearTerm);
  if (_input->_debug_enabled == PETSC_TRUE &&
      (_input->_debug_thread == _Rank || _input->_debug_thread == -1))
    printf("[%d]NavierStokes::NonlinearTermFFT-Start\n", _Rank);
  AppCtx *user = (AppCtx *)ptr;

  PetscInt NZ = user->NZ;
  PetscInt FM = user->FM;

  bool debug = false;

  if (_input->_time_scheme_nonlinear_term_extrapolation_f == PETSC_TRUE &&
      _fs_N0_initialized == PETSC_TRUE) {
    for (PetscInt i = 0; i < user->Nelocal * D1_D1 * FM; i++) {
      _fs_N0[i][0] = _fs_N[i][0];
      _fs_N0[i][1] = _fs_N[i][1];
    }
  }

  // ===================================================================
  // Reading Fourier modes
  PetscScalar *x_pt;
  if (_input->_time_scheme_nonlinear_term_extrapolation_a == PETSC_TRUE)
    x_pt = user->lX1_e;
  else
    x_pt = user->lX0;

  PetscReal beta;
  for (PetscInt modeIdx = 0; modeIdx < user->FM; modeIdx++) {
    beta = 2.0 * PETSC_PI * modeIdx / _input->_L;
    for (PetscInt v = 0; v < (user->Nvlocal + user->NvGhost); v++) {
      for (PetscInt var = 0; var < D1; var++) {
        _fs_U[D1 * FM * v + FM * var + modeIdx][0] =
            PetscRealPart(x_pt[user->lN * modeIdx + NVAR * v + 1 + var]);
        _fs_U[D1 * FM * v + FM * var + modeIdx][1] =
            PetscImaginaryPart(x_pt[user->lN * modeIdx + NVAR * v + 1 + var]);

        _fs_Uz[D1 * FM * v + FM * var + modeIdx][0] =
            beta * PetscRealPart(PETSC_i *
                                 x_pt[user->lN * modeIdx + NVAR * v + 1 + var]);
        _fs_Uz[D1 * FM * v + FM * var + modeIdx][1] =
            beta * PetscImaginaryPart(
                       PETSC_i * x_pt[user->lN * modeIdx + NVAR * v + 1 + var]);

        if (!isfinite(_fs_U[D1 * FM * v + FM * var + modeIdx][0]) ||
            !isfinite(_fs_U[D1 * FM * v + FM * var + modeIdx][1])) {
          printf("NAN: _fs_U[%ld+%ld+%ld]\n", 1 * FM * v, FM * var, modeIdx);
          PressEnterToContinue();
        }
      }
    }
  }

  // ===================================================================
  // velocity FFT: Fourier space -> physical space
  // printf("velocity FFT: Fourier space -> physical space\n");
  // PressEnterToContinue();

  if (debug) {
    printf("NZ=%ld\n", NZ);
    printf("--------------------------------------------\n_fs_U\n");
    for (PetscInt v = 0; v < (user->Nvlocal + user->NvGhost); v++) {
      if (v == user->VByE[0][0]) {
        printf("# node: %ld\n", v);
        for (PetscInt modeIdx = 0; modeIdx < user->FM; modeIdx++) {
          printf("  %ld\t", modeIdx);
          for (PetscInt var = 0; var < D1; var++) {
            if (var == 2)
              printf("(%.26lf, %.26lf)\t",
                     _fs_U[D1 * FM * v + FM * var + modeIdx][0],
                     _fs_U[D1 * FM * v + FM * var + modeIdx][1]);
          }
          printf("\n");
        }
      }
    }
    printf("--------------------------------------------\n_fs_Uz\n");
    for (PetscInt v = 0; v < (user->Nvlocal + user->NvGhost); v++) {
      if (v == user->VByE[0][0]) {
        printf("# node: %ld\n", v);
        for (PetscInt modeIdx = 0; modeIdx < user->FM; modeIdx++) {
          printf("  %ld\t", modeIdx);
          for (PetscInt var = 0; var < D1; var++) {
            if (var == 2)
              printf("(%.26lf, %.26lf)\t",
                     _fs_Uz[D1 * FM * v + FM * var + modeIdx][0],
                     _fs_Uz[D1 * FM * v + FM * var + modeIdx][1]);
          }
          printf("\n");
        }
      }
    }
  }

  fftw_execute(_FFTW_PlanC2R_U);
  fftw_execute(_FFTW_PlanC2R_Uz);

  if (debug) {
    printf("--------------------------------------------\n_ps_U\n");
    for (PetscInt v = 0; v < (user->Nvlocal + user->NvGhost); v++) {
      if (v == user->VByE[0][0]) {
        printf("# node: %ld\n", v);
        for (PetscInt zIdx = 0; zIdx < NZ; zIdx++) {
          printf("  %ld\t", zIdx);
          for (PetscInt var = 0; var < D1; var++) {
            if (var == 2) printf("%.26lf\t", _ps_U[NZ * (D1 * v + var) + zIdx]);
          }
          printf("\n");
        }
      }
    }
    printf("--------------------------------------------\n_ps_Uz\n");
    for (PetscInt v = 0; v < (user->Nvlocal + user->NvGhost); v++) {
      if (v == user->VByE[0][0]) {
        printf("# node: %ld\n", v);
        for (PetscInt zIdx = 0; zIdx < NZ; zIdx++) {
          printf("  %ld\t", zIdx);
          for (PetscInt var = 0; var < D1; var++) {
            if (var == 2)
              printf("%.26lf\t", _ps_Uz[NZ * (D1 * v + var) + zIdx]);
          }
          printf("\n");
        }
      }
    }
    PressEnterToContinue();
  }
  // ===================================================================
  // nonlinear term calculations
  // printf("nonlinear term calculations\n"); PressEnterToContinue();
  PetscInt vtx, idx0, idx1, idx2;

  for (PetscInt ElIdx = 0; ElIdx < user->Nelocal; ElIdx++)  // Every element
  {
    // ---------------------------------------------------------
    // Derivatives (x,y)
    for (PetscInt zIdx = 0; zIdx < NZ; zIdx++)  // z-layer
    {
      for (PetscInt var = 0; var < D1; var++)  // velocity component
      {
        idx0 = NZ * (D1 * ElIdx + var) + zIdx;
        // x- and y-derivative calculation
        for (PetscInt dir = 0; dir < DIM; dir++)  // direction of derivative
        {
          _ps_U_x[dir][idx0] = 0.0;
          for (PetscInt v = 0; v < D1; v++)
            _ps_U_x[dir][idx0] +=
                _ps_U[NZ * (D1 * user->VByE[ElIdx][v] + var) + zIdx] *
                user->Hx[D1_D * ElIdx + DIM * v + dir];

          if (!isfinite(_ps_U_x[dir][idx0])) {
            printf("NAN: _ps_U_x[%ld][%ld]\n", dir, idx0);
            PressEnterToContinue();
          }
        }
      }
    }

    // ---------------------------------------------------------
    for (PetscInt k = 0; k < D1; k++) {
      vtx = user->VByE[ElIdx][k];

      // Conservation of momentum. Nonlinear term:
      // u\vec{u}_{,x}+v\vec{u}_{,y}+w\vec{u}_{,z}
      for (PetscInt var = 0; var < D1; var++) {
        idx0 = NZ * (D1_D1 * ElIdx + D1 * var + k);
        idx1 = D1 * NZ * vtx;
        for (PetscInt zIdx = 0; zIdx < NZ; zIdx++) {
          idx2 = NZ * (D1 * ElIdx + var) + zIdx;
          _ps_N[idx0 + zIdx] =
              _ps_U[idx1 + NZ * 0 + zIdx] *
                  _ps_U_x[0][idx2] +  // u\vec{u}_{,x}+
              _ps_U[idx1 + NZ * 1 + zIdx] *
                  _ps_U_x[1][idx2] +  // v\vec{u}_{,y}+
              _ps_U[idx1 + NZ * 2 + zIdx] *
                  _ps_Uz[idx1 + NZ * var + zIdx];  // w\vec{u}_{,z}

          if (!isfinite(_ps_N[idx0 + zIdx])) {
            printf("NAN: _ps_N[%ld+%ld]\n", idx0, zIdx);
            PressEnterToContinue();
          }
        }
      }
    }
  }

  // ===================================================================
  // nonlinear term FFT: physical space -> Fourier space
  // printf("nonlinear term FFT: physical space -> Fourier space\n");
  // PressEnterToContinue();

  if (debug) {
    printf("--------------------------------------------\n_ps_N\n");
    for (PetscInt ElIdx = 0; ElIdx < user->Nelocal; ElIdx++) {
      for (PetscInt k = 0; k < D1; k++) {
        if (ElIdx == 0 && k == 0) {
          printf("# element: %ld (v=%ld: x=%lf y=%lf)\n", ElIdx, k,
                 user->X1[user->VByE[ElIdx][k]],
                 user->X2[user->VByE[ElIdx][k]]);
          for (PetscInt zIdx = 0; zIdx < NZ; zIdx++) {
            if (true)  // zIdx == 0)
            {
              printf("  %ld\t", zIdx);
              for (PetscInt var = 0; var < D1; var++) {
                if (var == 2)
                  printf("%.26lf\t",
                         _ps_N[NZ * (D1_D1 * ElIdx + D1 * var + k) + zIdx]);
              }
              printf("\n");
            }
          }
        }
      }
    }
  }

  fftw_execute(_FFTW_PlanR2C_N);

  // normalisation
  for (PetscInt i = 0; i < user->Nelocal * D1_D1 * FM; i++) {
    _fs_N[i][0] = _fs_N[i][0] / NZ;
    _fs_N[i][1] = _fs_N[i][1] / NZ;

    if (!isfinite(_fs_N[i][0]) || !isfinite(_fs_N[i][1])) {
      printf("NAN: _fs_N[%ld]\n", i);
      PressEnterToContinue();
    }
  }

  if (debug) {
    printf("--------------------------------------------\n_fs_N\n");
    for (PetscInt ElIdx = 0; ElIdx < user->Nelocal; ElIdx++) {
      for (PetscInt k = 0; k < D1; k++) {
        for (PetscInt modeIdx = 0; modeIdx < user->FM; modeIdx++) {
          for (PetscInt var = 0; var < D1; var++) {
            PetscInt idxN = FM * (D1_D1 * ElIdx + D1 * var + k) + modeIdx;
            PetscInt idxU = D1 * FM * user->VByE[ElIdx][k] + FM * var + modeIdx;
            if (fabs(_fs_N[idxN][0] - _fs_U[idxU][0]) > 0.000000000001 ||
                fabs(_fs_N[idxN][1] - _fs_U[idxU][1]) > 0.000000000001) {
              printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
              printf("Fourier\n");
              printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
              PressEnterToContinue();
            }
          }
        }
      }
    }

    for (PetscInt ElIdx = 0; ElIdx < user->Nelocal; ElIdx++) {
      for (PetscInt k = 0; k < D1; k++) {
        if (ElIdx == 0 && k == 0) {
          printf("# element: %ld (v=%ld: x=%lf y=%lf)\n", ElIdx, k,
                 user->X1[user->VByE[ElIdx][k]],
                 user->X2[user->VByE[ElIdx][k]]);
          for (PetscInt modeIdx = 0; modeIdx < user->FM; modeIdx++) {
            if (true)  // modeIdx == 0)
            {
              printf("  %ld\t", modeIdx);
              for (PetscInt var = 0; var < D1; var++) {
                if (var == 2)
                  printf(
                      "%.26lf+i%.26lf\t",
                      _fs_N[user->FM * (D1_D1 * ElIdx + D1 * var + k) + modeIdx]
                           [0],
                      _fs_N[user->FM * (D1_D1 * ElIdx + D1 * var + k) + modeIdx]
                           [1]);
              }
              printf("\n");
            }
          }
        }
      }
    }
    PressEnterToContinue();
  }

  if (_input->_time_scheme_nonlinear_term_extrapolation_f == PETSC_TRUE &&
      _fs_N0_initialized == PETSC_FALSE) {
    for (PetscInt i = 0; i < user->Nelocal * D1_D1 * FM; i++) {
      _fs_N0[i][0] = _fs_N[i][0];
      _fs_N0[i][1] = _fs_N[i][1];
    }
    _fs_N0_initialized = PETSC_TRUE;
  }

  if (_input->_debug_enabled == PETSC_TRUE &&
      (_input->_debug_thread == _Rank || _input->_debug_thread == -1))
    printf("[%d]NavierStokes::NonlinearTermFFT-End\n", _Rank);

  _monitor->EndTimeMeasurement(EqNonlinearTerm);
  return 0;
}

PetscErrorCode NavierStokes::DestroyAuxVars() {
  if (_input->_debug_enabled == PETSC_TRUE &&
      (_input->_debug_thread == _Rank || _input->_debug_thread == -1))
    printf("[%d]NavierStokes::DestroyAuxVars-Start\n", _Rank);

  PetscErrorCode ierr;

  ierr = PetscFree(_aF);
  CHKERRQ(ierr);
  ierr = PetscFree(_aDF);
  CHKERRQ(ierr);
  // VecDestroy(&_LocalX_t);
  // VecDestroy(&_LocalY);
  ierr = PetscFree(_jRows);
  CHKERRQ(ierr);
  ierr = PetscFree(_jValues);
  CHKERRQ(ierr);

  for (int i = 0; i < D1; i++) {
    ierr = PetscFree(_uY_x[i]);
    CHKERRQ(ierr);
    ierr = PetscFree(_uY_x_base[i]);
    CHKERRQ(ierr);
  }

  ierr = PetscFree(_uY_x);
  CHKERRQ(ierr);
  ierr = PetscFree(_uY_x_base);
  CHKERRQ(ierr);

  ierr = PetscFree(_eH);
  CHKERRQ(ierr);

  ierr = PetscFree(_mW);
  CHKERRQ(ierr);
  ierr = PetscFree(_mWi);
  CHKERRQ(ierr);
  ierr = PetscFree(_mWGLS);
  CHKERRQ(ierr);
  ierr = PetscFree(_mY_baseflow);
  CHKERRQ(ierr);

  ierr = PetscFree(_C1);
  CHKERRQ(ierr);
  ierr = PetscFree(aC1);
  CHKERRQ(ierr);
  ierr = PetscFree(_quadWi);
  CHKERRQ(ierr);
  ierr = PetscFree(quadWi);
  CHKERRQ(ierr);

  ierr = PetscFree(mW);
  CHKERRQ(ierr);
  ierr = PetscFree(mWi);
  CHKERRQ(ierr);
  ierr = PetscFree(mWGLS);
  CHKERRQ(ierr);

  // ===================================================================
  // Finalize aux arrays for nonlinear term calculations

  fftw_destroy_plan(_FFTW_PlanC2R_U);
  fftw_destroy_plan(_FFTW_PlanC2R_Uz);
  fftw_destroy_plan(_FFTW_PlanR2C_N);

  // Fourier space
  fftw_free(_fs_U);
  fftw_free(_fs_Uz);

  ierr = PetscFree(_ps_U);
  CHKERRQ(ierr);
  ierr = PetscFree(_ps_Uz);
  CHKERRQ(ierr);
  for (int i = 0; i < DIM; i++) {
    ierr = PetscFree(_ps_U_x[i]);
    CHKERRQ(ierr);
  }

  if (_input->_time_scheme_nonlinear_term_extrapolation_f == PETSC_TRUE)
    fftw_free(_fs_N0);

  fftw_free(_fs_N);
  // ierr = PetscFree(_fs_N);CHKERRQ(ierr);
  ierr = PetscFree(_ps_N);
  CHKERRQ(ierr);

  ierr = PetscFree(_pScaleRows);
  CHKERRQ(ierr);

  fftw_cleanup();

  if (_input->_debug_enabled == PETSC_TRUE &&
      (_input->_debug_thread == _Rank || _input->_debug_thread == -1))
    printf("[%d]NavierStokes::DestroyAuxVars-End\n", _Rank);
  return (0);
}

PetscScalar NavierStokes::WeakFormMain(
    PetscInt pH,
    AppCtx *user)  //, PetscScalar* pY, PetscScalar* pY_t, PetscScalar* pLS
{
  PetscInt i0 = pH % NVAR;
  PetscInt iH = pH / NVAR;

  PetscInt k, e, d;
  PetscScalar result = 0.0;
  PetscInt ElementIdx, HatIdx = -1;

  for (e = 0; e < user->NEByV[iH]; e++) {
    ElementIdx = user->EByV[iH][e];

    for (k = 0; k < D1; k++)
      if (iH == user->VByE[ElementIdx][k]) HatIdx = k;

    result += Element::Quad_FHat(&(_mW[D1_N * ElementIdx + D1 * i0]), HatIdx,
                                 user->cell_volume[ElementIdx], DIM);

    for (d = 0; d < DIM; d++) {
      result += user->Hx[D1_D * ElementIdx + DIM * HatIdx + d] *
                _quadWi[DIM * NVAR * ElementIdx + DIM * i0 + d];
    }
  }

  return result;
}

// pH - номер уравнения
PetscScalar NavierStokes::WeakFormDomainBorder(PetscInt pH, AppCtx *user) {
  return 0.0;
}

void NavierStokes::GetTau(const PetscScalar *pY, PetscInt pE, AppCtx *user,
                          PetscReal *rTauPSPG, PetscReal *rTauSUPG,
                          PetscReal *rTauLSIC) {
  PetscInt i, k, pIndexes[D1];
  PetscScalar U[DIM];
  PetscReal absU, aux;

  // ------------------------------
  // absU
  for (i = 0; i < DIM; i++) U[i] = 0.0;

  for (k = 0; k < D1; k++) {
    pIndexes[k] = user->VByE[pE][k];
    for (i = 0; i < DIM; i++) U[i] += pY[NVAR * pIndexes[k] + 1 + i];
  }
  absU = 0.0;
  for (i = 0; i < DIM; i++) {
    aux = PetscRealPart(U[i]);
    absU += aux * aux;
  }

  absU = PetscSqrtReal(absU) / D1;

  // ------------------------------
  // hu
  /*PetscReal hu = 0;
  for (k = 0; k < D1; k++)
      for(i=0; i<DIM; i++)
          hu =max(PetscAbsReal(user->Hx[D1_D*pE+DIM*k+i]), hu);
  hu = 2/hu;*/

  // ------------------------------
  // Stabilization

  if (_input->_solver_fem_tau_LSIC[user->CurFM] < TAU_PRECISION)
    (*rTauLSIC) = 0.0;
  else {
    (*rTauLSIC) = 0.5 * _eH[pE] * absU;
    PetscReal Reh = _input->_Re * (*rTauLSIC);
    if (Reh <= 3.0) (*rTauLSIC) *= Reh / 3.0;

    (*rTauLSIC) = _input->_solver_fem_tau_LSIC[user->CurFM] * (*rTauLSIC);
  }

  if (_input->_solver_fem_tau_SUPG[user->CurFM] < TAU_PRECISION) {
    (*rTauSUPG) = 0.0;
  } else {
    PetscReal tau0 = 2.0 / _input->_dT;
    tau0 = tau0 * tau0;
    PetscReal tau1 = 2.0 * absU / _eH[pE];
    tau1 = tau1 * tau1;
    PetscReal tau2 = 4.0 / (_input->_Re * _eH[pE] * _eH[pE]);
    tau2 = tau2 * tau2;

    (*rTauSUPG) = _input->_solver_fem_tau_SUPG[user->CurFM] /
                  PetscSqrtReal(tau0 + tau1 + tau2);
  }

  if (_input->_solver_fem_tau_PSPG[user->CurFM] < TAU_PRECISION)
    (*rTauPSPG) = 0.0;
  else if (_input->_solver_fem_tau_SUPG[user->CurFM] < TAU_PRECISION) {
    PetscReal tau0 = 2.0 / _input->_dT;
    tau0 = tau0 * tau0;
    PetscReal tau1 = 2.0 * absU / _eH[pE];
    tau1 = tau1 * tau1;
    PetscReal tau2 = 4.0 / (_input->_Re * _eH[pE] * _eH[pE]);
    tau2 = tau2 * tau2;

    (*rTauPSPG) = _input->_solver_fem_tau_PSPG[user->CurFM] /
                  PetscSqrtReal(tau0 + tau1 + tau2);
  } else
    (*rTauPSPG) = _input->_solver_fem_tau_PSPG[user->CurFM] * (*rTauSUPG) /
                  _input->_solver_fem_tau_SUPG[user->CurFM];
}

PetscErrorCode NavierStokes::TestNullSpace(Mat pM, Vec pX) {
  PetscErrorCode ierr;
  Vec vecP;
  MatNullSpace nullspace;

  ierr = VecDuplicate(pX, &vecP);
  CHKERRQ(ierr);

  ierr = VecSet(vecP, 0.0);
  CHKERRQ(ierr);

  PetscInt istart, iend;
  VecGetOwnershipRange(vecP, &istart, &iend);

  PetscScalar val = 1.0;
  for (PetscInt i = istart; i < iend; i++)
    if (i % NVAR == 0) VecSetValues(vecP, 1, &i, &val, INSERT_VALUES);

  ierr = VecAssemblyBegin(vecP);
  CHKERRQ(ierr);
  ierr = VecAssemblyEnd(vecP);
  CHKERRQ(ierr);

  VecNormalize(vecP, NULL);
  MatNullSpaceCreate(MPI_COMM_WORLD, PETSC_FALSE, 1, &vecP, &nullspace);

  VecDestroy(&vecP);
  PetscBool result;
  MatNullSpaceTest(nullspace, pM, &result);

  if (result == PETSC_TRUE)
    printf("Nullspace is correct!\n");
  else
    printf("ERROR: Nullspace is not correct!\n");

  MatNullSpaceDestroy(&nullspace);

  return 0;
}

PetscScalar NavierStokes::ComplexMult(fftw_complex p_fftw_complex,
                                      PetscScalar p_petsc_complex) {
  return p_fftw_complex[0] * PetscRealPart(p_petsc_complex) -
         p_fftw_complex[1] * PetscImaginaryPart(p_petsc_complex) +
         PETSC_i * (p_fftw_complex[0] * PetscImaginaryPart(p_petsc_complex) +
                    p_fftw_complex[1] * PetscRealPart(p_petsc_complex));
}
