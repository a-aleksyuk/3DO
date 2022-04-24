#include "monitor.h"

using namespace O3D;

Monitor::Monitor() {
  MPI_Comm_rank(MPI_COMM_WORLD, &_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &_size);
}

void Monitor::Initialize(Input *p_input) {
  _input = p_input;

  _snes_iterations = 0;
  _snes_failed_iterations = 0;

  _monitor_efficiency = _input->_output_efficiency_enabled;
  if (_monitor_efficiency) {
    _t_SNESSolver.total = 0.0;
    _t_NonlinearTerm.total = 0.0;
    _t_StepUpdate.total = 0.0;
    _t_SaveResults.total = 0.0;
    _t_TotalSolver.total = 0.0;
    _t_Iteration.total = 0.0;
    _t_create_solver.total = 0.0;

    _t_function.total = 0.0;
    _t_jacobian.total = 0.0;
    _t_jacobian_loop.total = 0.0;
    _t_WeakFormMain.total = 0.0;
    _t_WeakFormDomainBorder.total = 0.0;
    _t_UpdateAuxVars.total = 0.0;
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

void Monitor::Reset() {
  if (_monitor_efficiency) {
    _t_SNESSolver.total = 0.0;
    _t_NonlinearTerm.total = 0.0;
    _t_StepUpdate.total = 0.0;
    _t_SaveResults.total = 0.0;
    _t_TotalSolver.total = 0.0;
    _t_Iteration.total = 0.0;

    _t_function.total = 0.0;
    _t_jacobian.total = 0.0;
    _t_jacobian_loop.total = 0.0;
    _t_WeakFormMain.total = 0.0;
    _t_WeakFormDomainBorder.total = 0.0;
    _t_UpdateAuxVars.total = 0.0;
  }
}

void Monitor::UpdateSNESIterations(PetscInt p_mode, PetscInt p_snes_iterations,
                                   PetscInt p_failed_iterations) {
  _snes_iterations += p_snes_iterations;
  _snes_failed_iterations += p_failed_iterations;

  if (_rank == 0 && p_failed_iterations != 0) {
    printf("\n######################\n");
    printf("MODE %ld\nUNSUCCESSFULL STEP (%ld)\n######################\n\n",
           p_mode, p_failed_iterations);
    printf("\n######################\n");
  }
}

void Monitor::StartTimeMeasurement(CodeType p_code_type) {
  if (_monitor_efficiency && _rank == 0) {
    if (p_code_type == StepUpdate) {
      PetscTime(&_t_SaveResults.start);
    } else if (p_code_type == SaveResults) {
      PetscTime(&_t_StepUpdate.start);
    } else if (p_code_type == EqNonlinearTerm) {
      PetscTime(&_t_NonlinearTerm.start);
    } else if (p_code_type == SNESSolver) {
      PetscTime(&_t_SNESSolver.start);
    } else if (p_code_type == TotalSolver) {
      PetscTime(&_t_TotalSolver.start);
    } else if (p_code_type == Iteration) {
      PetscTime(&_t_Iteration.start);
    } else if (p_code_type == CreateSolver) {
      PetscTime(&_t_create_solver.start);
    } else if (p_code_type == EqFunction) {
      PetscTime(&_t_function.start);
    } else if (p_code_type == EqJacobian) {
      PetscTime(&_t_jacobian.start);
    } else if (p_code_type == EqJacobianLoop) {
      PetscTime(&_t_jacobian_loop.start);
    } else if (p_code_type == EqWeakFormMain) {
      PetscTime(&_t_WeakFormMain.start);
    } else if (p_code_type == EqWeakFormDomainBorder) {
      PetscTime(&_t_WeakFormDomainBorder.start);
    } else if (p_code_type == EqUpdateAuxVars) {
      PetscTime(&_t_UpdateAuxVars.start);
    }
  }
}
void Monitor::EndTimeMeasurement(CodeType p_code_type) {
  if (_monitor_efficiency && _rank == 0) {
    if (p_code_type == StepUpdate) {
      PetscTime(&_t_SaveResults.end);
      _t_SaveResults.total += _t_SaveResults.end - _t_SaveResults.start;
    } else if (p_code_type == SaveResults) {
      PetscTime(&_t_StepUpdate.end);
      _t_StepUpdate.total += _t_StepUpdate.end - _t_StepUpdate.start;
    } else if (p_code_type == EqNonlinearTerm) {
      PetscTime(&_t_NonlinearTerm.end);
      _t_NonlinearTerm.total += _t_NonlinearTerm.end - _t_NonlinearTerm.start;
    } else if (p_code_type == SNESSolver) {
      PetscTime(&_t_SNESSolver.end);
      _t_SNESSolver.total += _t_SNESSolver.end - _t_SNESSolver.start;
    } else if (p_code_type == TotalSolver) {
      PetscTime(&_t_TotalSolver.end);
      _t_TotalSolver.total += _t_TotalSolver.end - _t_TotalSolver.start;
    } else if (p_code_type == Iteration) {
      PetscTime(&_t_Iteration.end);
      _t_Iteration.total += _t_Iteration.end - _t_Iteration.start;
    } else if (p_code_type == CreateSolver) {
      PetscTime(&_t_create_solver.end);
      _t_create_solver.total += _t_create_solver.end - _t_create_solver.start;
    } else if (p_code_type == EqFunction) {
      PetscTime(&_t_function.end);
      _t_function.total += _t_function.end - _t_function.start;
    } else if (p_code_type == EqJacobian) {
      PetscTime(&_t_jacobian.end);
      _t_jacobian.total += _t_jacobian.end - _t_jacobian.start;
    } else if (p_code_type == EqJacobianLoop) {
      PetscTime(&_t_jacobian_loop.end);
      _t_jacobian_loop.total += _t_jacobian_loop.end - _t_jacobian_loop.start;
    } else if (p_code_type == EqWeakFormMain) {
      PetscTime(&_t_WeakFormMain.end);
      _t_WeakFormMain.total += _t_WeakFormMain.end - _t_WeakFormMain.start;
    } else if (p_code_type == EqWeakFormDomainBorder) {
      PetscTime(&_t_WeakFormDomainBorder.end);
      _t_WeakFormDomainBorder.total +=
          _t_WeakFormDomainBorder.end - _t_WeakFormDomainBorder.start;
    } else if (p_code_type == EqUpdateAuxVars) {
      PetscTime(&_t_UpdateAuxVars.end);
      _t_UpdateAuxVars.total += _t_UpdateAuxVars.end - _t_UpdateAuxVars.start;
    }
  }
}

void Monitor::PrintMonitor(PetscInt p_iteration, PetscReal p_time) {
  if (_rank == 0 && (p_iteration % _input->_output_monitor_period == 0 ||
                     p_iteration == _input->_NT)) {
    PetscReal one_iteration_time;
    if (p_iteration != 0) {
      EndTimeMeasurement(Iteration);
      one_iteration_time = (_t_Iteration.end - _t_Iteration.start) /
                           (p_iteration - _last_iteration);
    } else {
      one_iteration_time = 0;
    }

    printf(" -> Iteration %ld (t = %.6lf, snes-its = %ld; %g sec.)\n",
           p_iteration + 1, p_time, _snes_iterations, one_iteration_time);

    StartTimeMeasurement(Iteration);

    _last_iteration = p_iteration;
  }
}

void Monitor::PrintStatistics() {
  if (_rank == 0) {
    printf("\n===============================\n");
    printf("Number of SNES iterations: %ld, ", _snes_iterations);
    printf("number of unsuccessful steps: %ld\n", _snes_failed_iterations);

    printf("Number of processes: %d\n", _size);
    printf("Create time: %g sec.\n", _t_create_solver.total);

    printf("Solver time: %g sec.\n", _t_TotalSolver.total);
    printf("  Nonlinear term time: %g sec.\n", _t_NonlinearTerm.total);
    printf("  SNES time: %g sec.\n", _t_SNESSolver.total);
    printf("\tFunction: %g sec.\n", _t_function.total);
    printf("\tUpdate aux vars: %g sec.\n", _t_UpdateAuxVars.total);
    printf("\tJacobian: %g sec.\n", _t_jacobian.total);

    printf("\t  JacobianLoop: %g sec.\n", _t_jacobian_loop.total);
    printf("\tSave time: %g sec.\n", _t_SaveResults.total);
    printf("\tShift time: %g sec.\n", _t_StepUpdate.total);
  }
}
void Monitor::SaveStatistics() {
  char fTime[512];

  sprintf(fTime, "%s/Performance.txt", _input->_Output);

  // Сохранение времен выполнения
  if (_rank == 0) {
    FILE *fptrTimes;
    fptrTimes = fopen(fTime, "w");

    fprintf(fptrTimes, "Number of SNES iterations: %ld, ", _snes_iterations);
    fprintf(fptrTimes, "number of unsuccessful steps: %ld\n",
            _snes_failed_iterations);

    fprintf(fptrTimes, "Number of processes: %d\n", _size);
    fprintf(fptrTimes, "Create time: %g sec.\n", _t_create_solver.total);

    fprintf(fptrTimes, "Solver time: %g sec.\n", _t_TotalSolver.total);
    fprintf(fptrTimes, "  Nonlinear term time: %g sec.\n",
            _t_NonlinearTerm.total);
    fprintf(fptrTimes, "  SNES time: %g sec.\n", _t_SNESSolver.total);
    fprintf(fptrTimes, "\tFunction: %g sec.\n", _t_function.total);
    fprintf(fptrTimes, "\tUpdate aux vars: %g sec.\n", _t_UpdateAuxVars.total);
    fprintf(fptrTimes, "\tJacobian: %g sec.\n", _t_jacobian.total);

    fprintf(fptrTimes, "\t  JacobianLoop: %g sec.\n", _t_jacobian_loop.total);
    fprintf(fptrTimes, "\tSave time: %g sec.\n", _t_SaveResults.total);
    fprintf(fptrTimes, "\tShift time: %g sec.\n", _t_StepUpdate.total);
    fclose(fptrTimes);
  }
}

PetscErrorCode Monitor::SaveMatrix(char *pMatName, PetscViewerFormat pFormat,
                                   Mat pMat) {
  if (true) return 0;
  // PETSC_VIEWER_ASCII_DENSE
  char file[512];
  sprintf(file, "%s/%s.m", _input->_Output, pMatName);
  PetscViewer viewerA;

  PetscViewerCreate(PETSC_COMM_WORLD, &viewerA);
  PetscViewerSetType(viewerA, PETSCVIEWERASCII);
  PetscViewerBinarySetSkipHeader(viewerA, PETSC_TRUE);
  PetscViewerBinarySetSkipOptions(viewerA, PETSC_TRUE);
  PetscViewerBinarySkipInfo(viewerA);
  PetscViewerFileSetMode(viewerA, FILE_MODE_WRITE);
  PetscViewerFileSetName(viewerA, file);

  PetscViewerPushFormat(viewerA, pFormat);
  PetscObjectSetName((PetscObject)pMat, pMatName);
  MatView(pMat, viewerA);
  PetscViewerPopFormat(viewerA);

  PetscViewerDestroy(&viewerA);

  if (_rank == 0) printf("Matrix '%s' saved\n", pMatName);
  return 0;
}

PetscErrorCode Monitor::SaveMatrix(char *pMatName, PetscViewerFormat pFormat,
                                   PetscInt pNRow, PetscInt pNCol,
                                   PetscReal *pMat) {
  if (true) return 0;
  if (_rank != 0) return 0;

  // PETSC_VIEWER_ASCII_DENSE
  char file[512];
  sprintf(file, "%s/%s.m", _input->_Output, pMatName);
  FILE *fptr;
  fptr = fopen(file, "w");

  if (pFormat == PETSC_VIEWER_ASCII_DENSE) {
    for (int row = 0; row < pNRow; row++) {
      for (int col = 0; col < pNCol; col++)
        fprintf(fptr, "%.10lf\t", pMat[pNCol * row + col]);
      fprintf(fptr, "\n");
    }
  } else if (pFormat == PETSC_VIEWER_ASCII_MATLAB) {
    fprintf(fptr, "%s=...\n[", pMatName);
    for (int row = 0; row < pNRow; row++) {
      for (int col = 0; col < pNCol; col++)
        fprintf(fptr, "%.10lf\t", pMat[pNCol * row + col]);
      fprintf(fptr, ";\n");
    }
    fprintf(fptr, "];\n");
  }

  fclose(fptr);

  if (_rank == 0) printf("Matrix '%s' saved\n", pMatName);

  return 0;
}

PetscErrorCode Monitor::SaveVector(char *pVecName, PetscViewerFormat pFormat,
                                   Vec pVec) {
  if (true) return 0;
  char file[512];
  sprintf(file, "%s/%s.m", _input->_Output, pVecName);
  PetscViewer viewerA;

  PetscViewerCreate(PETSC_COMM_WORLD, &viewerA);
  PetscViewerSetType(viewerA, PETSCVIEWERASCII);
  PetscViewerBinarySetSkipHeader(viewerA, PETSC_TRUE);
  PetscViewerBinarySetSkipOptions(viewerA, PETSC_TRUE);
  PetscViewerBinarySkipInfo(viewerA);
  PetscViewerFileSetMode(viewerA, FILE_MODE_WRITE);
  PetscViewerFileSetName(viewerA, file);

  PetscViewerPushFormat(viewerA, pFormat);
  PetscObjectSetName((PetscObject)pVec, pVecName);
  VecView(pVec, viewerA);
  PetscViewerPopFormat(viewerA);

  PetscViewerDestroy(&viewerA);

  if (_rank == 0) printf("Vector '%s' saved\n", pVecName);

  return 0;
}

PetscErrorCode Monitor::SaveVector(char *pVecName, PetscViewerFormat pFormat,
                                   PetscInt pNRow, PetscReal *pVec) {
  if (true) return 0;
  if (_rank != 0) return 0;

  char file[512];
  sprintf(file, "%s/%s.m", _input->_Output, pVecName);
  FILE *fptr;
  fptr = fopen(file, "w");

  if (pFormat == PETSC_VIEWER_ASCII_DENSE) {
    for (int row = 0; row < pNRow; row++) fprintf(fptr, "%.20lf\n", pVec[row]);
  } else if (pFormat == PETSC_VIEWER_ASCII_MATLAB) {
    fprintf(fptr, "%s=...\n[", pVecName);
    for (int row = 0; row < pNRow; row++) fprintf(fptr, "%.20lf;\n", pVec[row]);
    fprintf(fptr, "];\n");
  }

  fclose(fptr);

  if (_rank == 0) printf("Vector '%s' saved\n", pVecName);

  return 0;
}

PetscErrorCode Monitor::SaveVector(char *pVecName, PetscViewerFormat pFormat,
                                   PetscInt pNRow, PetscScalar *pVec) {
  if (true) return 0;
  if (_rank != 0) return 0;

  char file[512];
  sprintf(file, "%s/%s.m", _input->_Output, pVecName);
  FILE *fptr;
  fptr = fopen(file, "w");

  if (pFormat == PETSC_VIEWER_ASCII_DENSE) {
    for (int row = 0; row < pNRow; row++)
      fprintf(fptr, "%.20lf\n", PetscRealPart(pVec[row]));
  } else if (pFormat == PETSC_VIEWER_ASCII_MATLAB) {
    fprintf(fptr, "%s=...\n[", pVecName);
    for (int row = 0; row < pNRow; row++)
      fprintf(fptr, "%.20lf;\n", PetscRealPart(pVec[row]));
    fprintf(fptr, "];\n");
  }

  fclose(fptr);

  if (_rank == 0) printf("Vector '%s' saved\n", pVecName);

  return 0;
}
