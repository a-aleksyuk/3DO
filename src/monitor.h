#ifndef MONITOR_H
#define MONITOR_H

#include "input.h"
#include "structures.h"

namespace O3D {
enum CodeType {
  TotalSolver,
  StepUpdate,
  SaveResults,
  EqNonlinearTerm,
  SNESSolver,
  Iteration,
  CreateSolver,
  EqFunction,
  EqJacobian,
  EqJacobianLoop,
  EqWeakFormMain,
  EqWeakFormDomainBorder,
  EqUpdateAuxVars
};

struct TimerSet {
  PetscReal start;
  PetscReal end;
  PetscReal total;
};

class Monitor {
 private:
  int _rank, _size;

  Input *_input;

  // SNES iterations
  PetscInt _snes_iterations, _snes_failed_iterations;

  // Monitor efficiency
  bool _monitor_efficiency;

  // Solver
  TimerSet _t_SNESSolver;
  TimerSet _t_NonlinearTerm;
  TimerSet _t_StepUpdate;
  TimerSet _t_SaveResults;
  TimerSet _t_TotalSolver;
  TimerSet _t_Iteration;
  TimerSet _t_create_solver;

  // Equation
  TimerSet _t_function;
  TimerSet _t_jacobian;
  TimerSet _t_jacobian_loop;
  TimerSet _t_WeakFormMain;
  TimerSet _t_WeakFormDomainBorder;
  TimerSet _t_UpdateAuxVars;

  PetscInt _last_iteration;

 public:
  Monitor();

  void Initialize(Input *p_input);
  void Reset();

  void UpdateSNESIterations(PetscInt p_mode, PetscInt p_snes_iterations,
                            PetscInt p_failed_iterations);

  void StartTimeMeasurement(CodeType p_code_type);
  void EndTimeMeasurement(CodeType p_code_type);

  void PrintMonitor(PetscInt p_iteration, PetscReal p_time);

  void PrintStatistics();
  void SaveStatistics();

  PetscErrorCode SaveMatrix(char *pMatName, PetscViewerFormat pFormat,
                            Mat pMat);
  PetscErrorCode SaveMatrix(char *pMatName, PetscViewerFormat pFormat,
                            PetscInt pNRow, PetscInt pNCol, PetscReal *pMat);
  PetscErrorCode SaveVector(char *pVecName, PetscViewerFormat pFormat,
                            Vec pVec);
  PetscErrorCode SaveVector(char *pVecName, PetscViewerFormat pFormat,
                            PetscInt pNRow, PetscReal *pVec);
  PetscErrorCode SaveVector(char *pVecName, PetscViewerFormat pFormat,
                            PetscInt pNRow, PetscScalar *pVec);
};
}  // namespace O3D
#endif  // MONITOR_H
