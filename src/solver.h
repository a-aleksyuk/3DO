#ifndef SOLVER_H_
#define SOLVER_H_

#include <petscsnes.h>
#include <petsctime.h>

#include "eigenproblem.h"
#include "input.h"
#include "meshpartition.h"
#include "monitor.h"
#include "structures.h"

namespace O3D {
enum ProblemType {
  PrimalBaseFlow = 0,
  PrimalPerturbations = 1,
  AdjointBaseFlow = 3,
  AdjointPerturbations = 2
};
enum SolverType { ReadFromFile, PETSc };
enum SavingRegime { Off, Usual, OptimalPertubations, KSIterations };

class Solver {
 private:
  int _Rank, _Size;
  Input *_input;
  Monitor *_monitor;
  Eigenproblem *_eigenproblem;

  PetscInt _eigenvalue_index;  // number of current initial perturbation
                               // (current eigenvalue)

  AppCtx _User;
  SNES _Snes[3];

  Mat _Jac[3];

  ISLocalToGlobalMapping _ISl2g;

  MeshPartition *_MP;

  PetscReal _CreateTime;

  PetscReal *_auxLocalX_real;
  PetscReal *_auxLocalX_img;

  PetscReal *_R;  // 3*DIM*[граница]+3*[составляющая x, y или z] + [полное,
                  // давление, вязкость]
  PetscReal *_RGlobal;

 private:
  // Norm calculation
  PetscReal *_norm_weigths;
  PetscReal *_norm_weigths_sqrt;

  void InitializeNormWeights();
  PetscReal GetNormEnergy(PetscScalar *pX);
  PetscReal GetVNorm2(PetscScalar *pX);
  void GetMaximumReIm(PetscScalar *pX, PetscInt p_var, PetscReal *r_max_re,
                      PetscReal *r_max_im);

 private:
  PetscErrorCode CreateMeshParameters();
  PetscErrorCode DestroyMeshParameters();

  PetscErrorCode CreateDataStructures();
  PetscErrorCode DestroyDataStructures();

  PetscErrorCode CreateDataStructuresAdjoint();
  PetscErrorCode DestroyDataStructuresAdjoint();

  static PetscErrorCode Jacobian(SNES, Vec, Mat, Mat, void *);
  static PetscErrorCode JacobianAdjoint(SNES, Vec, Mat, Mat, void *);
  static PetscErrorCode JacobianNonZeros(SNES, Vec, Mat, Mat, void *);
  static PetscErrorCode Function(SNES, Vec, Vec, void *);
  static PetscErrorCode FunctionAdjoint(SNES, Vec, Vec, void *);

  void InitialConditions_BaseFlow();
  void InitialConditions_Perturbations();
  void InitialConditions_Perturbations_FromAdjointSolution(
      PetscReal *p_energy_0, PetscReal *p_energy_T);

  void UpdateLocalX0(PetscInt modeIdx);
  void UpdateLocalVector(PetscScalar *r_local_vector, Vec p_global_vector);
  void UpdateGlobalVector(PetscScalar *p_local_vector, Vec *r_global_vector);
  void ConfinePerturbations();

  PetscErrorCode StepX0();
  PetscErrorCode StepY0();

  PetscErrorCode SaveCList();

  void SaveLocalGlobalIdx();
  PetscErrorCode SaveGrid();

  SavingRegime _saving_regime;
  PetscErrorCode Save(PetscInt p_time_iteration);
  PetscErrorCode SaveSolution();
  PetscErrorCode SaveForces();
  PetscErrorCode SaveEnergy();
  PetscErrorCode SaveProbes();
  PetscErrorCode SaveNormWeights();

  void CalculateMassFlux();
  PetscErrorCode TestNullSpace(Mat pM, Vec pX);

 public:
  Solver(Input *p_input, Monitor *p_monitor, IEquation *pEq,
         MeshPartition *pMP);
  virtual ~Solver();
  PetscErrorCode Create();

  PetscErrorCode Solve();
  PetscErrorCode Solve_GeneralProblem();
  PetscErrorCode Solve_Floquet();
  PetscErrorCode Solve_OptimalPerurbations();

  void SolveSystem(PetscReal p_time, ProblemType p_problem,
                   SolverType p_solver);

  void Solve_PrimalAdjointCycle(PetscReal *r_energy_0, PetscReal *r_energy_T);
  void Solve_PrimalProblem();
  void Solve_AdjointProblem();
};
}  // namespace O3D
#endif /* SOLVER_H_ */
