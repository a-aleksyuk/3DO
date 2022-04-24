#ifndef NAVIERSTOKES_H_
#define NAVIERSTOKES_H_

#include <fftw3.h>

#include "element.h"
#include "input.h"
#include "monitor.h"
#include "structures.h"

namespace O3D {

class NavierStokes : public IEquation {
 private:
  PetscMPIInt _Rank;
  Input *_input;
  Monitor *_monitor;

  PetscInt _NGroups;
  PetscInt *_VertsMap;
  PetscBool *_gEIdx;
  PetscScalar *_gY;
  PetscScalar *_gY_t;

  PetscScalar *_aF;
  PetscScalar *_aF1;
  PetscScalar *_aDF;
  PetscInt *_jRows;
  PetscScalar *_jValues;

  PetscScalar **_uY_x;
  PetscScalar **_uY_x_base;

  PetscScalar *mW;
  PetscScalar *mWi;
  PetscScalar *mWGLS;

  PetscScalar *_mW;
  PetscScalar *_mWi;
  PetscScalar *_mWGLS;
  PetscScalar *_mY_baseflow;

  PetscReal *_eH;
  PetscReal *_eTauPSPG;
  PetscReal *_eTauSUPG;
  PetscReal *_eTauLSIC;

  PetscScalar *_C1;
  PetscScalar *aC1;
  PetscScalar *_quadWi;
  PetscScalar *quadWi;

  // -------------------------------------------
  // Aux arrays for nonlinear term calculations
  fftw_plan _FFTW_PlanC2R_U, _FFTW_PlanC2R_Uz, _FFTW_PlanR2C_N;

  fftw_complex *_fs_U;      // Velocity in the Fourier space
  PetscReal *_ps_U;         // Velocity in the physical space
  fftw_complex *_fs_Uz;     // z-derivative of velocity in the Fourier space
  PetscReal *_ps_Uz;        // z-derivative of velocity in the physical space
  PetscReal *_ps_U_x[DIM];  // Derrivatives of velocity

  PetscBool _fs_N0_initialized;
  fftw_complex *_fs_N0;  // Convective term in the Fourier space
  fftw_complex *_fs_N;   // Convective term in the Fourier space
  PetscReal *_ps_N;      // Convective term in the physical space

  fftw_complex *_fs_J;  // Matrix J in the Fourier space: all the blocks (4x4)
                        // at each time
  fftw_complex *_fs_J_backward;  // Matrix J (backward) in the Fourier space:
                                 // all the blocks (4x4) at each time
  PetscBool _jacobi_matrix_alias_enabled;
  PetscScalar *_jacobi_matrix_alias;
  PetscScalar *_exp_t;

  PetscReal *_pScaleRows;
  PetscReal _pScale;

 public:
  NavierStokes(Input *pInp, Monitor *p_monitor);
  virtual ~NavierStokes();

  PetscErrorCode UpdateBaseFlow(void *ptr, PetscReal pT);
  void InitializeJacobianInterpolation(void *p_user);
  void FinalizeJacobianInterpolation();

  PetscErrorCode Jacobian(SNES pSNES, Vec pX, Mat, Mat, void *);
  PetscErrorCode JacobianPrimal(SNES pSNES, Vec pX, Mat, Mat, void *);
  PetscErrorCode JacobianPrimalPeriodic(SNES pSNES, Vec pX, Mat, Mat, void *);
  PetscErrorCode JacobianAdjoint(SNES pSNES, Vec pX, Mat, Mat, void *);
  PetscErrorCode JacobianNonZeros(SNES, Vec, Mat, Mat, void *);

  PetscErrorCode SaveAux(PetscInt ElIdx, void *ptr);
  PetscErrorCode RestoreAux(PetscInt pVIdx, void *ptr);

  PetscErrorCode Function(SNES, Vec, Vec, void *);
  PetscErrorCode FunctionAdjoint(SNES, Vec, Vec, void *);
  PetscErrorCode InitialConditions(PetscInt modeIdx, Vec X, void *);

  PetscErrorCode InitAuxVars(void *ptr);

  PetscErrorCode UpdateAuxVars(AppCtx *user, PetscInt pVrtx);
  PetscErrorCode UpdateAuxVars_Complex(AppCtx *user, PetscInt pVrtx);
  PetscErrorCode UpdateAuxVars_Real(AppCtx *user, PetscInt pVrtx);

  PetscErrorCode NonlinearTermFFT(void *ptr);

  PetscErrorCode DestroyAuxVars();
  PetscScalar WeakFormMain(PetscInt pH, AppCtx *user);
  PetscScalar WeakFormDomainBorder(PetscInt pH, AppCtx *user);

  /////////////////////////////////////////////
  // Stabilisation parameters
  /////////////////////////////////////////////
  void GetTau(const PetscScalar *pY, PetscInt pElementIdx, AppCtx *user,
              PetscReal *rTauPSPG, PetscReal *rTauSUPG, PetscReal *rTauLSIC);

  PetscErrorCode TestNullSpace(Mat pM, Vec pX);

  PetscScalar ComplexMult(fftw_complex p_fftw_complex,
                          PetscScalar p_petsc_complex);
};

}  // namespace O3D

#endif /* NAVIERSTOKES_H_ */
