#ifndef EQUATION_H_
#define EQUATION_H_

#include <petscsnes.h>

class IEquation {
 public:
  virtual ~IEquation();

  virtual PetscErrorCode UpdateBaseFlow(void* p_user, PetscReal pT);
  virtual void InitializeJacobianInterpolation(void* p_user);
  virtual void FinalizeJacobianInterpolation();

  virtual PetscErrorCode Jacobian(SNES, Vec, Mat, Mat, void*);
  virtual PetscErrorCode JacobianAdjoint(SNES, Vec, Mat, Mat, void*);
  virtual PetscErrorCode JacobianNonZeros(SNES, Vec, Mat, Mat, void*);
  virtual PetscErrorCode Function(SNES, Vec, Vec, void*);
  virtual PetscErrorCode FunctionAdjoint(SNES, Vec, Vec, void*);
  virtual PetscErrorCode InitialConditions(PetscInt modeIdx, Vec X, void*);

  virtual PetscErrorCode NonlinearTermFFT(void*);

  virtual PetscErrorCode InitAuxVars(void*);
  virtual PetscErrorCode DestroyAuxVars();

  virtual PetscReal tFunction();
  virtual PetscReal tJacobian();
  virtual PetscReal tJacobianLoop();
  virtual PetscReal tWeakFormMain();
  virtual PetscReal tWeakFormDomainBorder();
  virtual PetscReal tUpdateAuxVars();
};

#endif /* EQUATION_H_ */
