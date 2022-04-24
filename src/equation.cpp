#include "equation.h"

IEquation::~IEquation() {}

PetscErrorCode IEquation::UpdateBaseFlow(void *p_user, PetscReal pT) {
  return 0;
}
void IEquation::InitializeJacobianInterpolation(void *p_user) {}
void IEquation::FinalizeJacobianInterpolation() {}

PetscErrorCode IEquation::Jacobian(SNES snes, Vec X, Mat J, Mat B, void *ptr) {
  return 0;
}
PetscErrorCode IEquation::JacobianAdjoint(SNES snes, Vec X, Mat J, Mat B,
                                          void *ptr) {
  return 0;
}
PetscErrorCode IEquation::JacobianNonZeros(SNES, Vec, Mat, Mat, void *) {
  return 0;
}
PetscErrorCode IEquation::Function(SNES snes, Vec X, Vec F, void *ptr) {
  return 0;
}
PetscErrorCode IEquation::FunctionAdjoint(SNES snes, Vec X, Vec F, void *ptr) {
  return 0;
}
PetscErrorCode IEquation::InitialConditions(PetscInt modeIdx, Vec X,
                                            void *ptr) {
  return 0;
}

PetscErrorCode IEquation::NonlinearTermFFT(void *) { return 0; }

PetscErrorCode IEquation::InitAuxVars(void *) { return 0; }
PetscErrorCode IEquation::DestroyAuxVars() { return 0; }

PetscReal IEquation::tFunction() { return 0; }
PetscReal IEquation::tJacobian() { return 0; }
PetscReal IEquation::tJacobianLoop() { return 0; }
PetscReal IEquation::tWeakFormMain() { return 0; }
PetscReal IEquation::tWeakFormDomainBorder() { return 0; }
PetscReal IEquation::tUpdateAuxVars() { return 0; }
