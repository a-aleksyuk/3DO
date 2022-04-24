#ifndef ELEMENT_H_
#define ELEMENT_H_

#include "structures.h"

class Element {
 public:
  static PetscReal VolumeDim(PetscReal *x1, PetscReal *x2, PetscInt pObjectDim);
  static PetscReal VolumeDim(PetscReal *x1, PetscReal *x2, PetscInt pObjectDim,
                             PetscInt *pI);

  static PetscScalar Quad(PetscScalar *f, PetscScalar *g, PetscReal V);
  static PetscScalar Quad_FHat(PetscScalar *f, PetscInt iHat, PetscReal V,
                               PetscInt pDim);
  static PetscScalar Quad(PetscScalar *f, PetscReal V);

  static PetscReal H_x(PetscInt pH, PetscInt pXi, PetscReal *x, PetscReal *y,
                       PetscInt *pI);

  static void Y_x(PetscInt pXi, PetscInt pElementIdx, PetscReal *Hx,
                  const PetscScalar *Y, PetscInt *pI, PetscScalar *rY_x);
  static void Y_x(PetscInt pXi, PetscInt pElementIdx, PetscReal *Hx,
                  PetscScalar *Y, PetscInt *pI, PetscScalar *rY_x);
};

#endif /* ELEMENT_H_ */
