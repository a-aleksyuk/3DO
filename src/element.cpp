#include "element.h"

#include "structures.h"

PetscReal Element::VolumeDim(PetscReal *x1, PetscReal *x2,
                             PetscInt pObjectDim) {
  if (pObjectDim == 1)
    return PetscSqrtReal((x1[1] - x1[0]) * (x1[1] - x1[0]) +
                         (x2[1] - x2[0]) * (x2[1] - x2[0]));
  else if (pObjectDim == 2)
    return 0.5 * PetscAbsScalar((x1[1] - x1[0]) * (x2[2] - x2[0]) -
                                (x2[1] - x2[0]) * (x1[2] - x1[0]));

  return 0.0;
}

PetscReal Element::VolumeDim(PetscReal *x1, PetscReal *x2, PetscInt pObjectDim,
                             PetscInt *pI) {
  if (pObjectDim == 1) {
    PetscReal a1 = (x1[pI[1]] - x1[pI[0]]);
    PetscReal a2 = (x2[pI[1]] - x2[pI[0]]);
    return PetscSqrtReal(a1 * a1 + a2 * a2);
  } else if (pObjectDim == 2) {
    PetscReal a1 = (x1[pI[1]] - x1[pI[0]]);
    PetscReal a2 = (x2[pI[1]] - x2[pI[0]]);
    PetscReal b1 = (x1[pI[2]] - x1[pI[0]]);
    PetscReal b2 = (x2[pI[2]] - x2[pI[0]]);
    return 0.5 * PetscAbsScalar(a1 * b2 - a2 * b1);
  }

  return 0.0;
}

PetscScalar Element::Quad(PetscScalar *f, PetscReal V) {
  PetscInt i;
  PetscScalar r = 0.0;
  for (i = 0; i < D1; i++) {
    r += f[i];
  }
  return V * r / ((PetscReal)D1);
}

PetscScalar Element::Quad(PetscScalar *f, PetscScalar *g, PetscReal V) {
  PetscInt i;
  PetscScalar k = 1.0 / (2.0 * DIM * (DIM + 1.0));

  PetscScalar rFG = 0.0;
  PetscScalar rG = 0.0;
  PetscScalar rF = 0.0;
  for (i = 0; i < D1; i++) {
    rFG += f[i] * g[i];
    rF += f[i];
    rG += g[i];
  }
  return V * k * (((PetscReal)(DIM - 1)) * rFG + rF * rG);
}

PetscScalar Element::Quad_FHat(PetscScalar *f, PetscInt iHat, PetscReal V,
                               PetscInt pDim) {
  PetscInt i;
  PetscScalar k = 1.0 / (2.0 * pDim * (pDim + 1.0));

  PetscScalar rF = 0.0;
  for (i = 0; i < pDim + 1; i++) rF += f[i];

  return V * k * ((pDim - 1) * f[iHat] + rF);
}

PetscReal Element::H_x(PetscInt pH, PetscInt pXi, PetscReal *x, PetscReal *y,
                       PetscInt *pI) {
  PetscInt i0, i1, i2;
  PetscReal d1, d2;

  i0 = pI[pH];
  i1 = pI[(pH + 1) % (D1)];
  i2 = pI[(pH + 2) % (D1)];

  d1 = (x[i1] - x[i0]) * (y[i2] - y[i0]) - (x[i2] - x[i0]) * (y[i1] - y[i0]);

  if (pXi == 1)
    d2 = y[i1] - y[i2];
  else
    d2 = x[i2] - x[i1];

  return d2 / d1;
}

void Element::Y_x(PetscInt pXi, PetscInt pElementIdx, PetscReal *Hx,
                  const PetscScalar *Y, PetscInt *pI, PetscScalar *rY_x) {
  if (pXi == DIM) {
    for (PetscInt i = 0; i < NVAR; i++) rY_x[i] = 0.0;
  } else {
    for (PetscInt i = 0; i < NVAR; i++) {
      rY_x[i] = 0.0;
      for (PetscInt k = 0; k < D1; k++) {
        rY_x[i] += Y[NVAR * pI[k] + i] * Hx[D1_D * pElementIdx + DIM * k + pXi];
      }
    }
  }
}

void Element::Y_x(PetscInt pXi, PetscInt pElementIdx, PetscReal *Hx,
                  PetscScalar *Y, PetscInt *pI, PetscScalar *rY_x) {
  if (pXi == DIM) {
    for (PetscInt i = 0; i < NVAR; i++) rY_x[i] = 0.0;
  } else {
    for (PetscInt i = 0; i < NVAR; i++) {
      rY_x[i] = 0.0;
      for (PetscInt k = 0; k < D1; k++) {
        rY_x[i] += Y[NVAR * pI[k] + i] * Hx[D1_D * pElementIdx + DIM * k + pXi];
      }
    }
  }
}
