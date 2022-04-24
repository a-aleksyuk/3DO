#ifndef STRUCTURES_H_
#define STRUCTURES_H_

#include <petscsnes.h>

#include <Eigen/Eigenvalues>

#include "equation.h"

namespace O3D {
#define PI 3.14159265359

#define DEBUGLEVEL 3
#define TAU_PRECISION 0.00000001
#define EPS_F_DERIVATIVE 0.00000001
#define DIM 2
#define NVAR 4

#define N2 16     // = NVAR*NVAR;
#define D1 3      // = (DIM+1);
#define D_N2 32   // = DIM*NVAR*NVAR;
#define D1_N 12   // = (DIM+1)*NVAR;
#define D1_D 6    // = (DIM+1)*DIM;
#define D1_D1 9   // = (DIM+1)*(DIM+1);
#define D2_N2 64  // = DIM*DIM*NVAR*NVAR;

extern void PressEnterToContinue();

typedef struct {
  MPI_Comm comm;  // MPI communicator
  PetscReal T;    // Current time
  IEquation *Eq;  // Equation

  // = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
  // Mesh related parameters
  PetscInt Nvglobal;    // Number of global nodes (on process)
  PetscInt Nvlocal;     // Number of local nodes (on process)
  PetscInt NvGhost;     // Number of ghost nodes (on process)
  PetscInt Nneighbors;  // Total number of neighbors (on process)
  PetscInt Nelocal;     // Number of cells (on process)

  PetscInt *NEByV;  // Number of cells sharing a node
  PetscInt **EByV;  // Indexes of cells sharing a node
  PetscInt **VByE;  // Indexes of nodes sharing a cell

  PetscInt *ITot;    // Number of node neighbors
  PetscInt NMaxAdj;  // Max number of node neighbors
  PetscInt **AdjM;   // Indexes of node neighbors

  PetscInt NEdgesLocal;  // Number of edges (on process)
  PetscInt *GloEdgeIdx;  // Global edge index

  PetscInt *BorderNEdgesByV;  // Number of edges sharing a node (0 - inner node)
  PetscInt **BorderEdgesIDsByV;  // Indexes of edges sharing a node
  PetscInt *BorderElement;       // Cell index by boundary edge
  PetscInt *EdgeByElement;       // Edge index by cell
  PetscInt **BorderVerts;        // Indexes of boundary nodes

  PetscReal **BorderN;  // Normal vector by edge
  PetscReal *BorderS;   // Length of the edge
  PetscReal *MassFlux;  // Boundary mass flux

  PetscInt *UniqueEdgeIndex;  // -1: edge is stored on another process

  PetscInt *LocInd;  // Global indexes of nodes
  PetscInt *GloInd;  // Local indexes of nodes

  VecScatter Scatter;               // Nodes scatter on processes (with ghosts)
  VecScatter ScatterWithoutGhosts;  // Nodes scatter on processes

  PetscReal *X1, *X2;  // (x,y) coordinates

  PetscReal *cell_volume;  // Cell volume
  PetscReal *Hx;           // Derrivatives of shape functions

  // = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
  // Fourier modes
  PetscInt FM;     // Amount of modes
  PetscInt NZ;     // Amount of z cross-sections
  PetscInt CurFM;  // Current mode

  // = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
  // System of algebraic equations
  Mat _mat_J_aux;
  Mat _matA;   // Forward Jacobian matrix
  Mat _matAH;  // Forward Jacobian matrix transposed
  Mat _matB;   // Backward Jacobian matrix
  Mat _matBH;  // Backward Jacobian matrix transposed

  Vec *GlobalX;    // Solution (global)
  Vec GlobalR;     // Resuidal (global)
  Vec GlobalAux;   // Aux global vector
  Vec GlobalAux1;  // Aux global vector

  Vec *GlobalY;
  Vec GlobalY0;
  Vec *GlobalBY0;

  Vec LocalX;  // Solution (local part)
  Vec LocalF;  // Resuidal (local part)

  PetscInt lN;
  PetscScalar *lX, *lX0, *lX_t, *lX00, *lX1_e;
  PetscScalar *lY0, *G, *G0;

  // = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
  // Solver
  PetscBool BackwardsInTime;  // Solve backwards in time
  PetscBool FirstTimeStep;    // Is it the first time iteration

  PetscBool FreezeJac;  // To calculate Jacobian only once (for linear systems)

  PetscBool InitialConditionsY;

} AppCtx;

}  // namespace O3D

#endif /* STRUCTURES_H_ */
