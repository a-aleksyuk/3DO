#ifndef MESH_H_
#define MESH_H_

#include "element.h"
#include "structures.h"

class Mesh {
 public:
  PetscInt _NVertex;   // Number of nodes
  PetscInt _NElement;  // Number of cells
  PetscInt _NEdge;     // Number of edges

  PetscInt _MaxVAdj, _MaxEAdj;  // Max number of node and cell neighbours

  PetscReal *_X1, *_X2;  // (x, y) coordinates

  int *_VertsElement;    // Cell nodes
  PetscInt *_VertsEdge;  // Edge nodes

  PetscInt *_PhysicalEntity;  // Physical entities

 public:
  Mesh();
  virtual ~Mesh();

  void InitializeFromFile(char *pFileName);
  void InitializeRegularGrid2D(PetscInt M1, PetscInt M2, PetscReal H);

  void Finalize();

  void GetVByE(PetscInt pElementIdx, PetscInt *rVerts);
  void GetEByV(PetscInt pVrtxIdx, PetscInt *rEN, PetscInt *rE);
  void GetEByV(PetscInt pVrtxIdx, PetscInt *rEN);
  void GetVByV(PetscInt pVrtxIdx, PetscInt *rVN, PetscInt *rV);
  void CalculateMaxCAndVAdjMPI();
  void CalculateMaxCAndVAdj();

  // Boundaries
  PetscBool IsBorder(PetscInt pVrtxIdx);
  PetscInt GetEByEdge(PetscInt pEdgeIdx);
  void GetVByEdge(PetscInt pEdgeIdx, PetscInt *rVerts);
  void GetEdgesByV(PetscInt pVrtxIdx, PetscInt *rNEdges, PetscInt *rEdges);
  void GetNormalByEdge(PetscInt pEdgeIdx, PetscReal **rNormal);
  void GetEdgeS(PetscInt pEdgeIdx, PetscReal *rS);

  PetscInt GetPhysicalEntity(PetscInt pVrtxIdx);
};

#endif /* MESH_H_ */
