#ifndef MESHPARTITION_H_
#define MESHPARTITION_H_

#include <parmetis.h>
#include <petscao.h>
#include <petsctime.h>

#include <iostream>

#include "mesh.h"
#include "structures.h"
using namespace std;

class MeshPartition {
 public:
  // Разбиение
  AO _AO;
  PetscInt _Nvglobal, _Nvlocal, _Nneighbors;
  PetscInt _Nelocal;

  // Элементы
  PetscInt *_EGloInd;
  PetscInt *_NEByV;
  PetscInt **_EByV;
  PetscInt **_VByE;

  // Граница
  PetscInt _NEdgesLocal;  // количество граней на текущем процессе
  PetscInt *_GloEdgeIdx;  // глобальный индекс грани
  PetscInt *_BorderNEdgesByV;  // количество граней содержащих узел
                               // (0-внутренняя точка)
  PetscInt **_BorderEdgesIDsByV;  // индексы граней содержащих узел
                                  // (NULL-внутренняя точка)
  PetscInt *_BorderElement;  // элемент составляющий грань (ребро)

  PetscInt *_EdgeByElement;  //

  PetscInt **_BorderVerts;  // индексы узлов принадлежащих грани

  PetscReal **_BorderN;  // вектор нормали на грани (ребре)
  PetscReal *_BorderS;  // площадь грани (длина ребра)

  PetscInt *_LocInd, *_GloInd;
  PetscInt *_V2P;  // точка -> процесс
  PetscInt **_AdjM;
  PetscInt *_ITot;
  PetscInt _NVertices;  // количество локальных узлов, включая фиктивные
  PetscInt *_Vertices;
  PetscInt _NMaxAdj;  // максимальное число соседних узлов

  // Вспомогательные переменные
  PetscReal *_cell_volume;  // Объемы
  PetscReal *_Hx;           // Производные функции формы
 private:
  // Сетка
  Mesh *_Mesh;

  // Переменные MPI
  MPI_Comm _Comm;
  PetscMPIInt _Size, _Rank;

  // =============================================
  // Parmetis
  idx_t *vtxdist;

  idx_t *xadj;
  idx_t *adjncy;

  idx_t *vwgt;
  idx_t *adjwgt;

  idx_t *part;
  idx_t wgtflag;
  idx_t numflag;
  idx_t ncon;
  idx_t ncommonnodes;
  real_t *tpwgts, ubvec;
  idx_t options[4], edgecut;
  idx_t nparts;
  real_t itr;
  idx_t *vsize;
  // =============================================

 public:
  MeshPartition(MPI_Comm pComm, Mesh *pMesh);
  virtual ~MeshPartition();
  void CreatePartition();
  void InitAuxVars(PetscReal *pX1, PetscReal *pX2);
  void FreeAuxVars();

  void ShowResults(FILE *fptr);

 private:
  void Partition();
  void FreeParmetis();
  PetscErrorCode CreateBorderParameters();
  PetscErrorCode CreateLocalNumbering();
  PetscErrorCode CreateGhostVertices();

  PetscErrorCode CheckNormals();
  void ShowInitialize();
  void Sort(PetscInt Length, PetscInt *pArray);
};

#endif /* MESHPARTITION_H_ */
