#include "mesh.h"

Mesh::Mesh() {}
Mesh::~Mesh() {}

void Mesh::InitializeFromFile(char *pFileName) {
  FILE *fRead;

  fRead = fopen(pFileName, "r");
  if (!fRead) printf("Could not open file: %s\n", pFileName);
  fpos_t pos;
  PetscInt idx, elmType, elmN;
  char *tag = new char[256];

  PetscMPIInt rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  PetscReal buf;

  while (fscanf(fRead, "%s", tag) != EOF) {
    if (strcmp(tag, "$Nodes") == 0) {
      fscanf(fRead, "%ld\n", &_NVertex);
      if (rank == 0)
        printf(
            "\t Memory-1[%d] (request=%ld, _NVertex=%ld, "
            "sizeof(PetscReal)=%ld)\n",
            rank, 3 * _NVertex * sizeof(PetscReal), _NVertex,
            sizeof(PetscReal));

      PetscMalloc(_NVertex * sizeof(PetscReal), &(_X1));
      PetscMalloc(_NVertex * sizeof(PetscReal), &(_X2));

      if (rank == 0) {
        PetscReal sizeMB;
        sizeMB = 3 * _NVertex * sizeof(PetscReal) / 1000000.0;
        printf("\t Memory: %.2lfMB\n", sizeMB);
      }
      for (PetscInt i = 0; i < _NVertex; i++) {
        fscanf(fRead, "%ld", &idx);
        idx--;
        fscanf(fRead, "%lf %lf %lf\n", &_X1[idx], &_X2[idx], &buf);
      }
    } else if (strcmp(tag, "$Elements") == 0) {
      fscanf(fRead, "%ld\n", &elmN);
      fgetpos(fRead, &pos);

      _NElement = 0;
      _NEdge = 0;
      for (PetscInt i = 0; i < elmN; i++) {
        fscanf(fRead, "%*d %ld%*[^\n]\n", &elmType);

        if (elmType == (DIM - 1)) {
          _NEdge++;
        } else if (elmType == 2 * (DIM - 1)) {
          _NElement++;
        }
      }

      if (_NEdge == 0 || _NElement == 0) printf("ERROR: mesh format!\n");

      PetscMalloc((DIM + 1) * _NElement * sizeof(int), &(_VertsElement));
      PetscMalloc(DIM * _NEdge * sizeof(PetscInt), &(_VertsEdge));
      PetscMalloc(_NEdge * sizeof(PetscInt), &(_PhysicalEntity));
      if (rank == 0) {
        PetscReal sizeMB;
        sizeMB = (DIM + 1) *
                 (_NElement * sizeof(int) + _NEdge * sizeof(PetscInt)) /
                 1000000.0;
        printf("\t Memory: %.2lfMB\n", sizeMB);
      }
      _NEdge = 0;
      _NElement = 0;
      fsetpos(fRead, &pos);
      for (PetscInt i = 0; i < elmN; i++) {
        fscanf(fRead, "%*d %ld", &elmType);
        if (elmType == (DIM - 1)) {
          // PetscMalloc(DIM*sizeof(PetscInt),&(_VertsEdge[_NEdge]));
          fscanf(fRead, "%*d %ld %*d\n", &(_PhysicalEntity[_NEdge]));
          for (PetscInt i = 0; i < DIM; i++) {
            fscanf(fRead, "%ld", &(_VertsEdge[DIM * _NEdge + i]));
            _VertsEdge[DIM * _NEdge + i]--;
          }
          fscanf(fRead, "\n");
          _NEdge++;
        } else if (elmType == 2 * (DIM - 1)) {
          fscanf(fRead, "%*d %*d %*d\n");
          for (PetscInt i = 0; i < DIM + 1; i++) {
            fscanf(fRead, "%d", &(_VertsElement[(DIM + 1) * _NElement + i]));
            _VertsElement[(DIM + 1) * _NElement + i]--;
          }
          fscanf(fRead, "\n");
          _NElement++;
        } else {
          fscanf(fRead, "%*[^\n]\n");
        }
      }
    }
  }
  delete[] tag;
  fclose(fRead);

  CalculateMaxCAndVAdjMPI();
}

void Mesh::InitializeRegularGrid2D(PetscInt pM1, PetscInt pM2, PetscReal pH) {
  // ------------------------------------------------------------------------------
  // Количество узлов
  _NVertex = pM1 * pM2;

  // ------------------------------------------------------------------------------
  // Максимальное количество соседей-узлов и соседей-элементов
  _MaxVAdj = 6;
  _MaxEAdj = 6;

  // ------------------------------------------------------------------------------
  // Координаты узлов
  PetscMalloc(_NVertex * sizeof(PetscReal), &(_X1));
  PetscMalloc(_NVertex * sizeof(PetscReal), &(_X2));

  PetscInt idxV;
  for (idxV = 0; idxV < _NVertex; idxV++) {
    _X1[idxV] = (PetscReal)(pH * (idxV / pM1));
    _X2[idxV] = (PetscReal)(pH * (idxV % pM1));
  }

  // ------------------------------------------------------------------------------
  // Количество элементов и составляющие их узлы
  _NElement = 2 * (pM1 - 1) * (pM2 - 1);
  PetscMalloc((DIM + 1) * _NElement * sizeof(PetscInt), &(_VertsElement));

  for (PetscInt e = 0; e < _NElement; e++) {
    _VertsElement[(DIM + 1) * e + 0] = e / 2 + e / (2 * (pM1 - 1));

    if (e % 2 == 0) {
      _VertsElement[(DIM + 1) * e + 1] = _VertsElement[(DIM + 1) * e + 0] + pM1;
      _VertsElement[(DIM + 1) * e + 2] = _VertsElement[(DIM + 1) * e + 1] + 1;
    } else {
      _VertsElement[(DIM + 1) * e + 1] =
          _VertsElement[(DIM + 1) * e + 0] + pM1 + 1;
      _VertsElement[(DIM + 1) * e + 2] = _VertsElement[(DIM + 1) * e + 0] + 1;
    }
  }

  // ------------------------------------------------------------------------------
  // Количество граней и составляющие их узлы
  _NEdge = 2 * (pM1 + pM2) - 4;
  PetscMalloc(DIM * _NEdge * sizeof(PetscInt), &(_VertsEdge));
  for (PetscInt e = 0; e < _NEdge; e++) {
    if (e < pM1 - 1) {
      _VertsEdge[DIM * e + 0] = e;
      _VertsEdge[DIM * e + 1] = _VertsEdge[DIM * e + 0] + 1;
    } else if (e < pM1 + pM2 - 2) {
      _VertsEdge[DIM * e + 0] = pM1 - 1 + pM1 * (e - (pM1 - 1));
      _VertsEdge[DIM * e + 1] = _VertsEdge[DIM * e + 0] + pM1;
    } else if (e < 2 * pM1 + pM2 - 3) {
      _VertsEdge[DIM * e + 0] = pM1 * pM2 - e + pM1 + pM2 - 3;
      _VertsEdge[DIM * e + 1] = _VertsEdge[DIM * e + 0] - 1;
    } else if (e < 2 * (pM1 + pM2) - 4) {
      _VertsEdge[DIM * e + 0] =
          pM1 * (pM2 - 1) - pM1 * (e - 2 * (pM1 - 1) - (pM2 - 1));
      _VertsEdge[DIM * e + 1] = _VertsEdge[DIM * e + 0] - pM1;
    }
  }

  // ------------------------------------------------------------------------------
  // Номера физических границ
  PetscMalloc(_NEdge * sizeof(PetscInt), &(_PhysicalEntity));
  for (PetscInt e = 0; e < _NEdge; e++) {
    if (e < pM1 - 1) {
      _PhysicalEntity[e] = 1;
    } else if (e < pM1 + pM2 - 2) {
      _PhysicalEntity[e] = 2;
    } else if (e < 2 * pM1 + pM2 - 3) {
      _PhysicalEntity[e] = 3;
    } else if (e < 2 * (pM1 + pM2) - 4) {
      _PhysicalEntity[e] = 4;
    }
  }
}
void Mesh::Finalize() {
  PetscFree(_X1);
  PetscFree(_X2);
  PetscFree(_VertsElement);
  PetscFree(_VertsEdge);
  PetscFree(_PhysicalEntity);
}

void Mesh::GetVByE(PetscInt pElementIdx, PetscInt *rVerts) {
  for (PetscInt i = 0; i < DIM + 1; i++)
    rVerts[i] = _VertsElement[(DIM + 1) * pElementIdx + i];
}
void Mesh::GetEByV(PetscInt pVrtxIdx, PetscInt *rEN, PetscInt *rE) {
  (*rEN) = 0;
  for (PetscInt i = 0; i < _NElement; i++) {
    for (PetscInt j = 0; j < DIM + 1; j++) {
      if (pVrtxIdx == _VertsElement[(DIM + 1) * i + j]) {
        if (rE != NULL) rE[(*rEN)] = i;

        (*rEN)++;
      }
    }
  }
}
void Mesh::GetEByV(PetscInt pVrtxIdx, PetscInt *rEN) {
  (*rEN) = 0;
  for (PetscInt i = 0; i < _NElement; i++) {
    for (PetscInt j = 0; j < DIM + 1; j++) {
      if (pVrtxIdx == _VertsElement[(DIM + 1) * i + j]) {
        (*rEN)++;
      }
    }
  }
}
void Mesh::GetVByV(PetscInt pVrtxIdx, PetscInt *rVN, PetscInt *rV) {
  PetscBool alreadyinclude;
  PetscInt EN, *E, *V;

  (*rVN) = 0;

  GetEByV(pVrtxIdx, &EN);
  PetscMalloc(DIM * EN * sizeof(PetscInt), &V);
  PetscMalloc(DIM * EN * sizeof(PetscInt), &E);
  GetEByV(pVrtxIdx, &EN, E);
  for (PetscInt i = 0; i < EN; i++) {
    for (PetscInt j = 0; j < DIM + 1; j++) {
      if (_VertsElement[(DIM + 1) * E[i] + j] != pVrtxIdx) {
        alreadyinclude = PETSC_FALSE;
        for (PetscInt k = 0; k < (*rVN); k++) {
          if (V[k] == _VertsElement[(DIM + 1) * E[i] + j]) {
            alreadyinclude = PETSC_TRUE;
            break;
          }
        }
        if (alreadyinclude == PETSC_FALSE) {
          V[(*rVN)] = _VertsElement[(DIM + 1) * E[i] + j];
          (*rVN)++;
        }
      }
    }
  }
  if (rV != NULL) {
    for (PetscInt i = 0; i < (*rVN); i++) {
      rV[i] = V[i];
    }
  }

  PetscFree(V);
  PetscFree(E);
}

void Mesh::CalculateMaxCAndVAdjMPI() {
  PetscMPIInt rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, 