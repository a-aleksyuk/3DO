#include "meshpartition.h"

MeshPartition::MeshPartition(MPI_Comm pComm, Mesh *pMesh) {
  MPI_Comm_size(pComm, &_Size);
  MPI_Comm_rank(pComm, &_Rank);

  MPI_Comm_dup(pComm, &_Comm);

  _Mesh = pMesh;
  _Nvglobal = _Mesh->_NVertex;
}

MeshPartition::~MeshPartition() {
  AODestroy(&_AO);
  PetscFree(_Vertices);
  PetscFree(_GloInd);
  PetscFree(_LocInd);
  PetscFree(_ITot);

  for (PetscInt i = 0; i < _NVertices; i++) {
    PetscFree(_EByV[i]);
    PetscFree(_AdjM[i]);
  }

  PetscFree(_AdjM);
  PetscFree(_EByV);
  PetscFree(_NEByV);

  PetscFree(_EGloInd);
  for (PetscInt i = 0; i < _Nelocal; i++) {
    PetscFree(_VByE[i]);
  }
  PetscFree(_VByE);

  PetscFree(_BorderNEdgesByV);
  PetscFree(_BorderElement);
  PetscFree(_BorderS);
  PetscFree(_EdgeByElement);

  PetscFree(_GloEdgeIdx);
  for (PetscInt i = 0; i < _Nvlocal; i++) {
    PetscFree(_BorderEdgesIDsByV[i]);
  }
  PetscFree(_BorderEdgesIDsByV);

  for (PetscInt i = 0; i < _NEdgesLocal; i++) {
    PetscFree(_BorderVerts[i]);
    PetscFree(_BorderN[i]);
  }
  PetscFree(_BorderVerts);
  PetscFree(_BorderN);

  FreeAuxVars();
}

void MeshPartition::CreatePartition() {
  PetscMPIInt rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  PetscReal t0 = 0, t1 = 0;

  if (rank == 0) {
    PetscTime(&t0);
    printf(" # MeshPartition-PARMETIS-Start\n");
  }

  ////////////////////////////////////////////////////////////////////////
  // Определение переменных Parmetis
  if (rank == 0) {
    printf("\t InitAuxParmetis...\n");
  }
  numflag = 0;
  ncon = 1;
  ncommonnodes = DIM;
  ubvec = 1.05;
  options[0] = 0;
  options[1] = 0;
  options[2] = 0;
  options[3] = 0;
  itr = 1000.0;
  vsize = NULL;
  adjwgt = NULL;

  nparts = _Size;
  tpwgts = new real_t[nparts];
  for (PetscInt i = 0; i < nparts; i++) {
    tpwgts[i] = 1.0 / ((double)nparts);
  }

  PetscInt q = _Nvglobal / _Size;
  vtxdist = new idx_t[_Size + 1];
  for (PetscInt i = 0; i < _Size; i++) {
    vtxdist[i] = i * q;
  }
  vtxdist[_Size] = _Nvglobal;

  vwgt = NULL;
  wgtflag = 0;

  xadj = new idx_t[vtxdist[_Rank + 1] - vtxdist[_Rank] + 1];
  part = new idx_t[vtxdist[_Rank + 1] - vtxdist[_Rank]];
  xadj[0] = 0;
  PetscInt *adj;
  PetscInt *adjncytmp;
  PetscMalloc(_Mesh->_MaxVAdj * sizeof(PetscInt), &adj);
  PetscMalloc(_Mesh->_MaxVAdj * (vtxdist[_Rank + 1] - vtxdist[_Rank]) *
                  sizeof(PetscInt),
              &adjncytmp);

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    printf("\t CalcAuxParmetis... i=1, ... , %ld\n",
           (vtxdist[_Rank + 1] - vtxdist[_Rank] + 1));
  }

  // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  for (PetscInt i = 1; i < vtxdist[_Rank + 1] - vtxdist[_Rank] + 1; i++) {
    part[i - 1] = (idx_t)_Rank;
    xadj[i] = 0;
  }
  PetscInt vtxFirst = vtxdist[_Rank];
  PetscInt vtxLast = vtxdist[_Rank + 1] - 1;
  PetscInt curVtx;
  PetscInt curIdx;
  PetscInt newAdjVtx;
  PetscBool alreadyinclude;
  for (PetscInt i = 0; i < _Mesh->_NElement; i++) {
    for (PetscInt j = 0; j < DIM + 1; j++) {
      curVtx = _Mesh->_VertsElement[(DIM + 1) * i + j];
      if (vtxFirst <= curVtx && vtxLast >= curVtx) {
        curIdx = curVtx - vtxFirst;
        for (PetscInt k = 0; k < DIM + 1; k++) {
          newAdjVtx = _Mesh->_VertsElement[(DIM + 1) * i + k];

          if (newAdjVtx != curVtx) {
            // Если новой точки еще нет в массиве соседей, то включаем ее
            alreadyinclude = PETSC_FALSE;

            // Цикл по уже добавленным к узлу соседям
            for (PetscInt s = 0; s < xadj[curIdx + 1]; s++) {
              if (adjncytmp[_Mesh->_MaxVAdj * curIdx + s] == newAdjVtx) {
                alreadyinclude = PETSC_TRUE;
                break;
              }
            }
            if (alreadyinclude == PETSC_FALSE) {
              adjncytmp[_Mesh->_MaxVAdj * curIdx + xadj[curIdx + 1]] =
                  newAdjVtx;
              xadj[curIdx + 1]++;
            }
          }
        }
      }
    }
  }
  for (PetscInt i = 1; i < vtxdist[_Rank + 1] - vtxdist[_Rank] + 1; i++) {
    xadj[i] = xadj[i - 1] + xadj[i];
  }
  adjncy = new idx_t[xadj[vtxdist[_Rank + 1] - vtxdist[_Rank]]];
  PetscInt nAdj = 0;
  for (PetscInt j = 0; j < (vtxdist[_Rank + 1] - vtxdist[_Rank]); j++) {
    for (PetscInt k = 0; k < xadj[j + 1] - xadj[j]; k++) {
      adjncy[nAdj] = adjncytmp[_Mesh->_MaxVAdj * j + k];
      nAdj++;
    }
  }
  // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  PetscFree(adj);
  PetscFree(adjncytmp);
  ////////////////////////////////////////////////////////////////////////
  // Выполнить разбиение Parmetis
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    printf("\t Partition...\n");
  }

  Partition();
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    printf("\t FreeParmetis...\n");
  }
  FreeParmetis();
  /////////////////////////////////////////////////////////////////////
  // Создание локальной нумерации и фиктивных узлов
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    PetscTime(&t1);
    printf(" # MeshPartition-PARMETIS-End (dT=%g sec)\n\n", (t1 - t0));
    printf(" # MeshPartition-MeshParameters-Start\n");
  }
  if (rank == 0) printf("\t CreateLocalNumbering...\n");
  CreateLocalNumbering();
  if (rank == 0) printf("\t CreateGhostVertices...\n");
  CreateGhostVertices();
  if (rank == 0) printf("\t CreateBorderParameters...\n");
  CreateBorderParameters();

  PetscMalloc(_Nelocal * sizeof(PetscInt), &_EdgeByElement);
  for (PetscInt i = 0; i < _Nelocal; i++) {
    _EdgeByElement[i] = -1;
    for (PetscInt j = 0; j < _NEdgesLocal; j++) {
      if (_BorderElement[j] == i) {
        _EdgeByElement[i] = j;
        break;
      }
    }
  }
  if (rank == 0) printf("\t CheckNormals...\n");
  CheckNormals();
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    PetscTime(&t0);
    printf(" # MeshPartition-MeshParameters-End (dT=%g sec)\n\n", -(t1 - t0));
  }
}

void MeshPartition::Partition() {
  MPI_Barrier(MPI_COMM_WORLD);
  if (_Rank == 0) {
    printf("\t\t PartMeshKway-Start...\n");
    printf("\t\t\tsizeof(PetscInt)=%ld, sizeof(idx_t)=%ld, sizeof(int)=%ld\n",
           sizeof(PetscInt), sizeof(idx_t), sizeof(int));
    printf(
        "\t\t\tsizeof(PetscReal)=%ld, sizeof(real_t)=%ld, sizeof(double)=%ld\n",
        sizeof(PetscReal), sizeof(real_t), sizeof(double));

    printf("\t\t\t");
    int size;
    MPI_Type_size(MPI_DOUBLE, &size);
    printf("sizeof(MPI_DOUBLE)=%d, ", size);
    MPI_Type_size(MPI_INT, &size);
    printf("sizeof(MPI_INT)=%d, ", size);
    MPI_Type_size(MPI_LONG_LONG, &size);
    printf("sizeof(MPI_LONG_LONG)=%d, ", size);
    MPI_Type_size(MPI_LONG_LONG_INT, &size);
    printf("sizeof(MPI_LONG_LONG_INT)=%d\n", size);
  }
  ParMETIS_V3_PartMeshKway(vtxdist, xadj, adjncy, vwgt, &wgtflag, &numflag,
                           &ncon, &ncommonnodes, &nparts, tpwgts, &ubvec,
                           options, &edgecut, part, &_Comm);

  MPI_Barrier(MPI_COMM_WORLD);
  if (_Rank == 0) printf("\t\t PartMeshKway-End...\n");

  int recvcounts[_Size];
  int displs[_Size];
  PetscInt *partInt;
  PetscMalloc((vtxdist[_Rank + 1] - vtxdist[_Rank]) * sizeof(PetscInt),
              &partInt);

  for (PetscInt i = 0; i < vtxdist[_Rank + 1] - vtxdist[_Rank]; i++)
    partInt[i] = part[i];

  for (PetscInt i = 0; i < _Size; i++) {
    recvcounts[i] = (int)(vtxdist[i + 1] - vtxdist[i]);
    displs[i] = (int)vtxdist[i];
  }
  PetscMalloc(_Nvglobal * sizeof(PetscInt), &_V2P);

  MPI_Barrier(MPI_COMM_WORLD);
  if (_Rank == 0) printf("\t\t Allgatherv...\n");
  MPI_Allgatherv(partInt, recvcounts[_Rank], MPI_LONG_LONG_INT, _V2P,
                 recvcounts, displs, MPI_LONG_LONG_INT, _Comm);

  _Nvlocal = 0;
  for (PetscInt i = 0; i < _Nvglobal; i++) {
    if (_Rank == _V2P[i]) {
      _Nvlocal++;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if (_Rank == 0) printf("\t\t PetscMalloc...\n");
  PetscMalloc(_Nvlocal * sizeof(PetscInt), &_GloInd);
  PetscMalloc(_Nvlocal * sizeof(PetscInt), &_ITot);
  PetscMalloc(_Nvlocal * sizeof(PetscInt *), &_AdjM);

  PetscMalloc(_Nvlocal * sizeof(PetscInt), &_NEByV);
  PetscMalloc(_Nvlocal * sizeof(PetscInt *), &_EByV);

  MPI_Barrier(MPI_COMM_WORLD);
  if (_Rank == 0) printf("\t\t Init AdjM, EByV...\n");

  // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  // Глобальный индекс узла

  int *locIdxByGlo;
  _NMaxAdj = _Mesh->_MaxVAdj;
  PetscMalloc(_Nvglobal * sizeof(int), &locIdxByGlo);
  PetscInt s = 0;
  for (PetscInt i = 0; i < _Nvglobal; i++) {
    if (_Rank == _V2P[i]) {
      _GloInd[s] = i;
      locIdxByGlo[i] = s;
      _ITot[s] = 0;
      _NEByV[s] = 0;
      s++;
    }
  }
  PetscInt gloIdx, locIdx, newAdjVtx;
  PetscBool alreadyinclude;
  PetscInt *adjncytmp;
  PetscMalloc(_Mesh->_MaxVAdj * _Nvlocal * sizeof(PetscInt), &adjncytmp);
  PetscInt *adjncyEl;
  PetscMalloc(_Mesh->_MaxEAdj * _Nvlocal * sizeof(PetscInt), &adjncyEl);
  for (PetscInt i = 0; i < _Mesh->_NElement; i++) {
    for (PetscInt j = 0; j < DIM + 1; j++) {
      gloIdx = _Mesh->_VertsElement[(DIM + 1) * i + j];
      if (_Rank == _V2P[gloIdx]) {
        locIdx = locIdxByGlo[gloIdx];
        for (PetscInt k = 0; k < DIM + 1; k++) {
          newAdjVtx = _Mesh->_VertsElement[(DIM + 1) * i + k];

          if (newAdjVtx != gloIdx) {
            // Если новой точки еще нет в массиве соседей, то включаем ее
            alreadyinclude = PETSC_FALSE;

            // Цикл по уже добавленным к узлу соседям
            for (PetscInt q = 0; q < _ITot[locIdx]; q++) {
              if (adjncytmp[_Mesh->_MaxVAdj * locIdx + q] == newAdjVtx) {
                alreadyinclude = PETSC_TRUE;
                break;
              }
            }
            if (alreadyinclude == PETSC_FALSE) {
              adjncytmp[_Mesh->_MaxVAdj * locIdx + _ITot[locIdx]] = newAdjVtx;
              _ITot[locIdx]++;
            }
          }
        }

        alreadyinclude = PETSC_FALSE;
        for (PetscInt q = 0; q < _NEByV[locIdx]; q++) {
          if (adjncyEl[_Mesh->_MaxEAdj * locIdx + q] == i) {
            alreadyinclude = PETSC_TRUE;
            break;
          }
        }
        if (alreadyinclude == PETSC_FALSE) {
          adjncyEl[_Mesh->_MaxEAdj * locIdx + _NEByV[locIdx]] = i;
          _NEByV[locIdx]++;
        }
      }
    }
  }
  PetscFree(locIdxByGlo);
  s = 0;

  _Nneighbors = 0;

  PetscInt k;
  for (PetscInt i = 0; i < _Nvglobal; i++) {
    if (_Rank == _V2P[i]) {
      PetscMalloc(_ITot[s] * sizeof(PetscInt), &(_AdjM[s]));
      for (k = 0; k < _ITot[s]; k++) {
        _AdjM[s][k] = adjncytmp[_Mesh->_MaxVAdj * s + k];
      }
      _Nneighbors += _ITot[s];

      PetscMalloc(_NEByV[s] * sizeof(PetscInt), &(_EByV[s]));
      for (PetscInt j = 0; j < _NEByV[s]; j++) {
        _EByV[s][j] = adjncyEl[_Mesh->_MaxEAdj * s + j];
      }
      s++;
    }
  }
  PetscFree(_V2P);
  PetscFree(adjncytmp);
  PetscFree(adjncyEl);

  int *adjncyEl1;
  PetscMalloc(_Mesh->_NElement * sizeof(int), &adjncyEl1);
  for (PetscInt i = 0; i < _Mesh->_NElement; i++) {
    adjncyEl1[i] = -1;
  }
  _Nelocal = 0;
  for (s = 0; s < _Nvlocal; s++) {
    for (PetscInt j = 0; j < _NEByV[s]; j++) {
      if (adjncyEl1[_EByV[s][j]] == -1) {
        adjncyEl1[_EByV[s][j]] = _Nelocal;
        _Nelocal++;
      }
    }
  }

  PetscMalloc(_Nelocal * sizeof(PetscInt), &_EGloInd);
  for (PetscInt i = 0; i < _Mesh->_NElement; i++) {
    if (adjncyEl1[i] != -1) {
      _EGloInd[adjncyEl1[i]] = i;
    }
  }
  PetscFree(adjncyEl1);

  // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  MPI_Barrier(MPI_COMM_WORLD);
  if (_Rank == 0) printf("\t\t Init EGloInd, VByE...\n");
  // Локальные элементы
  PetscMalloc(_Nelocal * sizeof(PetscInt *), &_VByE);
  for (PetscInt i = 0; i < _Nelocal; i++) {
    PetscMalloc((D1) * sizeof(PetscInt), &(_VByE[i]));
    _Mesh->GetVByE(_EGloInd[i], _VByE[i]);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (_Rank == 0) printf("\t\t Renumerate _EByV... ");
  // Локальная нумерация элементов
  for (s = 0; s < _Nvlocal; s++) {
    if (_Rank == 0 && (10 * s) % (_Nvlocal - 1) == 0)
      printf("%d%% ", (int)(100 * s / (_Nvlocal - 1)));

    for (PetscInt j = 0; j < _NEByV[s]; j++) {
      for (PetscInt i = 0; i < _Nelocal; i++) {
        if (_EByV[s][j] == _EGloInd[i]) {
          _EByV[s][j] = i;
          break;
        }
      }
    }
  }
  if (_Rank == 0) printf("\n");

  PetscFree(partInt);
}
PetscErrorCode MeshPartition::CreateBorderParameters() {
  PetscErrorCode ierr;

  ierr = PetscMalloc(_Nvlocal * sizeof(PetscInt), &_BorderNEdgesByV);
  CHKERRQ(ierr);
  ierr = PetscMalloc(_Nvlocal * sizeof(PetscInt *), &_BorderEdgesIDsByV);
  CHKERRQ(ierr);

  PetscInt NEdgesGlobal = _Mesh->_NEdge;
  _NEdgesLocal = 0;  // количество граней на текущем процессе

  PetscInt k;
  PetscInt vtx[DIM];
  for (PetscInt i = 0; i < NEdgesGlobal; i++) {
    _Mesh->GetVByEdge(i, vtx);

    for (PetscInt j = 0; j < _Nvlocal; j++) {
      for (k = 0; k < DIM; k++) {
        if (vtx[k] == _GloInd[j]) {
          _NEdgesLocal++;
          k = -1;
          break;
        }
      }
      if (k == -1) break;
    }
  }

  for (PetscInt i = 0; i < _Nvlocal; i++) {
    if (_Mesh->IsBorder(_GloInd[i]) == PETSC_TRUE) {
      _Mesh->GetEdgesByV(_GloInd[i], &(_BorderNEdgesByV[i]), NULL);
      ierr = PetscMalloc(_BorderNEdgesByV[i] * sizeof(PetscInt),
                         &(_BorderEdgesIDsByV[i]));
      CHKERRQ(ierr);
      _Mesh->GetEdgesByV(_GloInd[i], &(_BorderNEdgesByV[i]),
                         _BorderEdgesIDsByV[i]);
    } else {
      _BorderNEdgesByV[i] = 0;
      _BorderEdgesIDsByV[i] = NULL;
    }
  }

  ierr = PetscMalloc(_NEdgesLocal * sizeof(PetscInt), &(_GloEdgeIdx));
  CHKERRQ(ierr);
  ierr = PetscMalloc(_NEdgesLocal * sizeof(PetscInt), &(_BorderElement));
  CHKERRQ(ierr);
  ierr = PetscMalloc(_NEdgesLocal * sizeof(PetscInt *), &(_BorderVerts));
  CHKERRQ(ierr);
  ierr = PetscMalloc(_NEdgesLocal * sizeof(PetscReal *), &(_BorderN));
  CHKERRQ(ierr);
  ierr = PetscMalloc(_NEdgesLocal * sizeof(PetscReal), &(_BorderS));
  CHKERRQ(ierr);

  PetscInt s = 0;
  for (PetscInt i = 0; i < NEdgesGlobal; i++) {
    _Mesh->GetVByEdge(i, vtx);
    for (PetscInt j = 0; j < _Nvlocal; j++) {
      for (k = 0; k < DIM; k++) {
        if (vtx[k] == _GloInd[j]) {
          _GloEdgeIdx[s] = i;
          _BorderElement[s] = _Mesh->GetEByEdge(i);
          ierr = PetscMalloc(DIM * sizeof(PetscInt), &(_BorderVerts[s]));
          CHKERRQ(ierr);
          _Mesh->GetVByEdge(i, _BorderVerts[s]);

          ierr = PetscMalloc(DIM * sizeof(PetscReal), &(_BorderN[s]));
          CHKERRQ(ierr);
          _Mesh->GetNormalByEdge(i, &(_BorderN[s]));
          _Mesh->GetEdgeS(i, &(_BorderS[s]));

          s++;
          k = -1;
          break;
        }
      }
      if (k == -1) break;
    }
  }

  // Перевод параметров границы в локальную нумерацию
  for (PetscInt i = 0; i < _NEdgesLocal; i++) {
    for (PetscInt j = 0; j < DIM; j++) {
      for (PetscInt k = 0; k < _NVertices; k++) {
        if (_BorderVerts[i][j] == _GloInd[k]) {
          _BorderVerts[i][j] = k;
          break;
        }
      }
    }
    for (PetscInt j = 0; j < _Nelocal; j++) {
      if (_BorderElement[i] == _EGloInd[j]) {
        _BorderElement[i] = j;
        break;
      }
    }
  }
  for (PetscInt i = 0; i < _Nvlocal; i++) {
    if (_BorderEdgesIDsByV[i] != NULL) {
      for (PetscInt j = 0; j < _BorderNEdgesByV[i]; j++) {
        for (PetscInt k = 0; k < _NEdgesLocal; k++) {
          if (_BorderEdgesIDsByV[i][j] == _GloEdgeIdx[k]) {
            _BorderEdgesIDsByV[i][j] = k;
            break;
          }
        }
      }
    }
  }

  return 0;
}

PetscErrorCode MeshPartition::CheckNormals() {
  PetscReal a[2];
  PetscReal result;
  PetscInt v0, v1, ElementIdx;
  PetscBool b;
  // Цикл по граничным ребрам
  for (PetscInt e = 0; e < _NEdgesLocal; e++) {
    ElementIdx = _BorderElement[e];
    v0 = -1, v1 = -1;
    for (PetscInt i = 0; i < DIM + 1; i++) {
      b = PETSC_FALSE;
      for (PetscInt k = 0; k < DIM; k++) {
        if (_BorderVerts[e][k] == _VByE[ElementIdx][i]) {
          b = PETSC_TRUE;
        }
      }
      if (b == PETSC_FALSE && v1 == -1) {
        v1 = _GloInd[_VByE[ElementIdx][i]];
      } else if (b == PETSC_TRUE && v0 == -1) {
        v0 = _GloInd[_VByE[ElementIdx][i]];
      } else if (v1 != -1 && v0 != -1) {
        break;
      }
    }

    a[0] = _Mesh->_X1[v1] - _Mesh->_X1[v0];
    a[1] = _Mesh->_X2[v1] - _Mesh->_X2[v0];

    result = 0.0;
    for (PetscInt i = 0; i < DIM; i++) result += _BorderN[e][i] * a[i];
    if (result >= 0) {
      for (PetscInt i = 0; i < DIM; i++) _BorderN[e][i] = -_BorderN[e][i];
    }
  }
  return 0;
}

void MeshPartition::FreeParmetis() {
  delete[] part;
  delete[] vtxdist;
  delete[] xadj;
  delete[] adjncy;
  delete[] tpwgts;
}

PetscErrorCode MeshPartition::CreateLocalNumbering() {
  PetscInt *pordering;
  PetscInt rstart;
  PetscInt *tmp, *tmpE;
  PetscErrorCode ierr;
  PetscInt i, j;

  ierr = MPI_Scan(&_Nvlocal, &rstart, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD);
  CHKERRQ(ierr);
  rstart -= _Nvlocal;
  ierr = PetscMalloc(_Nvlocal * sizeof(PetscInt), &pordering);
  CHKERRQ(ierr);
  for (i = 0; i < _Nvlocal; i++) {
    pordering[i] = rstart + i;
  }

  /*
  Create the AO object
  */
  ierr = AOCreateBasic(MPI_COMM_WORLD, _Nvlocal, _GloInd, pordering, &_AO);
  CHKERRQ(ierr);
  ierr = PetscFree(pordering);
  CHKERRQ(ierr);

  /*
  Keep the global indices for later use
  */
  ierr = PetscMalloc(_Nvlocal * sizeof(PetscInt), &_LocInd);
  CHKERRQ(ierr);
  ierr = PetscMalloc(_Nneighbors * sizeof(PetscInt), &tmp);
  CHKERRQ(ierr);
  ierr = PetscMalloc((D1)*_Nelocal * sizeof(PetscInt), &tmpE);
  CHKERRQ(ierr);

  for (i = 0; i < _Nvlocal; i++) {
    _LocInd[i] = _GloInd[i];
  }
  rstart = 0;
  for (i = 0; i < _Nvlocal; i++) {
    for (j = 0; j < _ITot[i]; j++) {
      tmp[j + rstart] = _AdjM[i][j];
    }
    rstart += _ITot[i];
  }
  for (i = 0; i < _Nelocal; i++) {
    for (j = 0; j < D1; j++) {
      tmpE[(D1)*i + j] = _VByE[i][j];
    }
  }
  /*
  Map the vlocal and neighbor lists to the PETSc ordering
  */
  ierr = AOApplicationToPetsc(_AO, _Nvlocal, _LocInd);
  CHKERRQ(ierr);
  ierr = AOApplicationToPetsc(_AO, _Nneighbors, tmp);
  CHKERRQ(ierr);
  ierr = AOApplicationToPetsc(_AO, (D1)*_Nelocal, tmpE);
  CHKERRQ(ierr);

  rstart = 0;
  for (i = 0; i < _Nvlocal; i++) {
    for (j = 0; j < _ITot[i]; j++) {
      _AdjM[i][j] = tmp[j + rstart];
    }
    rstart += _ITot[i];
  }
  for (i = 0; i < _Nelocal; i++) {
    for (j = 0; j < (D1); j++) {
      _VByE[i][j] = tmpE[(D1)*i + j];
    }
  }
  ierr = PetscFree(tmp);
  CHKERRQ(ierr);
  ierr = PetscFree(tmpE);
  CHKERRQ(ierr);
  return (0);
}

PetscErrorCode MeshPartition::CreateGhostVertices() {
  PetscErrorCode ierr;
  PetscInt nb, i, j;

  PetscInt *verticesmask;

  ierr = PetscMalloc(_Nvglobal * sizeof(PetscInt), &_Vertices);
  CHKERRQ(ierr);
  ierr = PetscMalloc(_Nvglobal * sizeof(PetscInt), &verticesmask);
  CHKERRQ(ierr);
  ierr = PetscMemzero(verticesmask, _Nvglobal * sizeof(PetscInt));
  CHKERRQ(ierr);
  _NVertices = 0;
  /*
  First load "owned vertices" into list
  */
  for (i = 0; i < _Nvlocal; i++) {
    _Vertices[_NVertices++] = _LocInd[i];
    verticesmask[_LocInd[i]] = _NVertices;
  }

  /*
  Now load ghost vertices into list
  */
  for (i = 0; i < _Nvlocal; i++) {
    for (j = 0; j < _ITot[i]; j++) {
      nb = _AdjM[i][j];
      if (!verticesmask[nb]) {
        _Vertices[_NVertices++] = nb;
        verticesmask[nb] = _NVertices;
      }
    }
  }
  /*
   Map the vertices listed in the neighbors to the local numbering from
  the global ordering that they contained initially.
  */
  for (i = 0; i < _Nvlocal; i++) {
    for (j = 0; j < _ITot[i]; j++) {
      nb = _AdjM[i][j];
      _AdjM[i][j] = verticesmask[nb] - 1;
    }
  }

  for (i = 0; i < _Nelocal; i++) {
    for (j = 0; j < (D1); j++) {
      _VByE[i][j] = verticesmask[_VByE[i][j]] - 1;
    }
  }
  ierr = PetscFree(verticesmask);
  CHKERRQ(ierr);

  ierr = PetscFree(_GloInd);
  CHKERRQ(ierr);
  ierr = PetscMalloc(_NVertices * sizeof(PetscInt), &_GloInd);
  CHKERRQ(ierr);
  for (i = 0; i < _NVertices; i++) {
    _GloInd[i] = _Vertices[i];
  }
  ierr = AOPetscToApplication(_AO, _NVertices, _GloInd);
  CHKERRQ(ierr);

  // ===============================================================================
  // Расширение AdjM и ITot на фиктивные ячейки
  if (_Rank == 0) printf("\t\t AdjM, ITot extension...\n");

  PetscInt *bufITot;
  PetscInt **bufAdjM;
  PetscInt *bufNEByV;
  PetscInt **bufEByV;
  PetscMalloc((_Nvlocal) * sizeof(PetscInt), &bufITot);
  PetscMalloc((_Nvlocal) * sizeof(PetscInt *), &bufAdjM);
  PetscMalloc((_Nvlocal) * sizeof(PetscInt), &bufNEByV);
  PetscMalloc((_Nvlocal) * sizeof(PetscInt *), &bufEByV);
  for (i = 0; i < _Nvlocal; i++) {
    bufITot[i] = _ITot[i];
    PetscMalloc(bufITot[i] * sizeof(PetscInt), &(bufAdjM[i]));
    for (j = 0; j < _ITot[i]; j++) bufAdjM[i][j] = _AdjM[i][j];
    bufNEByV[i] = _NEByV[i];
    PetscMalloc(bufNEByV[i] * sizeof(PetscInt), &(bufEByV[i]));
    for (j = 0; j < _NEByV[i]; j++) bufEByV[i][j] = _EByV[i][j];
  }

  for (PetscInt i = 0; i < _Nvlocal; i++) {
    PetscFree(_AdjM[i]);
    PetscFree(_EByV[i]);
  }
  PetscFree(_ITot);
  PetscFree(_AdjM);
  PetscFree(_NEByV);
  PetscFree(_EByV);

  PetscMalloc((_NVertices) * sizeof(PetscInt), &_ITot);
  PetscMalloc((_NVertices) * sizeof(PetscInt *), &_AdjM);
  PetscMalloc((_NVertices) * sizeof(PetscInt), &_NEByV);
  PetscMalloc((_NVertices) * sizeof(PetscInt *), &_EByV);
  for (i = 0; i < _Nvlocal; i++) {
    _ITot[i] = bufITot[i];
    PetscMalloc(_ITot[i] * sizeof(PetscInt), &(_AdjM[i]));
    for (j = 0; j < _ITot[i]; j++) _AdjM[i][j] = bufAdjM[i][j];

    _NEByV[i] = bufNEByV[i];
    PetscMalloc(bufNEByV[i] * sizeof(PetscInt), &(_EByV[i]));
    for (j = 0; j < _NEByV[i]; j++) _EByV[i][j] = bufEByV[i][j];
  }
  for (PetscInt i = 0; i < _Nvlocal; i++) {
    PetscFree(bufAdjM[i]);
    PetscFree(bufEByV[i]);
  }
  PetscFree(bufITot);
  PetscFree(bufAdjM);
  PetscFree(bufNEByV);
  PetscFree(bufEByV);

  for (PetscInt k = _Nvlocal; k < _NVertices; k++) {
    _ITot[k] = 0;
    for (i = 0; i < _Nvlocal; i++)
      for (j = 0; j < _ITot[i]; j++)
        if (_AdjM[i][j] == k) _ITot[k]++;

    PetscMalloc(_ITot[k] * sizeof(PetscInt), &(_AdjM[k]));
    _ITot[k] = 0;
    for (i = 0; i < _Nvlocal; i++)
      for (j = 0; j < _ITot[i]; j++)
        if (_AdjM[i][j] == k) {
          _AdjM[k][_ITot[k]] = i;
          _ITot[k]++;
        }

    _NEByV[k] = 0;
    for (i = 0; i < _Nelocal; i++)
      for (j = 0; j < DIM + 1; j++)
        if (_VByE[i][j] == k) _NEByV[k]++;

    PetscMalloc(_NEByV[k] * sizeof(PetscInt), &(_EByV[k]));
    _NEByV[k] = 0;
    for (i = 0; i < _Nelocal; i++)
      for (j = 0; j < DIM + 1; j++)
        if (_VByE[i][j] == k) {
          _EByV[k][_NEByV[k]] = i;
          _NEByV[k]++;
        }
  }
  // ===============================================================================
  // ===============================================================================
  // Сортировка AdjM
  for (PetscInt k = 0; k < _NVertices; k++) {
    Sort(_ITot[k], &(_AdjM[k][0]));
  }
  return (0);
}
//Следующие три строки отвечают за сортировку
void MeshPartition::Sort(PetscInt Length, PetscInt *pArray) {
  for (int i = 0; i < Length; ++i) {
    for (int j = i + 1; j < Length; ++j) {
      if (pArray[j] < pArray[i]) swap(pArray[i], pArray[j]);
    }
  }
}
void MeshPartition::InitAuxVars(PetscReal *pX1, PetscReal *pX2) {
  PetscMalloc(_Nelocal * sizeof(PetscReal), &_cell_volume);
  PetscMalloc(_Nelocal * (D1)*DIM * sizeof(PetscReal), &_Hx);

  PetscInt vtx, e, d;
  for (e = 0; e < _Nelocal; e++) {
    _cell_volume[e] = Element::VolumeDim(pX1, pX2, DIM, _VByE[e]);

    for (vtx = 0; vtx < D1; vtx++)
      for (d = 0; d < DIM; d++)
        _Hx[(D1)*DIM * e + DIM * vtx + d] =
            Element::H_x(vtx, d + 1, pX1, pX2, _VByE[e]);
  }
}

void MeshPartition::FreeAuxVars() {
  PetscFree(_cell_volume);
  PetscFree(_Hx);
}
///////////////////////////////////////////////////////////////////////////////
// Визуализация
void MeshPartition::ShowInitialize() {
  PetscInt i, j;

  MPI_Barrier(_Comm);
  if (_Rank == 0)
    cout << "============================================\nInitialize\n";

  for (i = 0; i < _Size; i++) {
    if (_Rank == i) {
      cout << "\nRank " << _Rank << '\n';
      for (j = 0; j < _Size + 1; j++) {
        cout << "vtxdist[" << j << "]: " << vtxdist[j] << '\n';
      }
      for (j = 0; j < vtxdist[_Rank + 1] - vtxdist[_Rank] + 1; j++) {
        cout << "xadj[" << j << "]: " << xadj[j] << '\n';
      }
      for (j = 0; j < xadj[vtxdist[_Rank + 1] - vtxdist[_Rank]]; j++) {
        cout << "adjncy[" << j << "]: " << adjncy[j] << '\n';
      }
    }
    MPI_Barrier(_Comm);
  }
}
void MeshPartition::ShowResults(FILE *fptr) {
  PetscInt i, j;

  MPI_Barrier(_Comm);
  PetscFPrintf(PETSC_COMM_SELF, fptr,
               "==============================================================="
               "===============\n");
  PetscFPrintf(PETSC_COMM_SELF, fptr, "Номер процесса: %D\n", _Rank);
  PetscFPrintf(PETSC_COMM_SELF, fptr,
               "==============================================================="
               "===============\n");
  PetscFPrintf(PETSC_COMM_SELF, fptr, "  _Nvglobal = %D\n", _Nvglobal);
  PetscFPrintf(PETSC_COMM_SELF, fptr, "  _Nvlocal = %D\n", _Nvlocal);
  PetscFPrintf(PETSC_COMM_SELF, fptr, "  _NVertices = %D\n", _NVertices);
  PetscFPrintf(PETSC_COMM_SELF, fptr, "  _Nelocal = %D\n", _Nelocal);
  PetscFPrintf(PETSC_COMM_SELF, fptr, "  _NEdgesLocal = %D\n", _NEdgesLocal);
  PetscFPrintf(PETSC_COMM_SELF, fptr,
               "==============================================================="
               "===============\n");
  PetscFPrintf(PETSC_COMM_SELF, fptr, "Узлы\n");
  for (i = 0; i < _NVertices; i++) {
    if (i < _Nvlocal) {
      PetscFPrintf(PETSC_COMM_SELF, fptr, " Узел №%D\n", i);
      PetscFPrintf(PETSC_COMM_SELF, fptr, "  _LocInd[%D] = %D;  ", i,
                   _LocInd[i]);
      PetscFPrintf(PETSC_COMM_SELF, fptr, "  _GloInd[%D] = %D;  ", i,
                   _GloInd[i]);
      PetscFPrintf(PETSC_COMM_SELF, fptr, "  _Vertices[%D] = %D\n", i,
                   _Vertices[i]);

      PetscFPrintf(PETSC_COMM_SELF, fptr, "\n  Соседние узлы (%D): ", _ITot[i]);
      for (j = 0; j < _ITot[i]; j++) {
        PetscFPrintf(PETSC_COMM_SELF, fptr, "  %D", _AdjM[i][j]);
        if (j != _ITot[i] - 1) PetscFPrintf(PETSC_COMM_SELF, fptr, ",");
      }
      PetscFPrintf(PETSC_COMM_SELF, fptr,
                   "\n  Соседние элементы (%D): ", _NEByV[i]);
      for (j = 0; j < _NEByV[i]; j++) {
        PetscFPrintf(PETSC_COMM_SELF, fptr, "  %D", _EByV[i][j]);
        if (j != _NEByV[i] - 1) PetscFPrintf(PETSC_COMM_SELF, fptr, ",");
      }
      if (_BorderNEdgesByV[i] != 0) {
        PetscFPrintf(PETSC_COMM_SELF, fptr,
                     "\n  Узел принадлежит границе. Соседние ребра (%D): ",
                     _BorderNEdgesByV[i]);
        for (j = 0; j < _BorderNEdgesByV[i]; j++) {
          PetscFPrintf(PETSC_COMM_SELF, fptr, "  %D", _BorderEdgesIDsByV[i][j]);
          if (j != _BorderNEdgesByV[i] - 1)
            PetscFPrintf(PETSC_COMM_SELF, fptr, ",");
        }
      }
    } else {
      PetscFPrintf(PETSC_COMM_SELF, fptr, " Узел №%D (фиктивный)\n", i);
      PetscFPrintf(PETSC_COMM_SELF, fptr, "  _Vertices[%D] = %D  ", i,
                   _Vertices[i]);
      PetscFPrintf(PETSC_COMM_SELF, fptr, "  _GloInd[%D] = %D\n", i,
                   _GloInd[i]);
      PetscFPrintf(PETSC_COMM_SELF, fptr,
                   "\n  Соседние узлы на этом потоке (%D): ", _ITot[i]);
      for (j = 0; j < _ITot[i]; j++) {
        PetscFPrintf(PETSC_COMM_SELF, fptr, "  %D", _AdjM[i][j]);
        if (j != _ITot[i] - 1) PetscFPrintf(PETSC_COMM_SELF, fptr, ",");
      }
      PetscFPrintf(PETSC_COMM_SELF, fptr,
                   "\n  Соседние элементы на этом потоке (%D): ", _NEByV[i]);
      for (j = 0; j < _NEByV[i]; j++) {
        PetscFPrintf(PETSC_COMM_SELF, fptr, "  %D", _EByV[i][j]);
        if (j != _NEByV[i] - 1) PetscFPrintf(PETSC_COMM_SELF, fptr, ",");
      }
    }
    if (i != _NVertices - 1)
      PetscFPrintf(
          PETSC_COMM_SELF, fptr,
          "\n------------------------------------------------------------\n");
  }
  PetscFPrintf(PETSC_COMM_SELF, fptr,
               "\n\n==========================================================="
               "===================\n");
  PetscFPrintf(PETSC_COMM_SELF, fptr,
               "==============================================================="
               "===============\n");
  PetscFPrintf(PETSC_COMM_SELF, fptr, "Элементы\n");
  for (i = 0; i < _Nelocal; i++) {
    PetscFPrintf(PETSC_COMM_SELF, fptr, " Элемент №%D\n", i);
    PetscFPrintf(PETSC_COMM_SELF, fptr, "  _EGloInd[%D] = %D\n", i,
                 _EGloInd[i]);

    PetscFPrintf(PETSC_COMM_SELF, fptr, "  Соседние узлы (%d): ", D1);
    for (j = 0; j < D1; j++) {
      PetscFPrintf(PETSC_COMM_SELF, fptr, "  %D", _VByE[i][j]);
      if (j != DIM) PetscFPrintf(PETSC_COMM_SELF, fptr, ",");
    }
    if (i != _Nelocal - 1)
      PetscFPrintf(
          PETSC_COMM_SELF, fptr,
          "\n------------------------------------------------------------\n");
  }

  PetscFPrintf(PETSC_COMM_SELF, fptr,
               "\n\n==========================================================="
               "===================\n");
  PetscFPrintf(PETSC_COMM_SELF, fptr,
               "==============================================================="
               "===============\n");
  PetscFPrintf(PETSC_COMM_SELF, fptr, "Граничные ребра\n");
  for (i = 0; i < _NEdgesLocal; i++) {
    PetscFPrintf(PETSC_COMM_SELF, fptr, " Ребро №%D\n", i);
    PetscFPrintf(PETSC_COMM_SELF, fptr,
                 "  Глобальный индекс (_GloEdgeIdx[%D]): %D\n", i,
                 _GloEdgeIdx[i]);
    PetscFPrintf(PETSC_COMM_SELF, fptr, "  Нормаль (_BorderN[%D]): (%F, %F)\n",
                 i, _BorderN[i][0], _BorderN[i][1]);
    PetscFPrintf(PETSC_COMM_SELF, fptr, "  Длина (_GloEdgeIdx[%D]): %F\n", i,
                 _BorderS[i]);
    PetscFPrintf(PETSC_COMM_SELF, fptr,
                 "  Элемент составляющий ребро (_BorderElement[%D]): %D\n", i,
                 _BorderElement[i]);

    PetscFPrintf(PETSC_COMM_SELF, fptr, "  Соседние узлы (%d): ", DIM);
    for (j = 0; j < DIM; j++) {
      PetscFPrintf(PETSC_COMM_SELF, fptr, "  %D", _BorderVerts[i][j]);
      if (j != DIM - 1) PetscFPrintf(PETSC_COMM_SELF, fptr, ",");
    }
    if (i != _NEdgesLocal - 1)
      PetscFPrintf(
          PETSC_COMM_SELF, fptr,
          "\n------------------------------------------------------------\n");
  }
}
