#include "solver.h"

#include <sys/stat.h>
#include <sys/types.h>

#include "petscksp.h"

using namespace O3D;

Solver::Solver(Input *p_input, Monitor *p_monitor, IEquation *pEq,
               MeshPartition *pMP) {
  MPI_Comm_size(MPI_COMM_WORLD, &_Size);
  MPI_Comm_rank(MPI_COMM_WORLD, &_Rank);

  _User.Eq = pEq;
  _MP = pMP;
  _input = p_input;
  _monitor = p_monitor;

  _User.InitialConditionsY = PETSC_FALSE;
  _saving_regime = Usual;
}

Solver::~Solver() {
  VecScatterDestroy(&_User.Scatter);
  VecScatterDestroy(&_User.ScatterWithoutGhosts);

  for (PetscInt solver_idx = 0; solver_idx < _input->_solver_count;
       solver_idx++) {
    MatDestroy(&_Jac[solver_idx]);
    SNESDestroy(&_Snes[solver_idx]);
  }

  MatDestroy(&_User._mat_J_aux);
  if (_input->_problem_type == "TSA") {
    MatDestroy(&_User._matA);
    MatDestroy(&_User._matAH);
    MatDestroy(&_User._matB);
    MatDestroy(&_User._matBH);
  }

  DestroyMeshParameters();
  DestroyDataStructures();

  if (_input->_problem_type == "TSA") {
    DestroyDataStructuresAdjoint();
  }

  ISLocalToGlobalMappingDestroy(&_ISl2g);
}

PetscErrorCode Solver::Create() {
  MPI_Barrier(MPI_COMM_WORLD);
  _monitor->StartTimeMeasurement(CreateSolver);
  CreateMeshParameters();

  MPI_Barrier(MPI_COMM_WORLD);
  if (_Rank == 0) printf(" # Solver-CreateMeshParameters-End\n\n");

  MPI_Barrier(MPI_COMM_WORLD);
  if (_Rank == 0) {
    printf(" # Solver-CreateDataStructures-Start\n");
  }
  CreateDataStructures();
  if (_input->_problem_type == "TSA") CreateDataStructuresAdjoint();

  if (_input->_problem_type != "DNS" && _input->_base_flow_type != "None")
    _input->InitPeriodicBaseFlow(_User.lN, _User.GloInd);

  MPI_Barrier(MPI_COMM_WORLD);
  if (_Rank == 0) {
    printf(" # Solver-CreateDataStructures-End\n\n");
    printf(" # Solver-InitAuxVars-Start\n");
  }
  _User.CurFM = 0;
  _User.Eq->InitAuxVars(&_User);
  if (_Rank == 0) {
    printf(" # Solver-InitAuxVars-End\n\n");
    printf(" # Solver-SNESCreate-Start\n");
  }

  PetscErrorCode ierr;

  // Create nonlinear solver context
  for (PetscInt solver_idx = 0; solver_idx < _input->_solver_count;
       solver_idx++) {
    ierr = SNESCreate(MPI_COMM_WORLD, &_Snes[solver_idx]);
    CHKERRQ(ierr);
    ierr = SNESSetType(_Snes[solver_idx],
                       _input->_solver_snes_type[solver_idx].c_str());
    CHKERRQ(ierr);
  }

  // Set routines for function and Jacobian evaluation
  for (PetscInt solver_idx = 1; solver_idx < _input->_solver_count;
       solver_idx++)
    MatDuplicate(_Jac[0], MAT_DO_NOT_COPY_VALUES, &_Jac[solver_idx]);

  MatDuplicate(_Jac[0], MAT_DO_NOT_COPY_VALUES, &_User._mat_J_aux);
  if (_input->_problem_type == "TSA") {
    MatDuplicate(_Jac[0], MAT_DO_NOT_COPY_VALUES, &_User._matAH);
    MatDuplicate(_Jac[0], MAT_DO_NOT_COPY_VALUES, &_User._matBH);
    MatDuplicate(_Jac[0], MAT_DO_NOT_COPY_VALUES, &_User._matA);
    MatDuplicate(_Jac[0], MAT_DO_NOT_COPY_VALUES, &_User._matB);
  }
  ierr = SNESSetFunction(_Snes[0], _User.GlobalR, Function, (void *)&_User);
  CHKERRQ(ierr);
  ierr = JacobianNonZeros(_Snes[0], _User.GlobalX[0], _User._mat_J_aux,
                          _User._mat_J_aux, &_User);
  CHKERRQ(ierr);
  for (PetscInt solver_idx = 0; solver_idx < _input->_solver_count;
       solver_idx++) {
    if (solver_idx == 2) {
      ierr = SNESSetFunction(_Snes[solver_idx], _User.GlobalR, FunctionAdjoint,
                             (void *)&_User);
      CHKERRQ(ierr);

      _User.InitialConditionsY = PETSC_TRUE;
      ierr = JacobianNonZeros(_Snes[solver_idx], _User.GlobalX[0],
                              _Jac[solver_idx], _Jac[solver_idx], &_User);
      CHKERRQ(ierr);
      ierr = JacobianNonZeros(_Snes[solver_idx], _User.GlobalY[0], _User._matA,
                              _User._matA, &_User);
      CHKERRQ(ierr);
      ierr = JacobianNonZeros(_Snes[solver_idx], _User.GlobalY[0], _User._matB,
                              _User._matB, &_User);
      CHKERRQ(ierr);
      _User.InitialConditionsY = PETSC_FALSE;
    } else {
      ierr = SNESSetFunction(_Snes[solver_idx], _User.GlobalR, Function,
                             (void *)&_User);
      CHKERRQ(ierr);
      ierr = JacobianNonZeros(_Snes[solver_idx], _User.GlobalX[0],
                              _Jac[solver_idx], _Jac[solver_idx], &_User);
      CHKERRQ(ierr);
    }
  }

  // --------------------
  // Null space
  Vec vecP;
  MatNullSpace nullspace;

  ierr = VecDuplicate(_User.GlobalX[0], &vecP);
  CHKERRQ(ierr);
  ierr = VecSet(vecP, 0.0);
  CHKERRQ(ierr);

  PetscInt istart, iend;
  VecGetOwnershipRange(vecP, &istart, &iend);

  PetscScalar val = 1.0;
  for (PetscInt i = istart; i < iend; i++)
    if (i % NVAR == 0) VecSetValues(vecP, 1, &i, &val, INSERT_VALUES);

  ierr = VecAssemblyBegin(vecP);
  CHKERRQ(ierr);
  ierr = VecAssemblyEnd(vecP);
  CHKERRQ(ierr);

  VecNormalize(vecP, NULL);
  MatNullSpaceCreate(MPI_COMM_WORLD, PETSC_FALSE, 1, &vecP, &nullspace);

  VecDestroy(&vecP);
  for (PetscInt solver_idx = 0; solver_idx < _input->_solver_count;
       solver_idx++) {
    if (_input->_solver_snes_near_null_space[solver_idx] == PETSC_TRUE)
      MatSetNearNullSpace(_Jac[solver_idx], nullspace);
  }
  MatNullSpaceDestroy(&nullspace);

  // --------------------

  for (PetscInt solver_idx = 0; solver_idx < _input->_solver_count;
       solver_idx++) {
    if (solver_idx == 2)
      ierr = SNESSetJacobian(_Snes[solver_idx], _Jac[solver_idx],
                             _Jac[solver_idx], JacobianAdjoint, (void *)&_User);
    else
      ierr = SNESSetJacobian(_Snes[solver_idx], _Jac[solver_idx],
                             _Jac[solver_idx], Jacobian, (void *)&_User);
    CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Customize nonlinear solver; set runtime options
  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
  Set linear solver defaults for this problem. By extracting the
  KSP and PC contexts from the SNES context, we can then
  directly call any KSP and PC routines to set various options.
  */
  // KSP ksp; /* linear solver context */
  // PC pc; /* preconditioner context */

  // ierr = SNESGetKSP(_Snes,&ksp);CHKERRQ(ierr);
  // ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  // ierr = PCSetType(pc,PCGAMG);CHKERRQ(ierr);
  // PCFactorSetReuseOrdering(pc, PETSC_TRUE);

  /*KSP ksp;
  PC pc;
  const PetscInt ufields[] = {1,2,3},pfields[] = {0};
  ierr = SNESGetKSP(_Snes,&ksp); CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc); CHKERRQ(ierr);

  // -pc_type fieldsplit -pc_fieldsplit_type schur
  (-pc_fieldsplit_detect_saddle_point) ierr = PCSetType(pc,
  PCFIELDSPLIT);CHKERRQ(ierr); PCFieldSplitSetBlockSize(pc, NVAR);
  PCFieldSplitSetFields(pc, "p", 1, pfields, pfields);
  PCFieldSplitSetFields(pc, "u", 3, ufields, ufields);
  PCFieldSplitSetSchurPre(pc, PC_FIELDSPLIT_SCHUR_PRE_SELF,NULL);
  PCFieldSplitSetType(pc, PC_COMPOSITE_SCHUR);*/

  /*
  Set SNES/KSP/KSP/PC runtime options, e.g.,
  -snes_view -snes_monitor -ksp_type <ksp> -pc_type <pc>
  These options will override those specified above as long as
  SNESSetFromOptions() is called _after_ any other customization
  routines.
  */

  for (PetscInt solver_idx = 0; solver_idx < _input->_solver_count;
       solver_idx++) {
    ierr = SNESSetFromOptions(_Snes[solver_idx]);
    CHKERRQ(ierr);
  }

  // PetscReal atol=0.000000000001;
  // PetscReal rtol;
  // PetscReal stol;
  // PetscReal dtol;
  // PetscInt maxit;
  // PetscInt maxf;

  // Установка точности SNES
  // SNESSetTolerances(_Snes,atol,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);
  // SNESGetTolerances(_Snes, &atol, &rtol, &stol, &maxit, &maxf);

  // Установка точности KSP
  // maxit = 1000;
  // KSPSetTolerances(ksp, PETSC_DEFAULT, atol,PETSC_DEFAULT,maxit);
  // KSPGetTolerances(ksp, &rtol, &atol, &dtol, &maxit);

  // ========================================================================
  // Save mesh data

  SaveLocalGlobalIdx();

  // ========================================================================

  MPI_Barrier(MPI_COMM_WORLD);
  _monitor->EndTimeMeasurement(CreateSolver);
  return (0);
}

PetscErrorCode Solver::DestroyMeshParameters() {
  PetscErrorCode ierr;

  ierr = PetscFree(_User.MassFlux);
  CHKERRQ(ierr);

  ierr = PetscFree(_User.X1);
  CHKERRQ(ierr);
  ierr = PetscFree(_User.X2);
  CHKERRQ(ierr);

  ierr = PetscFree(_User.UniqueEdgeIndex);
  CHKERRQ(ierr);

  _User.Eq->DestroyAuxVars();
  return (0);
}

PetscErrorCode Solver::DestroyDataStructures() {
  PetscErrorCode ierr;
  VecDestroy(&_User.GlobalR);
  VecDestroy(&_User.GlobalAux);
  VecDestroy(&_User.GlobalAux1);
  for (PetscInt i = 0; i < _User.FM; i++) {
    ierr = VecDestroy(&(_User.GlobalX[i]));
    CHKERRQ(ierr);
  }

  ierr = PetscFree(_User.GlobalX);
  CHKERRQ(ierr);
  ierr = VecDestroy(&_User.LocalX);
  CHKERRQ(ierr);
  ierr = VecDestroy(&_User.LocalF);
  CHKERRQ(ierr);

  ierr = PetscFree(_auxLocalX_real);
  CHKERRQ(ierr);
  ierr = PetscFree(_auxLocalX_img);
  CHKERRQ(ierr);

  ierr = PetscFree(_User.lX);
  CHKERRQ(ierr);
  ierr = PetscFree(_User.lX0);
  CHKERRQ(ierr);
  ierr = PetscFree(_User.lX_t);
  CHKERRQ(ierr);
  ierr = PetscFree(_User.lX1_e);
  CHKERRQ(ierr);
  ierr = PetscFree(_User.lX00);
  CHKERRQ(ierr);

  PetscFree(_R);
  PetscFree(_RGlobal);
  return (0);
}

PetscErrorCode Solver::DestroyDataStructuresAdjoint() {
  PetscErrorCode ierr;

  VecDestroy(&_User.GlobalY0);

  for (PetscInt i = 0; i < _User.FM; i++) {
    ierr = VecDestroy(&(_User.GlobalY[i]));
    CHKERRQ(ierr);

    if (_input->_problem_type == "TSA") VecDestroy(&(_User.GlobalBY0[i]));
  }

  ierr = PetscFree(_User.GlobalY);
  CHKERRQ(ierr);

  if (_input->_problem_type == "TSA") PetscFree(_User.GlobalBY0);

  ierr = PetscFree(_User.G);
  CHKERRQ(ierr);
  ierr = PetscFree(_User.G0);
  CHKERRQ(ierr);
  ierr = PetscFree(_User.lY0);
  CHKERRQ(ierr);

  return (0);
}

PetscErrorCode Solver::CreateMeshParameters() {
  PetscErrorCode ierr;
  /////////////////////////////////////////////////////////////////////
  // AppCtx
  if (_Rank == 0) printf("\t AppCtx\n");
  _User.Nvglobal = _MP->_Nvglobal;
  _User.Nvlocal = _MP->_Nvlocal;
  _User.NvGhost = _MP->_NVertices - _MP->_Nvlocal;
  _User.Nneighbors = _MP->_Nneighbors;
  _User.GloInd = _MP->_GloInd;
  _User.LocInd = _MP->_LocInd;
  _User.ITot = _MP->_ITot;
  _User.AdjM = _MP->_AdjM;
  _User.Nelocal = _MP->_Nelocal;
  _User.NEByV = _MP->_NEByV;
  _User.EByV = _MP->_EByV;
  _User.VByE = _MP->_VByE;
  _User.NMaxAdj = _MP->_NMaxAdj;

  _User.NEdgesLocal = _MP->_NEdgesLocal;
  _User.GloEdgeIdx = _MP->_GloEdgeIdx;
  _User.BorderNEdgesByV = _MP->_BorderNEdgesByV;
  _User.BorderEdgesIDsByV = _MP->_BorderEdgesIDsByV;
  _User.BorderElement = _MP->_BorderElement;
  _User.EdgeByElement = _MP->_EdgeByElement;
  _User.BorderVerts = _MP->_BorderVerts;
  _User.BorderN = _MP->_BorderN;
  _User.BorderS = _MP->_BorderS;

  // =================================================================
  // Outpit: probes

  PetscInt glo_idx_min = 0;
  PetscReal dist_min, dist_cur;
  _input->_output_probe_count_local = 0;
  for (PetscInt probe_idx = 0; probe_idx < _input->_output_probe_count;
       probe_idx++) {
    for (PetscInt i = 0; i < _input->_Mesh._NVertex; i++) {
      dist_cur =
          (_input->_Mesh._X1[i] - _input->_output_probe_x[probe_idx]) *
              (_input->_Mesh._X1[i] - _input->_output_probe_x[probe_idx]) +
          (_input->_Mesh._X2[i] - _input->_output_probe_y[probe_idx]) *
              (_input->_Mesh._X2[i] - _input->_output_probe_y[probe_idx]);

      if (i == 0) {
        dist_min = dist_cur;
        glo_idx_min = i;
      } else if (dist_min > dist_cur) {
        dist_min = dist_cur;
        glo_idx_min = i;
      }
    }

    for (PetscInt i = 0; i < _User.Nvlocal; i++) {
      if (_MP->_GloInd[i] == glo_idx_min) {
        _input->_output_probe_idx_local[_input->_output_probe_count_local] = i;
        _input->_output_probe_x_local[_input->_output_probe_count_local] =
            _input->_Mesh._X1[glo_idx_min];
        _input->_output_probe_y_local[_input->_output_probe_count_local] =
            _input->_Mesh._X2[glo_idx_min];
        _input->_output_probe_count_local++;
      }
    }
  }

  ierr = PetscMalloc(_input->_boundaries_count * sizeof(PetscReal),
                     &(_User.MassFlux));
  CHKERRQ(ierr);
  for (PetscInt b = 0; b < _input->_boundaries_count; b++)
    _User.MassFlux[b] = 0;

  if (_Rank == 0)
    printf("\t Init X1, X2: %.2lfMB\n",
           2 * _MP->_NVertices * sizeof(PetscReal) / 1000000.0);
  ierr = PetscMalloc(_MP->_NVertices * sizeof(PetscReal), &(_User.X1));
  CHKERRQ(ierr);
  ierr = PetscMalloc(_MP->_NVertices * sizeof(PetscReal), &(_User.X2));
  CHKERRQ(ierr);

  for (PetscInt i = 0; i < _MP->_NVertices; i++) {
    _User.X1[i] = _input->_Mesh._X1[_MP->_GloInd[i]];
    _User.X2[i] = _input->_Mesh._X2[_MP->_GloInd[i]];
  }

  if (_Rank == 0) printf("\t InitAuxVars\n");
  MPI_Barrier(MPI_COMM_WORLD);

  _MP->InitAuxVars(_User.X1, _User.X2);

  _User.cell_volume = _MP->_cell_volume;
  _User.Hx = _MP->_Hx;

  // Вывод параметров разбиения в файл
  char fPartition[512];
  sprintf(fPartition, "%s/Partition.dat", _input->_Output);
  FILE *fptrPartition;
  for (PetscInt t = 0; t < _Size; t++) {
    if (_Rank == t) {
      if (_Rank == 0) {
        fptrPartition = fopen(fPartition, "w");
      } else {
        fptrPartition = fopen(fPartition, "a");
      }

      if (_Rank == 0) {
        // Заголовок файла решения
        ierr = PetscFPrintf(PETSC_COMM_SELF, fptrPartition,
                            "Nv (global):\t%d\n", _User.Nvglobal);
        CHKERRQ(ierr);
        ierr = PetscFPrintf(PETSC_COMM_SELF, fptrPartition,
                            "Proc\tNv (local)\tNv (ghost)\tMaxAdj\n");
        CHKERRQ(ierr);
      }
      ierr =
          PetscFPrintf(PETSC_COMM_SELF, fptrPartition, "%ld\t%ld\t%ld\t%ld\n",
                       _Rank, _User.Nvlocal, _User.NvGhost, _User.NMaxAdj);
      CHKERRQ(ierr);
      fclose(fptrPartition);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  // =================================================================
  // Определение массив уникальных граней (без повторений на разных процессах)
  // Массив с глобальными индексами

  PetscMalloc(_User.NEdgesLocal * sizeof(PetscInt), &_User.UniqueEdgeIndex);
  for (PetscInt i = 0; i < _User.NEdgesLocal; i++) {
    _User.UniqueEdgeIndex[i] = _User.GloEdgeIdx[i];
  }

  // Значение _User.NEdgesLocal на каждом процессе
  int *counts;
  int *displ;
  PetscMalloc(_Size * sizeof(int), &counts);
  PetscMalloc(_Size * sizeof(int), &displ);
  int val = _User.NEdgesLocal;
  MPI_Allgather(&val, 1, MPI_INT, counts, 1, MPI_INT, MPI_COMM_WORLD);

  PetscInt total = 0;
  for (PetscInt i = 0; i < _Size; i++) total += counts[i];
  displ[0] = 0;
  for (PetscInt i = 1; i < _Size; i++) displ[i] = displ[i - 1] + counts[i - 1];

  PetscInt
      *GloEdgeIndexes;  // массив с глобальными индексами на каждом процессе
  PetscMalloc(total * sizeof(PetscInt), &GloEdgeIndexes);
  MPI_Allgatherv(_User.UniqueEdgeIndex, _User.NEdgesLocal, MPI_LONG_LONG_INT,
                 GloEdgeIndexes, counts, displ, MPI_LONG_LONG_INT,
                 MPI_COMM_WORLD);

  for (PetscInt i = 0; i < _User.NEdgesLocal; i++) {
    for (PetscInt r = 0; r < _Rank; r++) {
      for (PetscInt k = 0; k < displ[r + 1]; k++) {
        if (GloEdgeIndexes[k] == _User.UniqueEdgeIndex[i])
          _User.UniqueEdgeIndex[i] = -1;
      }
    }
  }

  PetscFree(counts);
  PetscFree(displ);
  PetscFree(GloEdgeIndexes);

  MPI_Barrier(MPI_COMM_WORLD);
  _input->_Mesh.Finalize();

  // =================================================================
  // Fourier modes
  _User.FM = _input->_FModes;
  _User.NZ = 2 * _User.FM - 1;

  if (_Rank == 0) {
    printf("\t Fourier modes: %ld (independent) / %ld (total)\n", _User.FM,
           _User.NZ);
  }

  return (0);
}

PetscErrorCode Solver::CreateDataStructures() {
  PetscErrorCode ierr;
  IS isglobal, islocal; /* global and local index sets */

  PetscMalloc(_User.FM * sizeof(Vec), &_User.GlobalX);

  _User.BackwardsInTime = PETSC_FALSE;

  MPI_Barrier(MPI_COMM_WORLD);
  if (_Rank == 0) printf("\tCreate vector data structures...\n");
  ierr = VecCreate(MPI_COMM_WORLD, &(_User.GlobalX[0]));
  CHKERRQ(ierr);
  ierr = VecSetSizes(_User.GlobalX[0], NVAR * _MP->_Nvlocal,
                     NVAR * _MP->_Nvglobal);
  CHKERRQ(ierr);
  ierr = VecSetBlockSize(_User.GlobalX[0], NVAR);
  CHKERRQ(ierr);
  ierr = VecSetFromOptions(_User.GlobalX[0]);
  CHKERRQ(ierr);

  for (PetscInt i = 1; i < _User.FM; i++) {
    ierr = VecDuplicate(_User.GlobalX[0], &(_User.GlobalX[i]));
    CHKERRQ(ierr);
  }

  ierr = VecDuplicate(_User.GlobalX[0], &_User.GlobalAux);
  CHKERRQ(ierr);
  ierr = VecDuplicate(_User.GlobalX[0], &_User.GlobalAux1);
  CHKERRQ(ierr);
  ierr = VecDuplicate(_User.GlobalX[0], &_User.GlobalR);
  CHKERRQ(ierr);
  ierr = VecCreateSeq(MPI_COMM_SELF, NVAR * _MP->_NVertices, &_User.LocalX);
  CHKERRQ(ierr);
  ierr = VecDuplicate(_User.LocalX, &_User.LocalF);
  CHKERRQ(ierr);

  _User.lN = NVAR * _MP->_NVertices;
  PetscMalloc(_User.FM * _User.lN * sizeof(PetscScalar), &_User.lX);
  PetscMalloc(_User.FM * _User.lN * sizeof(PetscScalar), &_User.lX0);
  PetscMalloc(_User.FM * _User.lN * sizeof(PetscScalar), &_User.lX_t);
  PetscMalloc(_User.FM * _User.lN * sizeof(PetscScalar), &_User.lX1_e);

  PetscMalloc(_User.FM * _User.lN * sizeof(PetscScalar), &_User.lX00);

  PetscMalloc(NVAR * _User.Nvlocal * sizeof(PetscReal), &_auxLocalX_real);
  PetscMalloc(NVAR * _User.Nvlocal * sizeof(PetscReal), &_auxLocalX_img);

  PetscMalloc(2 * DIM * _input->_output_force_count * sizeof(PetscReal), &_R);
  PetscMalloc(2 * DIM * _input->_output_force_count * sizeof(PetscReal),
              &_RGlobal);

  MPI_Barrier(MPI_COMM_WORLD);
  if (_Rank == 0)
    printf(
        "\tCreate the scatter between the global representation and the local "
        "representation...\n");

  ierr = ISCreateStride(MPI_COMM_SELF, NVAR * _MP->_NVertices, 0, 1, &islocal);
  CHKERRQ(ierr);
  ierr = ISCreateBlock(MPI_COMM_SELF, NVAR, _MP->_NVertices, _MP->_Vertices,
                       PETSC_COPY_VALUES, &isglobal);
  CHKERRQ(ierr);
  ierr = VecScatterCreate(_User.GlobalX[0], isglobal, _User.LocalX, islocal,
                          &_User.Scatter);
  CHKERRQ(ierr);

  ierr = ISDestroy(&isglobal);
  CHKERRQ(ierr);
  ierr = ISDestroy(&islocal);
  CHKERRQ(ierr);

  ierr = ISCreateStride(MPI_COMM_SELF, NVAR * _User.Nvlocal, 0, 1, &islocal);
  CHKERRQ(ierr);
  ierr = ISCreateBlock(MPI_COMM_SELF, NVAR, _User.Nvlocal, _MP->_Vertices,
                       PETSC_COPY_VALUES, &isglobal);
  CHKERRQ(ierr);
  ierr = VecScatterCreate(_User.GlobalX[0], isglobal, _User.LocalX, islocal,
                          &_User.ScatterWithoutGhosts);
  CHKERRQ(ierr);

  ierr = ISDestroy(&isglobal);
  CHKERRQ(ierr);
  ierr = ISDestroy(&islocal);
  CHKERRQ(ierr);

  MPI_Barrier(MPI_COMM_WORLD);
  if (_Rank == 0) {
    printf("\tCreate matrix data structure...\n");
  }

  ierr = MatCreate(MPI_COMM_WORLD, &_Jac[0]);
  CHKERRQ(ierr);
  ierr = MatSetSizes(_Jac[0], NVAR * _MP->_Nvlocal, NVAR * _MP->_Nvlocal,
                     NVAR * _MP->_Nvglobal, NVAR * _MP->_Nvglobal);
  CHKERRQ(ierr);
  ierr = MatSetBlockSize(_Jac[0], NVAR);
  CHKERRQ(ierr);
  ierr = MatSetFromOptions(_Jac[0]);
  CHKERRQ(ierr);

  MPI_Barrier(MPI_COMM_WORLD);
  if (_Rank == 0) {
    printf("\tMatrix preallocation (seq)...\n");
  }

  PetscInt maxAdj = 0;
  PetscInt curAdj;
  PetscInt maxGhostAdj = 0;
  PetscInt curGhostAdj;

  PetscInt *d_nnz;  // Число ненулевых столбцов в строке "на диагонали"
  PetscInt *o_nnz;  // Число ненулевых столбцов в строке "вне диагонали"
  ierr = PetscMalloc(NVAR * _User.Nvlocal * sizeof(PetscInt), &d_nnz);
  CHKERRQ(ierr);
  ierr = PetscMalloc(NVAR * _User.Nvlocal * sizeof(PetscInt), &o_nnz);
  CHKERRQ(ierr);
  for (PetscInt i = 0; i < _User.Nvlocal; i++) {
    curGhostAdj = 0;
    curAdj = 0;
    for (PetscInt j = 0; j < _User.ITot[i]; j++) {
      if (_User.AdjM[i][j] >= _User.Nvlocal) {
        curGhostAdj++;
      } else if (_User.AdjM[i][j] < _User.Nvlocal) {
        curAdj++;
      }
    }
    if (maxGhostAdj < curGhostAdj) maxGhostAdj = curGhostAdj;
    if (maxAdj < curAdj) maxAdj = curAdj;

    for (PetscInt j = 0; j < NVAR; j++) {
      d_nnz[NVAR * i + j] = NVAR * (curAdj + 1);
      o_nnz[NVAR * i + j] = NVAR * curGhostAdj;
    }
  }
  MatSeqAIJSetPreallocation(_Jac[0], 0, d_nnz);

  MPI_Barrier(MPI_COMM_WORLD);
  if (_Rank == 0) printf("\tMatrix preallocation (mpi)...\n");
  MatMPIAIJSetPreallocation(_Jac[0], 0, d_nnz, 0, o_nnz);

  ierr = PetscFree(d_nnz);
  CHKERRQ(ierr);
  ierr = PetscFree(o_nnz);
  CHKERRQ(ierr);

  MatSetOption(_Jac[0], MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);

  // The following routine allows us to set the matrix values in local ordering
  MPI_Barrier(MPI_COMM_WORLD);
  if (_Rank == 0) printf("\tMatrix ordering...\n");
  ierr =
      ISLocalToGlobalMappingCreate(MPI_COMM_SELF, NVAR, _MP->_NVertices,
                                   _MP->_Vertices, PETSC_COPY_VALUES, &_ISl2g);
  CHKERRQ(ierr);
  ierr = MatSetLocalToGlobalMapping(_Jac[0], _ISl2g, _ISl2g);
  CHKERRQ(ierr);

  return (0);
}

PetscErrorCode Solver::CreateDataStructuresAdjoint() {
  PetscErrorCode ierr;

  PetscMalloc(_User.FM * sizeof(Vec), &_User.GlobalY);
  PetscMalloc(_User.FM * sizeof(Vec), &_User.GlobalBY0);

  MPI_Barrier(MPI_COMM_WORLD);
  if (_Rank == 0) printf("\tCreate vector data structures...\n");
  ierr = VecCreate(MPI_COMM_WORLD, &(_User.GlobalY[0]));
  CHKERRQ(ierr);
  ierr = VecSetSizes(_User.GlobalY[0], NVAR * _MP->_Nvlocal,
                     NVAR * _MP->_Nvglobal);
  CHKERRQ(ierr);
  ierr = VecSetBlockSize(_User.GlobalY[0], NVAR);
  CHKERRQ(ierr);
  ierr = VecSetFromOptions(_User.GlobalY[0]);
  CHKERRQ(ierr);

  for (PetscInt i = 1; i < _User.FM; i++) {
    ierr = VecDuplicate(_User.GlobalY[0], &(_User.GlobalY[i]));
    CHKERRQ(ierr);
  }

  if (_input->_problem_type == "TSA") {
    for (PetscInt i = 0; i < _User.FM; i++)
      VecDuplicate(_User.GlobalY[0], &(_User.GlobalBY0[i]));
  }

  ierr = VecDuplicate(_User.GlobalY[0], &_User.GlobalY0);
  CHKERRQ(ierr);

  _User.lN = NVAR * _MP->_NVertices;
  PetscMalloc(_User.FM * _User.lN * sizeof(PetscScalar), &_User.lY0);

  PetscMalloc(_User.FM * _User.lN * sizeof(PetscScalar), &_User.G);
  PetscMalloc(_User.FM * _User.lN * sizeof(PetscScalar), &_User.G0);

  return (0);
}

PetscErrorCode Solver::Solve() {
  if (_input->_problem_type == "TSA")
    Solve_OptimalPerurbations();
  else if (_input->_problem_type == "Floquet")
    Solve_Floquet();
  else
    Solve_GeneralProblem();

  return (0);
}

void Solver::SolveSystem(PetscReal p_time, ProblemType p_problem,
                         SolverType p_solver) {
  _monitor->StartTimeMeasurement(SNESSolver);

  PetscInt snes_iterations, failed_iterations;
  if (p_solver == ReadFromFile && p_problem == PrimalBaseFlow) {
    _User.Eq->UpdateBaseFlow(&_User, p_time);
  } else if (p_problem == AdjointPerturbations) {
    SNESSolve(_Snes[p_problem], NULL, _User.GlobalY[_User.CurFM]);
    SNESGetIterationNumber(_Snes[p_problem], &snes_iterations);
    SNESGetNonlinearStepFailures(_Snes[p_problem], &failed_iterations);

    _monitor->UpdateSNESIterations(_User.CurFM, snes_iterations,
                                   failed_iterations);
  } else {
    SNESSolve(_Snes[p_problem], NULL, _User.GlobalX[_User.CurFM]);
    SNESGetIterationNumber(_Snes[p_problem], &snes_iterations);
    SNESGetNonlinearStepFailures(_Snes[p_problem], &failed_iterations);

    _monitor->UpdateSNESIterations(_User.CurFM, snes_iterations,
                                   failed_iterations);
  }

  _monitor->EndTimeMeasurement(SNESSolver);
}

PetscErrorCode Solver::Solve_GeneralProblem() {
  if (_Rank == 0) {
    printf(
        ")\n*************************************************************\n");
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // Compute weights W for norm calculations
  InitializeNormWeights();
  SaveNormWeights();

  // ========================================================================
  // Fourier decomposition of matrices
  if (_input->_problem_type != "DNS" &&
      _input->_jacobi_matrix_interpolation == PETSC_TRUE) {
    InitialConditions_Perturbations();
    _User.Eq->InitializeJacobianInterpolation(&_User);
  }

  // Initial conditions
  InitialConditions_BaseFlow();
  InitialConditions_Perturbations();

  MPI_Barrier(MPI_COMM_WORLD);

  Solve_PrimalProblem();

  _monitor->PrintStatistics();
  _monitor->SaveStatistics();
  // =========================================
  // Finalize
  PetscFree(_norm_weigths);
  PetscFree(_norm_weigths_sqrt);

  if (_input->_problem_type != "DNS" &&
      _input->_jacobi_matrix_interpolation == PETSC_TRUE) {
    _User.Eq->FinalizeJacobianInterpolation();
  }

  if (_input->_problem_type != "DNS") _input->FinalizePeriodicBaseFlow();

  return (0);
}

PetscErrorCode Solver::Solve_Floquet() {
  PetscReal normE0;

  if (_Rank == 0) {
    printf(
        "\n********************************************************************"
        "************\n");
  }

  // ========================================================================
  // Initialize
  // Residual
  PetscReal *residual;
  PetscMalloc(_User.FM * sizeof(PetscReal), &residual);

  // Initial and final energy
  PetscReal *energy_0;
  PetscReal *energy_T;

  PetscMalloc(_User.FM * sizeof(PetscReal), &energy_0);
  PetscMalloc(_User.FM * sizeof(PetscReal), &energy_T);

  // Eigensolver
  _eigenproblem = new Eigenproblem[_User.FM - 1];
  for (PetscInt mode = 1; mode < _User.FM; mode++)
    _eigenproblem[mode - 1].Initialize(_input->_solver_eigenproblem_KS_size,
                                       NVAR * _User.Nvlocal, PETSC_TRUE);

  PetscComplex *debug_mat;

  if (_input->_debug_test == 3) {
    PetscMalloc(NVAR * _User.Nvlocal * sizeof(PetscComplex), &debug_mat);

    for (int i = 0; i < NVAR * _User.Nvlocal; i++)
      debug_mat[i] =
          2.0 * (2.0 * ((PetscReal)rand()) / ((PetscReal)RAND_MAX) - 1.0) +
          2.0 * PETSC_i *
              (2.0 * ((PetscReal)rand()) / ((PetscReal)RAND_MAX) - 1.0);

    char fileName[512];
    sprintf(fileName, "%s/test3_random_eigenvalues.txt", _input->_Output);
    FILE *fTest;
    fTest = fopen(fileName, "a");
    if (!fTest) {
      printf("Could not open input file: %s\n", fileName);
    } else {
      for (int i = 0; i < NVAR * _User.Nvlocal; i++)
        fprintf(fTest, "%.20lf\t%.20lf\n", PetscRealPart(debug_mat[i]),
                PetscImaginaryPart(debug_mat[i]));
      fclose(fTest);
    }
  }
  // =========================================================================
  // Compute weights W for norm calculations
  InitializeNormWeights();
  SaveNormWeights();

  // ========================================================================
  // Fourier decomposition of matrices
  if (_input->_jacobi_matrix_interpolation == PETSC_TRUE) {
    InitialConditions_Perturbations();
    _User.Eq->InitializeJacobianInterpolation(&_User);
  }

  // ========================================================================
  // Initial conditions
  // Base flow
  _User.T = _input->_T_start;
  _User.Eq->UpdateBaseFlow(&_User, _User.T);

  // Perturbations
  InitialConditions_Perturbations();

  // Real part is zero

  if (_input->_debug_test == 3) {
    // DEBUG: random initial conditions for the test
    for (int i = 0; i < NVAR * _User.Nvlocal; i++)
      _User.lX0[_User.lN * 1 + i] =
          (2.0 * ((PetscReal)rand()) / ((PetscReal)RAND_MAX) - 1.0) +
          PETSC_i * (2.0 * ((PetscReal)rand()) / ((PetscReal)RAND_MAX) - 1.0);
  }

  // ========================================================================
  // Main cylce
  // --------------------------------------------------------------------
  char fileName[512];
  sprintf(fileName, "%s/Monitor.txt", _input->_Output);
  FILE *fMonitor;

  // =====================================================
  // Several iterations of power method
  for (PetscInt cycle = 0;
       cycle < _input->_solver_eigenproblem_power_method_its; cycle++) {
    if (_Rank == 0)
      printf("\n========> POWER METHOD ITERATION: %ld/%ld\n", cycle + 1,
             _input->_solver_eigenproblem_power_method_its);

    // =========================================================================
    // Forward time loop (primal problem)

    // ---------------------------------
    // Solve primal problem
    if (_input->_debug_test == 3) {
      // DEBUG: multiply matrix on vector
      for (int i = 0; i < NVAR * _User.Nvlocal; i++)
        _User.lX0[_User.lN * 1 + i] =
            debug_mat[i] * _User.lX0[_User.lN * 1 + i];
    } else
      Solve_PrimalProblem();

    // =========================================================================
    // New initial conditions for perturbations
    for (PetscInt mode = 1; mode < _User.FM; mode++)
      UpdateGlobalVector(&(_User.lX0[_User.lN * mode]), &(_User.GlobalX[mode]));
  }

  // =========================================================================
  // Orthogonalize X0
  for (PetscInt mode = 1; mode < _User.FM; mode++)
    _eigenproblem[mode - 1].ArnoldiUpdate(0, PETSC_TRUE, PETSC_FALSE,
                                          _norm_weigths_sqrt,
                                          &(_User.lX0[_User.lN * mode]));
  for (PetscInt mode = 1; mode < _User.FM; mode++)
    UpdateGlobalVector(&(_User.lX0[_User.lN * mode]), &(_User.GlobalX[mode]));

  if (_input->_debug_tsa_KS_iterations_save == PETSC_TRUE)
    _saving_regime = KSIterations;
  else
    _saving_regime = Off;

  _monitor->Reset();

  // =====================================================
  // Construct Krylov subspace: b, Ab, A^2b,...
  // A^{_solver_eigenproblem_KS_size}b
  for (PetscInt cycle = 0; cycle < _input->_solver_eigenproblem_KS_size;
       cycle++) {
    if (_Rank == 0)
      printf("\n========> KRYLOV SUBSPACE CYCLE ITERATION: %ld/%ld\n",
             cycle + 1, _input->_solver_eigenproblem_KS_size);

    // =========================================================================
    // Forward time loop (primal problem)

    // ---------------------------------
    // Solve primal problem
    for (PetscInt mode = 1; mode < _User.FM; mode++)
      energy_0[mode] = GetNormEnergy(&(_User.lX0[_User.lN * mode]));

    /*if(_input->_debug_jacobi_save == PETSC_TRUE)
    {
        char fX[512];
        printf("XX_in\n");
        sprintf(fX,"%s/XX%g.m",_input->_Output, _User.T);
        PetscViewer viewerX;

        PetscViewerCreate(PETSC_COMM_WORLD,&viewerX);
        PetscViewerSetType(viewerX,PETSCVIEWERASCII);
        PetscViewerBinarySetSkipHeader(viewerX,PETSC_TRUE);
        PetscViewerBinarySetSkipOptions(viewerX,PETSC_TRUE);
        PetscViewerBinarySkipInfo(viewerX);
        PetscViewerFileSetMode(viewerX,FILE_MODE_WRITE);
        PetscViewerFileSetName(viewerX,fX);

        PetscViewerPushFormat(viewerX,PETSC_VIEWER_ASCII_MATLAB);
        PetscObjectSetName((PetscObject)_User.GlobalX[0],"X");
        VecView(_User.GlobalX[1],viewerX);
        PetscViewerPopFormat(viewerX);
        PetscViewerDestroy(&viewerX);

        if(_Rank == 0)
            PressEnterToContinue();
    }*/

    if (_input->_debug_test == 3) {
      // DEBUG: multiply matrix on vector
      for (int i = 0; i < NVAR * _User.Nvlocal; i++)
        _User.lX0[_User.lN * 1 + i] =
            debug_mat[i] * _User.lX0[_User.lN * 1 + i];
    } else
      Solve_PrimalProblem();

    /*if(_input->_debug_jacobi_save == PETSC_TRUE)
    {
        char fX[512];
        printf("XX_out\n");
        sprintf(fX,"%s/XX%g.m",_input->_Output, _User.T);
        PetscViewer viewerX;

        PetscViewerCreate(PETSC_COMM_WORLD,&viewerX);
        PetscViewerSetType(viewerX,PETSCVIEWERASCII);
        PetscViewerBinarySetSkipHeader(viewerX,PETSC_TRUE);
        PetscViewerBinarySetSkipOptions(viewerX,PETSC_TRUE);
        PetscViewerBinarySkipInfo(viewerX);
        PetscViewerFileSetMode(viewerX,FILE_MODE_WRITE);
        PetscViewerFileSetName(viewerX,fX);

        PetscViewerPushFormat(viewerX,PETSC_VIEWER_ASCII_MATLAB);
        PetscObjectSetName((PetscObject)_User.GlobalX[0],"X");
        VecView(_User.GlobalX[1],viewerX);
        PetscViewerPopFormat(viewerX);
        PetscViewerDestroy(&viewerX);

        if(_Rank == 0)
            PressEnterToContinue();
    }*/

    for (PetscInt mode = 1; mode < _User.FM; mode++)
      energy_T[mode] = GetNormEnergy(&(_User.lX0[_User.lN * mode]));

    if (_Rank == 0)
      for (PetscInt mode = 1; mode < _User.FM; mode++)
        printf("\nE_T[%ld]/E_0=%.20lf\tE_0=%.20lf\n\n", mode,
               energy_T[mode] / energy_0[mode], energy_0[mode]);

    // =========================================================================
    // New initial conditions for perturbations

    // Orthogonalize X0
    for (PetscInt mode = 1; mode < _User.FM; mode++)
      _eigenproblem[mode - 1].ArnoldiUpdate(cycle + 1, PETSC_TRUE, PETSC_FALSE,
                                            _norm_weigths_sqrt,
                                            &(_User.lX0[_User.lN * mode]));

    for (PetscInt mode = 1; mode < _User.FM; mode++)
      UpdateGlobalVector(&(_User.lX0[_User.lN * mode]), &(_User.GlobalX[mode]));
  }

  // =========================================
  // Compute Eigenvalues and Eigenvectors
  for (PetscInt mode = 1; mode < _User.FM; mode++) {
    _eigenproblem[mode - 1].Solve();
    _eigenproblem[mode - 1].PrintEigenValues();
    _eigenproblem[mode - 1].SaveEigenValues(_input->_Output);
  }

  // ---------------------------------------------------------
  // Check the accuracy

  _saving_regime = OptimalPertubations;
  if (_Rank == 0) {
    printf(
        "======================================================================"
        "===\n");
    printf("Check the accuracy\n");
  }

  PetscScalar *X_initial;
  PetscMalloc(_User.FM * NVAR * _MP->_NVertices * sizeof(PetscScalar),
              &X_initial);

  int start_idx = _input->_solver_eigenproblem_KS_size -
                  _input->_output_eigenvalues_monitor;
  for (_eigenvalue_index = max(0, start_idx);
       _eigenvalue_index < _input->_solver_eigenproblem_KS_size;
       _eigenvalue_index++) {
    if (true) {
      if (_Rank == 0)
        printf("\n *** run #: %ld/%ld ***\n\n", _eigenvalue_index + 1,
               _input->_solver_eigenproblem_KS_size);

      // =========================================================================
      // Forward time loop (primal problem)

      // =========================================================================
      // Set initial condition
      // Base flow
      _User.T = _input->_T_start;
      _User.Eq->UpdateBaseFlow(&_User, _User.T);

      // Perturbations
      for (PetscInt mode = 1; mode < _User.FM; mode++) {
        _eigenproblem[mode - 1].GetEigenVector(_eigenvalue_index, PETSC_FALSE,
                                               _norm_weigths_sqrt,
                                               &(_User.lX0[_User.lN * mode]));
        UpdateGlobalVector(&(_User.lX0[_User.lN * mode]),
                           &(_User.GlobalX[mode]));
      }

      // Normalize
      /*for(PetscInt mode=1; mode<_User.FM; mode++)
      {
          normE0 = sqrt(GetNormEnergy(&(_User.lX0[_User.lN*mode])));
          for(PetscInt i=0; i<_User.lN; i++)
              _User.lX0[_User.lN*mode+i] = _User.lX0[_User.lN*mode+i]/normE0;

          UpdateGlobalVector(&(_User.lX0[_User.lN*mode]),
      &(_User.GlobalX[mode]));
      }*/

      // =========================================================================
      // Store initial guess
      for (PetscInt mode = 1; mode < _User.FM; mode++)
        for (int i = 0; i < _User.lN; i++)
          X_initial[_User.lN * mode + i] = _User.lX0[_User.lN * mode + i];

      // ---------------------------------
      // Solve primal problem
      for (PetscInt mode = 1; mode < _User.FM; mode++)
        energy_0[mode] = GetNormEnergy(&(_User.lX0[_User.lN * mode]));

      if (_input->_debug_test == 3) {
        // DEBUG: multiply matrix on vector
        for (int i = 0; i < NVAR * _User.Nvlocal; i++)
          _User.lX0[_User.lN * 1 + i] =
              debug_mat[i] * _User.lX0[_User.lN * 1 + i];
      } else
        Solve_PrimalProblem();

      for (PetscInt mode = 1; mode < _User.FM; mode++)
        energy_T[mode] = GetNormEnergy(&(_User.lX0[_User.lN * mode]));

      if (_Rank == 0)
        for (PetscInt mode = 1; mode < _User.FM; mode++)
          printf("\nE_T[%ld]/E_0=%.20lf\tE_0=%.20lf\n\n", mode,
                 energy_T[mode] / energy_0[mode], energy_0[mode]);

      // =========================================================================
      // Calculate residual: ||X^{N}-X^{0}||_2/||X^{0}||_2
      PetscReal norm_X_initial;
      PetscReal norm_X_delta;
      for (PetscInt mode = 1; mode < _User.FM; mode++) {
        norm_X_initial = GetVNorm2(&(X_initial[_User.lN * mode]));
        for (PetscInt nodeIdx = 0; nodeIdx < _User.Nvlocal; nodeIdx++) {
          for (PetscInt var = 1; var < NVAR; var++)
            X_initial[_User.lN * mode + NVAR * nodeIdx + var] =
                _User.lX0[_User.lN * mode + NVAR * nodeIdx + var] -
                _eigenproblem[mode - 1]._eigenvalue[_eigenvalue_index] *
                    X_initial[_User.lN * mode + NVAR * nodeIdx + var];
        }
        norm_X_delta = GetVNorm2(&(X_initial[_User.lN * mode]));
        residual[mode] = norm_X_delta / norm_X_initial;
      }

      // Print and save residual etc.
      if (_Rank == 0) {
        fMonitor = fopen(fileName, "a");
        if (!fMonitor) {
          printf("Could not open input file: %s\n", fileName);
        } else {
          for (PetscInt mode = 1; mode < _User.FM; mode++)
            fprintf(
                fMonitor, "%.20lf\t%.20lf\t%.20lf\t%.20lf\t%.20lf\t\t",
                PetscRealPart(
                    _eigenproblem[mode - 1]._eigenvalue[_eigenvalue_index]),
                PetscImaginaryPart(
                    _eigenproblem[mode - 1]._eigenvalue[_eigenvalue_index]),
                PetscRealPart(
                    _eigenproblem[mode - 1]._eigenvalue[_eigenvalue_index]) *
                        PetscRealPart(_eigenproblem[mode - 1]
                                          ._eigenvalue[_eigenvalue_index]) +
                    PetscImaginaryPart(_eigenproblem[mode - 1]
                                           ._eigenvalue[_eigenvalue_index]) *
                        PetscImaginaryPart(_eigenproblem[mode - 1]
                                               ._eigenvalue[_eigenvalue_index]),
                energy_T[mode] / energy_0[mode], residual[mode]);

          fprintf(fMonitor, "\n");
          fclose(fMonitor);
        }

        printf("\n***\nRESIDUALS:\n");
        for (PetscInt mode = 1; mode < _User.FM; mode++)
          printf(
              "mode-%ld: L=%.20lf+i%.20lf\t |L|^2=%.20lf\t E_T/E_0=%.20lf\t "
              "res=%.20lf (||X_0||=%.20lf)\n",
              mode,
              PetscRealPart(
                  _eigenproblem[mode - 1]._eigenvalue[_eigenvalue_index]),
              PetscImaginaryPart(
                  _eigenproblem[mode - 1]._eigenvalue[_eigenvalue_index]),
              PetscRealPart(
                  _eigenproblem[mode - 1]._eigenvalue[_eigenvalue_index]) *
                      PetscRealPart(_eigenproblem[mode - 1]
                                        ._eigenvalue[_eigenvalue_index]) +
                  PetscImaginaryPart(
                      _eigenproblem[mode - 1]._eigenvalue[_eigenvalue_index]) *
                      PetscImaginaryPart(_eigenproblem[mode - 1]
                                             ._eigenvalue[_eigenvalue_index]),
              energy_T[mode] / energy_0[mode], residual[mode], norm_X_initial);
        printf("***\n\n");
      }
    }
  }
  if (_Rank == 0)
    printf(
        "======================================================================"
        "===\n");

  // =========================================
  // Finalize
  PetscFree(residual);
  PetscFree(energy_0);
  PetscFree(energy_T);
  PetscFree(_norm_weigths);
  PetscFree(_norm_weigths_sqrt);
  PetscFree(X_initial);

  if (_input->_jacobi_matrix_interpolation == PETSC_TRUE) {
    _User.Eq->FinalizeJacobianInterpolation();
  }

  _input->FinalizePeriodicBaseFlow();

  // ========================================================================
  // Finalize eigensolver
  for (PetscInt mode = 1; mode < _User.FM; mode++)
    _eigenproblem[mode - 1].Finalize();
  delete[] _eigenproblem;

  if (_input->_debug_test == 3) PetscFree(debug_mat);
  // ========================================================================
  // Print results
  _monitor->PrintStatistics();
  _monitor->SaveStatistics();

  return (0);
}

PetscErrorCode Solver::Solve_OptimalPerurbations() {
  PetscReal normE0;

  if (_Rank == 0) {
    printf(
        "\n********************************************************************"
        "************\n");
  }

  // ========================================================================
  // Initialize
  // Residual
  PetscReal *residual;
  PetscMalloc(_User.FM * sizeof(PetscReal), &residual);

  // Initial and final energy
  PetscReal *energy_0;
  PetscReal *energy_T;

  PetscMalloc(_User.FM * sizeof(PetscReal), &energy_0);
  PetscMalloc(_User.FM * sizeof(PetscReal), &energy_T);

  for (PetscInt mode = 1; mode < _User.FM; mode++)
    VecZeroEntries(_User.GlobalY[mode]);

  // Eigensolver
  _eigenproblem = new Eigenproblem[_User.FM - 1];
  for (PetscInt mode = 1; mode < _User.FM; mode++)
    _eigenproblem[mode - 1].Initialize(_input->_solver_eigenproblem_KS_size,
                                       NVAR * _User.Nvlocal, PETSC_FALSE);

  // =========================================================================
  // Compute weights W for norm calculations
  InitializeNormWeights();
  SaveNormWeights();

  // ========================================================================
  // Fourier decomposition of matrices
  if (_input->_jacobi_matrix_interpolation == PETSC_TRUE) {
    InitialConditions_Perturbations();
    _User.Eq->InitializeJacobianInterpolation(&_User);
  }

  // ========================================================================
  // Initial conditions
  // Base flow
  // if(_Rank==0)
  //    printf(" -> Initial conditions (primal problem)\n");
  _User.T = _input->_T_start;
  _User.Eq->UpdateBaseFlow(&_User, _User.T);

  // Perturbations
  InitialConditions_Perturbations();

  // ========================================================================
  // Main cylce
  // --------------------------------------------------------------------
  char fileName[512];
  sprintf(fileName, "%s/Monitor.txt", _input->_Output);
  FILE *fMonitor;

  // =========================================================================
  // Orthogonalize X0

  //_eigenproblem[0].SaveMatrixMATLAB(_input->_Output, "DEBUG_X0_mode_1",
  //_User.lN, 1, &(_User.lX0[_User.lN*1]));
  //_eigenproblem[0].SaveMatrixMATLAB(_input->_Output, "weights_sqrt", _User.lN,
  // 1, _norm_weigths_sqrt);

  for (PetscInt mode = 1; mode < _User.FM; mode++)
    _eigenproblem[mode - 1].ArnoldiUpdate(0, PETSC_FALSE, PETSC_TRUE,
                                          _norm_weigths_sqrt,
                                          &(_User.lX0[_User.lN * mode]));

  for (PetscInt mode = 1; mode < _User.FM; mode++)
    UpdateGlobalVector(&(_User.lX0[_User.lN * mode]), &(_User.GlobalX[mode]));

  //_eigenproblem[0].SaveMatrixMATLAB(_input->_Output, "DEBUG_X1_mode_1",
  //_User.lN, 1, &(_User.lX0[_User.lN*1]));

  if (_input->_debug_tsa_KS_iterations_save == PETSC_TRUE)
    _saving_regime = KSIterations;
  else
    _saving_regime = Off;

  _monitor->Reset();
  // Construct Krylov subspace: b, Ab, A^2b,...
  // A^{_solver_eigenproblem_KS_size}b
  for (PetscInt cycle = 0; cycle < _input->_solver_eigenproblem_KS_size;
       cycle++) {
    //_eigenproblem[0].SaveMatrixMATLAB(_input->_Output,
    //("X0_mode1_"+to_string(cycle)).c_str(), _User.lN, 1,
    //&(_User.lX0[_User.lN*1]));

    if (_Rank == 0)
      printf("\n========> KRYLOV SUBSPACE CYCLE ITERATION: %ld/%ld\n",
             cycle + 1, _input->_solver_eigenproblem_KS_size);

    // =========================================================================
    // Forward time loop (primal problem)

    Solve_PrimalAdjointCycle(energy_0, energy_T);

    // =========================================================================
    // New initial conditions for perturbations
    // X_0=0.5(E_0^2/|E_T)W^{-1}A^HY_0
    // R = A^H Y_0
    // if(_Rank==0)
    //    printf(" -> Initial conditions (primal problem)\n");
    _User.T = _input->_T_start;
    _User.Eq->UpdateBaseFlow(&_User, _User.T);

    //_eigenproblem[0].SaveMatrixMATLAB(_input->_Output,
    //("X0_mode1-0_"+to_string(cycle)).c_str(), _User.lN, 1,
    //&(_User.lX0[_User.lN*1]));

    InitialConditions_Perturbations_FromAdjointSolution(energy_0, energy_T);

    //_eigenproblem[0].SaveMatrixMATLAB(_input->_Output,
    //("X0_mode1-1_"+to_string(cycle)).c_str(), _User.lN, 1,
    //&(_User.lX0[_User.lN*1]));
    // Orthogonalize X0
    for (PetscInt mode = 1; mode < _User.FM; mode++)
      _eigenproblem[mode - 1].ArnoldiUpdate(cycle + 1, PETSC_FALSE, PETSC_TRUE,
                                            _norm_weigths_sqrt,
                                            &(_User.lX0[_User.lN * mode]));

    for (PetscInt mode = 1; mode < _User.FM; mode++)
      UpdateGlobalVector(&(_User.lX0[_User.lN * mode]), &(_User.GlobalX[mode]));
  }

  // =========================================
  // Compute Eigenvalues and Eigenvectors
  for (PetscInt mode = 1; mode < _User.FM; mode++) {
    _eigenproblem[mode - 1].Solve();
    _eigenproblem[mode - 1].PrintEigenValues();
    _eigenproblem[mode - 1].SaveEigenValues(_input->_Output);

    //_eigenproblem[mode-1].SaveMatrixHQMATLAB(_input->_Output, mode);
  }

  // ---------------------------------------------------------
  // Check the accuracy
  _saving_regime = OptimalPertubations;
  if (_Rank == 0) {
    printf(
        "======================================================================"
        "===\n");
    printf("Check the accuracy\n");
  }
  PetscScalar *X_initial;
  PetscMalloc(_User.FM * NVAR * _MP->_NVertices * sizeof(PetscScalar),
              &X_initial);

  // PetscBool stable_perturbations;
  int start_idx = _input->_solver_eigenproblem_KS_size -
                  _input->_output_eigenvalues_monitor;
  for (_eigenvalue_index = max(0, start_idx);
       _eigenvalue_index < _input->_solver_eigenproblem_KS_size;
       _eigenvalue_index++) {
    if (true)  //(stable_perturbations == PETSC_TRUE || _eigenvalue_index >=
               //_input->_solver_eigenproblem_KS_size-_input->_output_eigenvalues_monitor)
    {
      if (_Rank == 0)
        printf("\n *** run #: %ld/%ld ***\n\n", _eigenvalue_index + 1,
               _input->_solver_eigenproblem_KS_size);

      // =========================================================================
      // Forward time loop (primal problem)

      // =========================================================================
      // Set initial condition
      // Base flow
      _User.T = _input->_T_start;
      _User.Eq->UpdateBaseFlow(&_User, _User.T);

      // Perturbations
      for (PetscInt mode = 1; mode < _User.FM; mode++) {
        _eigenproblem[mode - 1].GetEigenVector(_eigenvalue_index, PETSC_TRUE,
                                               _norm_weigths_sqrt,
                                               &(_User.lX0[_User.lN * mode]));
        UpdateGlobalVector(&(_User.lX0[_User.lN * mode]),
                           &(_User.GlobalX[mode]));
      }

      // Normalize
      for (PetscInt mode = 1; mode < _User.FM; mode++) {
        normE0 = sqrt(GetNormEnergy(&(_User.lX0[_User.lN * mode])));
        for (PetscInt i = 0; i < _User.lN; i++)
          _User.lX0[_User.lN * mode + i] =
              _User.lX0[_User.lN * mode + i] / normE0;

        UpdateGlobalVector(&(_User.lX0[_User.lN * mode]),
                           &(_User.GlobalX[mode]));
      }

      // =========================================================================
      // Store initial guess
      for (PetscInt mode = 1; mode < _User.FM; mode++)
        for (int i = 0; i < _User.lN; i++)
          X_initial[_User.lN * mode + i] = _User.lX0[_User.lN * mode + i];

      Solve_PrimalAdjointCycle(energy_0, energy_T);

      // =========================================================================
      // New initial conditions for perturbations
      // X_0=0.5(E_0^2/|E_T)W^{-1}A^HY_0
      // R = A^H Y_0
      InitialConditions_Perturbations_FromAdjointSolution(energy_0, energy_T);
      for (PetscInt mode = 1; mode < _User.FM; mode++)
        UpdateGlobalVector(&(_User.lX0[_User.lN * mode]),
                           &(_User.GlobalX[mode]));

      // =========================================================================
      // Calculate residual: max|X_0^{n+1}-X_0^{n}|
      PetscReal locResidual;
      PetscReal auxResidual;
      for (PetscInt mode = 1; mode < _User.FM; mode++) {
        locResidual = 0.0;
        for (PetscInt nodeIdx = 0; nodeIdx < _User.Nvlocal; nodeIdx++) {
          for (PetscInt var = 0; var < 3; var++) {
            auxResidual =
                fabs(energy_0[mode] *
                         _User.lX0[_User.lN * mode + NVAR * nodeIdx + var + 1] -
                     energy_T[mode] *
                         X_initial[_User.lN * mode + NVAR * nodeIdx + var + 1]);
            if (auxResidual > locResidual) locResidual = auxResidual;
          }
        }
        MPI_Allreduce(&locResidual, &(residual[mode]), 1, MPI_DOUBLE, MPI_MAX,
                      MPI_COMM_WORLD);
      }

      // Print and save residual etc.
      if (_Rank == 0) {
        fMonitor = fopen(fileName, "a");
        if (!fMonitor) {
          printf("Could not open input file: %s\n", fileName);
        } else {
          for (PetscInt mode = 1; mode < _User.FM; mode++)
            fprintf(fMonitor, "%.16lf\t%.16lf\t%.16lf\t%.16lf\t\t",
                    PetscRealPart(
                        _eigenproblem[mode - 1]._eigenvalue[_eigenvalue_index]),
                    PetscImaginaryPart(
                        _eigenproblem[mode - 1]._eigenvalue[_eigenvalue_index]),
                    energy_T[mode], residual[mode]);

          fprintf(fMonitor, "\n");
          fclose(fMonitor);
        }

        printf("\n***\nRESIDUALS:\n");
        for (PetscInt mode = 1; mode < _User.FM; mode++)
          printf("mode-%ld: L=%.20lf+i%.20lf\t E_T/E_0=%.20lf\t res=%.20lf\n",
                 mode,
                 PetscRealPart(
                     _eigenproblem[mode - 1]._eigenvalue[_eigenvalue_index]),
                 PetscImaginaryPart(
                     _eigenproblem[mode - 1]._eigenvalue[_eigenvalue_index]),
                 energy_T[mode] / energy_0[mode], residual[mode]);
        printf("***\n\n");
      }
    }
  }
  if (_Rank == 0)
    printf(
        "======================================================================"
        "===\n");

  // =========================================
  // Finalize
  PetscFree(residual);
  PetscFree(energy_0);
  PetscFree(energy_T);
  PetscFree(_norm_weigths);
  PetscFree(_norm_weigths_sqrt);
  PetscFree(X_initial);

  if (_input->_jacobi_matrix_interpolation == PETSC_TRUE) {
    _User.Eq->FinalizeJacobianInterpolation();
  }

  _input->FinalizePeriodicBaseFlow();

  // ========================================================================
  // Finalize eigensolver
  for (PetscInt mode = 1; mode < _User.FM; mode++)
    _eigenproblem[mode - 1].Finalize();
  delete[] _eigenproblem;

  // ========================================================================
  // Print results
  _monitor->PrintStatistics();
  _monitor->SaveStatistics();

  return (0);
}

void Solver::Solve_PrimalAdjointCycle(PetscReal *r_energy_0,
                                      PetscReal *r_energy_T) {
  for (PetscInt mode = 1; mode < _User.FM; mode++)
    r_energy_0[mode] = GetNormEnergy(&(_User.lX0[_User.lN * mode]));

  // ---------------------------------
  // Solve primal problem
  Solve_PrimalProblem();

  for (PetscInt mode = 1; mode < _User.FM; mode++)
    r_energy_T[mode] = GetNormEnergy(&(_User.lX0[_User.lN * mode]));

  if (_Rank == 0)
    for (PetscInt mode = 1; mode < _User.FM; mode++)
      printf("\nE_T[%ld]/E_0=%.20lf\tE_0=%.20lf\n\n", mode,
             r_energy_T[mode] / r_energy_0[mode], r_energy_0[mode]);

  // =========================================================================
  // Save amplified solution

  _User.T = _input->_T_start + _input->_dT * _input->_NT;
  _User.Eq->UpdateBaseFlow(&_User, _User.T);

  // =========================================================================
  // =========================================================================
  // Backward time loop (adjoint problem)

  // Initial conditions for perturbations (AY_T - 2 WX_T / E_0 = 0)
  // if(_Rank==0)
  //    printf(" -> Initial conditions (adjoint problem)\n");
  for (PetscInt mode = 1; mode < _User.FM; mode++) {
    _User.CurFM = mode;

    // Calculate R (GlobalAux) = - 2 WX_T / E_0
    for (PetscInt i = 0; i < NVAR * _User.Nvlocal; i++)
      _User.lY0[_User.lN * mode + i] = -2.0 * _norm_weigths[i] *
                                       _User.lX0[_User.lN * mode + i] /
                                       r_energy_0[mode];
    UpdateGlobalVector(&(_User.lY0[_User.lN * mode]), &(_User.GlobalAux));

    // Calculate F = AX-BY and A = dF/dX
    if (_input->_NT == 1) _User.FirstTimeStep = PETSC_TRUE;

    if (_input->_jacobi_matrix_interpolation == false || _input->_NT == 1)
      _User.Eq->Function(_Snes[2], _User.GlobalY[_User.CurFM], _User.GlobalR,
                         &_User);
    _User.FreezeJac = PETSC_FALSE;
    _User.Eq->Jacobian(_Snes[2], _User.GlobalY[_User.CurFM], _User._matA,
                       _User._matA, &_User);

    // A^H
    MatDestroy(&_User._matAH);
    MatHermitianTranspose(_User._matA, MAT_INITIAL_MATRIX, &_User._matAH);
    if (_input->_NT == 1) _User.FirstTimeStep = PETSC_FALSE;

    // Solve A^HY_T + R = 0
    _User.InitialConditionsY = PETSC_TRUE;
    SolveSystem(_User.T, AdjointPerturbations, PETSc);
    _User.InitialConditionsY = PETSC_FALSE;
  }

  Solve_AdjointProblem();
}

void Solver::Solve_PrimalProblem() {
  _monitor->StartTimeMeasurement(TotalSolver);

  _User.FirstTimeStep = PETSC_TRUE;
  for (PetscInt time_iteration = 0; time_iteration < _input->_NT;
       time_iteration++) {
    // --------------------------------------------------------------------
    // New step update
    _User.T =
        _input->_T_start + _input->_dT * time_iteration;  // Update current time
    StepX0();  // Shift solution to new layer
    /*if(_Rank==0)
    {
        printf("DEBUG\n");
        char fileName[512];
        sprintf(fileName,"%s/test2.txt",_input->_Output);
        FILE* fMonitor = fopen(fileName, "a");
        if (!fMonitor) { printf("Could not open input file: %s\n", fileName); }
        else
        {
            fprintf(fMonitor, "%.16lf\t%.16lf\n", _User.T,
    _User.lX0[_User.lN*1+1]); fclose(fMonitor);
        }
    }*/
    // -------------------------------------------------------------------------
    // Output
    _monitor->PrintMonitor(time_iteration, _User.T);  // Print iteration

    Save(time_iteration);

    // ----------------------------------------------------------------------
    // Update nonlinear term (coupling)
    if (_input->_problem_type == "DNS") _User.Eq->NonlinearTermFFT(&_User);

    // --------------------------------------------------------------------
    // Compute 2D 'base' flow
    _User.CurFM = 0;  // Set current mode
    _User.FreezeJac = PETSC_FALSE;

    _User.T += _input->_dT;
    if (_input->_base_flow_type == "None")
      SolveSystem(_User.T, PrimalBaseFlow, PETSc);
    else
      _User.Eq->UpdateBaseFlow(&_User, _User.T);

    // --------------------------------------------------------------------
    // Compute 3D 'perturbations'
    for (PetscInt mode = 1; mode < _User.FM; mode++) {
      _User.CurFM = mode;  // Set current mode
      _User.FreezeJac = PETSC_FALSE;
      SolveSystem(_User.T, PrimalPerturbations, PETSc);
    }
    _User.T -= _input->_dT;
    _User.FirstTimeStep = PETSC_FALSE;
  }

  _User.T =
      _input->_T_start + _input->_dT * _input->_NT;  // Update current time
  StepX0();                                      // Shift solution to new layer
  _monitor->PrintMonitor(_input->_NT, _User.T);  // Print iteration

  // -------------------------------------------------------------------------
  // Save results
  Save(_input->_NT);

  _monitor->EndTimeMeasurement(TotalSolver);
}

void Solver::Solve_AdjointProblem() {
  _monitor->StartTimeMeasurement(TotalSolver);

  for (PetscInt time_iteration = _input->_NT; time_iteration > 0;
       time_iteration--) {
    // --------------------------------------------------------------------
    // New step update
    _User.T = _input->_T_start +
              _input->_dT * (time_iteration - 1);  // Update current time
    StepY0();  // Shift solution to new layer

    // --------------------------------------------------------------------
    // Output
    _monitor->PrintMonitor(_input->_NT - time_iteration,
                           _User.T + _input->_dT);  // Print iteration
    // !!! DIFF
    // Save(time_iteration);

    // --------------------------------------------------------------------
    // Small perturbations (solution of linearised N.-S. equations)

    for (PetscInt mode = 1; mode < _User.FM; mode++) {
      // Set current mode
      _User.CurFM = mode;

      if (_input->_time_scheme_acceleration[1] == BDF2) {
        if (time_iteration != _input->_NT)
          MatMult(_User._matBH, _User.GlobalY0, _User.GlobalBY0[mode]);
        else
          VecSet(_User.GlobalBY0[mode], 0.0);
      }

      // Save global initial value
      UpdateGlobalVector(&(_User.lY0[_User.lN * mode]), &(_User.GlobalY0));

      // --------------------------------------------------------------------
      // Calculate (A^H)^(n-1)
      if (time_iteration != 1) {
        if (time_iteration == 2) _User.FirstTimeStep = PETSC_TRUE;

        if (_input->_jacobi_matrix_interpolation == false ||
            time_iteration == 2) {
          _User.Eq->UpdateBaseFlow(&_User, _User.T);

          _User.FreezeJac = PETSC_FALSE;
          _User.Eq->Function(_Snes[2], _User.GlobalY[_User.CurFM],
                             _User.GlobalR, &_User);
        }
        _User.Eq->Jacobian(_Snes[2], _User.GlobalY[_User.CurFM], _User._matA,
                           _User._matA, &_User);

        MatDestroy(&_User._matAH);
        MatHermitianTranspose(_User._matA, MAT_INITIAL_MATRIX, &_User._matAH);

        if (time_iteration == 2) _User.FirstTimeStep = PETSC_FALSE;
      }
      // sprintf(pVecName, "AH_%ld_%ld", cycle, time_iteration);
      // SaveMatrix(pVecName, PETSC_VIEWER_ASCII_MATLAB, _User._matAH);
      // --------------------------------------------------------------------
      // Calculate (-B^H)^(n)
      if (time_iteration == 1) _User.FirstTimeStep = PETSC_TRUE;

      _User.T += _input->_dT;  // !!! temp
      if (_input->_jacobi_matrix_interpolation == false ||
          time_iteration == 1) {
        _User.Eq->UpdateBaseFlow(&_User, _User.T);
        _User.Eq->Function(_Snes[2], _User.GlobalY[_User.CurFM], _User.GlobalR,
                           &_User);
      }
      _User.BackwardsInTime = PETSC_TRUE;
      _User.FreezeJac = PETSC_FALSE;
      _User.Eq->Jacobian(_Snes[2], _User.GlobalY[_User.CurFM], _User._matB,
                         _User._matB, &_User);
      _User.BackwardsInTime = PETSC_FALSE;

      MatDestroy(&_User._matBH);
      MatHermitianTranspose(_User._matB, MAT_INITIAL_MATRIX, &_User._matBH);

      if (time_iteration == 1) _User.FirstTimeStep = PETSC_FALSE;

      // Run Adjoint Solver
      if (time_iteration != 1)
        SolveSystem(_User.T, AdjointPerturbations, PETSc);
      else {
        // GlobalY^0 = (A^H)^0*Y^0=-(B^H)^1*Y^1-(C^H)^2*Y^2
        MatMult(_User._matBH, _User.GlobalY0, _User.GlobalY[_User.CurFM]);
        if (_input->_time_scheme_acceleration[1] == BDF2)
          VecAXPY(_User.GlobalY[_User.CurFM], -0.25,
                  _User.GlobalBY0[_User.CurFM]);  // C = -0.25B
        VecScale(_User.GlobalY[_User.CurFM], -1.0);
      }
      _User.T -= _input->_dT;  // !!! temp
    }
  }

  _User.T = _input->_T_start;  // Update current time
  StepY0();                    // Shift solution to new layer
  _monitor->PrintMonitor(_input->_NT, _User.T);
  //_eigenproblem[0].SaveMatrixMATLAB(_input->_Output, ("Y0_adjoint_end_mode1"),
  //_User.lN, 1, &(_User.lY0[_User.lN*1]));/**/

  _monitor->EndTimeMeasurement(TotalSolver);
}

void Solver::InitializeNormWeights() {
  PetscReal eps = 1e-10;
  PetscReal node_volume;
  PetscMalloc(NVAR * _User.Nvlocal * sizeof(PetscReal), &_norm_weigths);
  PetscMalloc(NVAR * _User.Nvlocal * sizeof(PetscReal), &_norm_weigths_sqrt);
  for (PetscInt node_idx = 0; node_idx < _User.Nvlocal; node_idx++) {
    node_volume = 0.0;
    for (PetscInt e = 0; e < _User.NEByV[node_idx]; e++)
      node_volume += _User.cell_volume[_User.EByV[node_idx][e]];

    node_volume = node_volume / 3.0;

    _norm_weigths[NVAR * node_idx + 0] = eps;
    for (PetscInt var = 1; var < NVAR; var++)
      _norm_weigths[NVAR * node_idx + var] = node_volume;
  }

  for (PetscInt i = 0; i < NVAR * _User.Nvlocal; i++)
    _norm_weigths_sqrt[i] = sqrt(_norm_weigths[i]);
}
PetscReal Solver::GetNormEnergy(PetscScalar *pX) {
  PetscReal integralLocal = 0.0;
  for (PetscInt nodeIdx = 0; nodeIdx < _User.Nvlocal; nodeIdx++) {
    for (PetscInt var = 0; var < NVAR; var++)
      integralLocal += _norm_weigths[NVAR * nodeIdx + var] *
                       (PetscRealPart(pX[NVAR * nodeIdx + var]) *
                            PetscRealPart(pX[NVAR * nodeIdx + var]) +
                        PetscImaginaryPart(pX[NVAR * nodeIdx + var]) *
                            PetscImaginaryPart(pX[NVAR * nodeIdx + var]));
  }

  PetscReal integralGlobal;
  MPI_Allreduce(&integralLocal, &integralGlobal, 1, MPI_DOUBLE, MPI_SUM,
                MPI_COMM_WORLD);

  return integralGlobal;
}
PetscReal Solver::GetVNorm2(PetscScalar *pX) {
  PetscReal sumLocal = 0.0;
  for (PetscInt nodeIdx = 0; nodeIdx < _User.Nvlocal; nodeIdx++) {
    for (PetscInt var = 1; var < NVAR; var++)
      sumLocal += (PetscRealPart(pX[NVAR * nodeIdx + var]) *
                       PetscRealPart(pX[NVAR * nodeIdx + var]) +
                   PetscImaginaryPart(pX[NVAR * nodeIdx + var]) *
                       PetscImaginaryPart(pX[NVAR * nodeIdx + var]));
  }

  PetscReal sumGlobal;
  MPI_Allreduce(&sumLocal, &sumGlobal, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  return PetscSqrtReal(sumLocal);
}
void Solver::GetMaximumReIm(PetscScalar *pX, PetscInt p_var,
                            PetscReal *r_max_re, PetscReal *r_max_im) {
  PetscReal local_max_re = fabs(PetscRealPart(pX[NVAR * 0 + p_var]));
  PetscReal local_max_im = fabs(PetscImaginaryPart(pX[NVAR * 0 + p_var]));

  for (PetscInt nodeIdx = 1; nodeIdx < _User.Nvlocal; nodeIdx++) {
    if (local_max_re < fabs(PetscRealPart(pX[NVAR * nodeIdx + p_var])))
      local_max_re = fabs(PetscRealPart(pX[NVAR * nodeIdx + p_var]));

    if (local_max_im < fabs(PetscImaginaryPart(pX[NVAR * nodeIdx + p_var])))
      local_max_im = fabs(PetscImaginaryPart(pX[NVAR * nodeIdx + p_var]));
  }

  MPI_Allreduce(&local_max_re, r_max_re, 1, MPI_DOUBLE, MPI_MAX,
                MPI_COMM_WORLD);
  MPI_Allreduce(&local_max_im, r_max_im, 1, MPI_DOUBLE, MPI_MAX,
                MPI_COMM_WORLD);
}

PetscErrorCode Solver::Jacobian(SNES snes, Vec X, Mat J, Mat B, void *ptr) {
  AppCtx *user = (AppCtx *)ptr;
  return user->Eq->Jacobian(snes, X, J, B, ptr);
}
PetscErrorCode Solver::JacobianAdjoint(SNES snes, Vec X, Mat J, Mat B,
                                       void *ptr) {
  AppCtx *user = (AppCtx *)ptr;
  return user->Eq->JacobianAdjoint(snes, X, J, B, ptr);
}
PetscErrorCode Solver::JacobianNonZeros(SNES snes, Vec X, Mat J, Mat B,
                                        void *ptr) {
  AppCtx *user = (AppCtx *)ptr;
  return user->Eq->JacobianNonZeros(snes, X, J, B, ptr);
}
PetscErrorCode Solver::Function(SNES snes, Vec X, Vec F, void *ptr) {
  AppCtx *user = (AppCtx *)ptr;
  return user->Eq->Function(snes, X, F, ptr);
}
PetscErrorCode Solver::FunctionAdjoint(SNES snes, Vec X, Vec F, void *ptr) {
  AppCtx *user = (AppCtx *)ptr;
  return user->Eq->FunctionAdjoint(snes, X, F, ptr);
}

void Solver::InitialConditions_BaseFlow() {
  _User.Eq->InitialConditions(0, _User.GlobalX[0], &_User);
  UpdateLocalX0(0);

  for (int i = 0; i < _User.lN; i++)
    _User.lX00[_User.lN * 0 + i] = _User.lX0[_User.lN * 0 + i];

  for (int i = 0; i < _User.lN; i++)
    _User.lX1_e[_User.lN * 0 + i] = _User.lX0[_User.lN * 0 + i];
}

void Solver::InitialConditions_Perturbations() {
  for (PetscInt mode = 1; mode < _User.FM; mode++) {
    _User.Eq->InitialConditions(mode, _User.GlobalX[mode], &_User);
    UpdateLocalX0(mode);

    for (int i = 0; i < _User.lN; i++)
      _User.lX00[_User.lN * mode + i] = _User.lX0[_User.lN * mode + i];

    for (int i = 0; i < _User.lN; i++)
      _User.lX1_e[_User.lN * mode + i] = _User.lX0[_User.lN * mode + i];
  }
}

void Solver::InitialConditions_Perturbations_FromAdjointSolution(
    PetscReal *p_energy_0, PetscReal *p_energy_T) {
  for (PetscInt mode = 1; mode < _User.FM; mode++) {
    // MatMult(_User._matAH, _User.GlobalY[mode], _User.GlobalAux);

    // UpdateLocalVector(&(_User.lX0[_User.lN*mode]), _User.GlobalAux);
    UpdateLocalVector(&(_User.lX0[_User.lN * mode]), _User.GlobalY[mode]);

    for (int i = 0; i < _User.lN; i++)
      _User.lX0[_User.lN * mode + i] = 0.5 * p_energy_0[mode] *
                                       _User.lX0[_User.lN * mode + i] /
                                       _norm_weigths[i];
    ConfinePerturbations();
  }
}

void Solver::UpdateLocalX0(PetscInt p_mode) {
  const PetscScalar *x;
  VecScatterBegin(_User.Scatter, _User.GlobalX[p_mode], _User.LocalX,
                  INSERT_VALUES, SCATTER_FORWARD);
  VecScatterEnd(_User.Scatter, _User.GlobalX[p_mode], _User.LocalX,
                INSERT_VALUES, SCATTER_FORWARD);

  VecGetArrayRead(_User.LocalX, &x);
  for (int i = 0; i < _User.lN; i++) _User.lX0[_User.lN * p_mode + i] = x[i];
  VecRestoreArrayRead(_User.LocalX, &x);
}

void Solver::UpdateLocalVector(PetscScalar *r_local_vector,
                               Vec p_global_vector) {
  const PetscScalar *x;
  VecScatterBegin(_User.Scatter, p_global_vector, _User.LocalX, INSERT_VALUES,
                  SCATTER_FORWARD);
  VecScatterEnd(_User.Scatter, p_global_vector, _User.LocalX, INSERT_VALUES,
                SCATTER_FORWARD);

  VecGetArrayRead(_User.LocalX, &x);
  for (int i = 0; i < _User.lN; i++) r_local_vector[i] = x[i];
  VecRestoreArrayRead(_User.LocalX, &x);
}

void Solver::UpdateGlobalVector(PetscScalar *p_local_vector,
                                Vec *r_global_vector) {
  PetscScalar *x;
  VecGetArray(_User.LocalX, &x);
  for (PetscInt i = 0; i < NVAR * _User.Nvlocal; i++) x[i] = p_local_vector[i];
  VecRestoreArray(_User.LocalX, &x);

  VecScatterBegin(_User.ScatterWithoutGhosts, _User.LocalX, *r_global_vector,
                  INSERT_VALUES, SCATTER_REVERSE);
  VecScatterEnd(_User.ScatterWithoutGhosts, _User.LocalX, *r_global_vector,
                INSERT_VALUES, SCATTER_REVERSE);
}

void Solver::ConfinePerturbations() {
  for (PetscInt mode = 1; mode < _User.FM; mode++) {
    for (int i = 0; i < _User.lN; i++) {
      if (_input->GetBorderIdxByVtx(_User.GloInd[i / NVAR]) != -1 ||
          _input->_ic_region[mode].IsInRegion(_User.X1[i / NVAR],
                                              _User.X2[i / NVAR]) == false)
        _User.lX0[_User.lN * mode + i] = 0.0;
    }
  }
}

PetscErrorCode Solver::StepX0() {
  _monitor->StartTimeMeasurement(StepUpdate);
  PetscErrorCode ierr;
  const PetscScalar *x;
  for (PetscInt mode = 0; mode < _User.FM; mode++) {
    if (_input->_base_flow_type == "None" || mode != 0) {
      ierr = VecScatterBegin(_User.Scatter, _User.GlobalX[mode], _User.LocalX,
                             INSERT_VALUES, SCATTER_FORWARD);
      CHKERRQ(ierr);
      ierr = VecScatterEnd(_User.Scatter, _User.GlobalX[mode], _User.LocalX,
                           INSERT_VALUES, SCATTER_FORWARD);
      CHKERRQ(ierr);

      ierr = VecGetArrayRead(_User.LocalX, &x);
      CHKERRQ(ierr);

      // X_{-1}
      if (_User.FirstTimeStep == PETSC_FALSE) {
        for (int i = 0; i < _User.lN; i++)
          _User.lX00[_User.lN * mode + i] = _User.lX0[_User.lN * mode + i];
      }

      // X_{0}
      for (int i = 0; i < _User.lN; i++) _User.lX0[_User.lN * mode + i] = x[i];

      // Extrapolation: X_{1} = 2X_{0}-X_{-1}
      if (_User.FirstTimeStep == PETSC_FALSE) {
        for (int i = 0; i < _User.lN; i++)
          _User.lX1_e[_User.lN * mode + i] =
              2.0 * _User.lX0[_User.lN * mode + i] -
              _User.lX00[_User.lN * mode + i];
      } else {
        for (int i = 0; i < _User.lN; i++)
          _User.lX00[_User.lN * mode + i] = _User.lX0[_User.lN * mode + i];
        for (int i = 0; i < _User.lN; i++)
          _User.lX1_e[_User.lN * mode + i] = _User.lX0[_User.lN * mode + i];
      }

      // Rescale
      if (_input->_solver_rescale_enabled == PETSC_TRUE) {
        PetscReal max_w_re;
        PetscReal max_w_im;
        GetMaximumReIm(&(_User.lX0[_User.lN * 1]), 3, &max_w_re, &max_w_im);

        PetscReal max_w = max(max_w_re, max_w_im);
        if (max_w < _input->_solver_rescale_threshold) {
          for (int i = 0; i < NVAR * _User.Nvlocal; i++) {
            _User.lX00[_User.lN * 1 + i] = _User.lX00[_User.lN * 1 + i] /
                                           _input->_solver_rescale_threshold;
            _User.lX0[_User.lN * 1 + i] =
                _User.lX0[_User.lN * 1 + i] / _input->_solver_rescale_threshold;
            _User.lX1_e[_User.lN * 1 + i] = _User.lX1_e[_User.lN * 1 + i] /
                                            _input->_solver_rescale_threshold;
          }
        }
      }

      ierr = VecRestoreArrayRead(_User.LocalX, &x);
      CHKERRQ(ierr);
    }
  }
  _monitor->EndTimeMeasurement(StepUpdate);
  return 0;
}

PetscErrorCode Solver::StepY0() {
  PetscErrorCode ierr;

  // ======================================================
  // Base flow
  // const PetscScalar *x;
  PetscInt mode = 0;

  // ======================================================
  // Adjoint perturbations
  const PetscScalar *y;
  for (mode = 1; mode < _User.FM; mode++) {
    ierr = VecScatterBegin(_User.Scatter, _User.GlobalY[mode], _User.LocalX,
                           INSERT_VALUES, SCATTER_FORWARD);
    CHKERRQ(ierr);
    ierr = VecScatterEnd(_User.Scatter, _User.GlobalY[mode], _User.LocalX,
                         INSERT_VALUES, SCATTER_FORWARD);
    CHKERRQ(ierr);

    ierr = VecGetArrayRead(_User.LocalX, &y);
    CHKERRQ(ierr);

    // Y_{0}
    for (int i = 0; i < _User.lN; i++) _User.lY0[_User.lN * mode + i] = y[i];

    ierr = VecRestoreArrayRead(_User.LocalX, &y);
    CHKERRQ(ierr);
  }

  return 0;
}

void Solver::SaveLocalGlobalIdx() {
  if (_Rank == 0) {
    char fileName[512];
    FILE *fptr;
    sprintf(fileName, "%s/Info.dat", _input->_Output);
    fptr = fopen(fileName, "w");
    PetscFPrintf(PETSC_COMM_SELF, fptr, "%d %ld %ld %ld\n", DIM,
                 _input->_Mesh._NVertex, _input->_Mesh._NElement,
                 _input->_Mesh._NEdge);
    fclose(fptr);
  }

  // Проверка соответствия типов
  int typeSize;
  MPI_Type_size(MPI_LONG_LONG, &typeSize);
  if (typeSize != sizeof(PetscInt)) {
    printf("MPI_LONG_LONG (%d) != PetscInt (%d).\n", typeSize,
           (int)sizeof(PetscInt));
    PressEnterToContinue();
  }

  MPI_File fptr;
  MPI_Status status;
  char fileName[512];
  sprintf(fileName, "%s/GlobalIdx.dat", _input->_Output);
  MPI_File_open(MPI_COMM_WORLD, fileName, MPI_MODE_CREATE | MPI_MODE_WRONLY,
                MPI_INFO_NULL, &fptr);
  MPI_File_write_ordered(fptr, _MP->_GloInd, _MP->_Nvlocal, MPI_LONG_LONG,
                         &status);
  MPI_File_close(&fptr);

  sprintf(fileName, "%s/LocalIdx.dat", _input->_Output);
  MPI_File_open(MPI_COMM_WORLD, fileName, MPI_MODE_CREATE | MPI_MODE_WRONLY,
                MPI_INFO_NULL, &fptr);
  MPI_File_write_ordered(fptr, _MP->_LocInd, _MP->_Nvlocal, MPI_LONG_LONG,
                         &status);
  MPI_File_close(&fptr);

  MPI_Barrier(MPI_COMM_WORLD);
}

PetscErrorCode Solver::SaveCList() {
  // Cоставление локального массива CList
  PetscInt *cList;
  PetscMalloc(_User.Nelocal * (DIM + 2) * sizeof(PetscInt), &cList);

  // printf("neloc =%d \n", _User.Nelocal);
  for (PetscInt i = 0; i < _User.Nelocal; i++) {
    cList[i * (DIM + 2)] = _MP->_EGloInd[i];
    for (PetscInt j = 0; j < DIM + 1; j++)
      cList[i * (DIM + 2) + j + 1] = _MP->_Vertices[_User.VByE[i][j]];
  }

  // Запись в файл
  char fCList[512];
  MPI_File fptr;
  MPI_Status status;
  sprintf(fCList, "%s/CList.dat", _input->_Output);

  MPI_File_open(MPI_COMM_WORLD, fCList, MPI_MODE_CREATE | MPI_MODE_WRONLY,
                MPI_INFO_NULL, &fptr);
  MPI_File_write_ordered(fptr, cList, _User.Nelocal * (DIM + 2),
                         MPI_LONG_LONG_INT, &status);
  MPI_File_close(&fptr);

  PetscFree(cList);

  return 0;
}

PetscErrorCode Solver::SaveGrid() {
  MPI_Barrier(MPI_COMM_WORLD);
  if (_Rank == 0) {
    printf("\t\t Grid-Part1\n");
  }
  char fGrid[512];
  sprintf(fGrid, "%s/Grid.dat", _input->_Output);
  FILE *fptrGrid = PETSC_NULL;
  for (int t = 0; t < _Size; t++) {
    if (_Rank == t) {
      if (_Rank == 0) {
        fptrGrid = fopen(fGrid, "w");
      } else {
        fptrGrid = fopen(fGrid, "a");
      }

      if (_Rank == 0) {
        // Заголовок файла сетки
        PetscFPrintf(PETSC_COMM_SELF, fptrGrid, "TITLE = \"Grid: %s\"\n",
                     _input->_MeshFile);
        PetscFPrintf(PETSC_COMM_SELF, fptrGrid, "FILETYPE = GRID\n");
        if (DIM == 3) {
          PetscFPrintf(PETSC_COMM_SELF, fptrGrid,
                       "VARIABLES = \"X\", \"Y\", \"Z\"\n");
          PetscFPrintf(PETSC_COMM_SELF, fptrGrid,
                       "ZONE NODES=%ld, ELEMENTS=%ld, DATAPACKING=POINT, "
                       "ZONETYPE=FETETRAHEDRON\n",
                       _input->_Mesh._NVertex, _input->_Mesh._NElement);
        } else {
          PetscFPrintf(PETSC_COMM_SELF, fptrGrid, "VARIABLES = \"X\", \"Y\"\n");
          PetscFPrintf(PETSC_COMM_SELF, fptrGrid,
                       "ZONE NODES=%ld, ELEMENTS=%ld, DATAPACKING=POINT, "
                       "ZONETYPE=FETRIANGLE\n",
                       _input->_Mesh._NVertex, _input->_Mesh._NElement);
        }
      }
      for (PetscInt i = 0; i < _User.Nvlocal; i++) {
        // Файл сетки
        PetscFPrintf(PETSC_COMM_SELF, fptrGrid, "%.16e %.16e\n", _User.X1[i],
                     _User.X2[i]);
      }
      fclose(fptrGrid);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  // Connectivity list

  MPI_Barrier(MPI_COMM_WORLD);
  if (_Rank == 0) {
    printf("\t\t Connectivity list\n");
  }

  char fCList[512];
  FILE *fptrCList;
  sprintf(fCList, "%s/CList.dat", _input->_Output);

  for (int t = 0; t < _Size; t++) {
    if (_Rank == t) {
      if (_Rank == 0) {
        fptrCList = fopen(fCList, "w");
      } else {
        fptrCList = fopen(fCList, "a");
      }
      for (PetscInt i = 0; i < _User.Nelocal; i++) {
        if (DIM == 2)
          PetscFPrintf(PETSC_COMM_SELF, fptrCList, "%ld %ld %ld %ld \n",
                       _MP->_EGloInd[i], _MP->_Vertices[_User.VByE[i][0]],
                       _MP->_Vertices[_User.VByE[i][1]],
                       _MP->_Vertices[_User.VByE[i][2]]);
        else
          PetscFPrintf(PETSC_COMM_SELF, fptrCList, "%ld %ld %ld %ld %ld \n",
                       _MP->_EGloInd[i], _MP->_Vertices[_User.VByE[i][0]],
                       _MP->_Vertices[_User.VByE[i][1]],
                       _MP->_Vertices[_User.VByE[i][2]],
                       _MP->_Vertices[_User.VByE[i][3]]);
      }
      fclose(fptrCList);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (_Rank == 0) {
    printf("\t\t Grid-Part2\n");
  }
  if (_Rank == 0) {
    PetscInt *cList;
    printf("\t\t Grid-Part2-PetscMalloc-Start (size=%ld*%d*%ld)\n",
           _input->_Mesh._NElement, DIM + 1, sizeof(PetscInt));
    PetscMalloc(_input->_Mesh._NElement * (DIM + 1) * sizeof(PetscInt), &cList);
    printf("\t\t Grid-Part2-PetscMalloc-End\n");
    fptrCList = fopen(fCList, "r");
    PetscInt n;
    while (fscanf(fptrCList, "%ld ", &n) != EOF) {
      if (DIM == 2)
        fscanf(fptrCList, "%ld %ld %ld \n", &(cList[n * (DIM + 1)]),
               &(cList[n * (DIM + 1) + 1]), &(cList[n * (DIM + 1) + 2]));
      else
        fscanf(fptrCList, "%ld %ld %ld %ld \n", &(cList[n * (DIM + 1)]),
               &(cList[n * (DIM + 1) + 1]), &(cList[n * (DIM + 1) + 2]),
               &(cList[n * (DIM + 1) + 3]));
    }
    fclose(fptrCList);

    fptrGrid = fopen(fGrid, "a");
    for (PetscInt i = 0; i < _input->_Mesh._NElement; i++) {
      if (DIM == 2)
        PetscFPrintf(PETSC_COMM_SELF, fptrGrid, "%ld %ld %ld\n",
                     cList[i * (DIM + 1)] + 1, cList[i * (DIM + 1) + 1] + 1,
                     cList[i * (DIM + 1) + 2] + 1);
      else
        PetscFPrintf(PETSC_COMM_SELF, fptrGrid, "%ld %ld %ld %ld\n",
                     cList[i * (DIM + 1)] + 1, cList[i * (DIM + 1) + 1] + 1,
                     cList[i * (DIM + 1) + 2] + 1,
                     cList[i * (DIM + 1) + 3] + 1);
    }
    fclose(fptrGrid);
    PetscFree(cList);
  }
  return 0;
}
void Solver::CalculateMassFlux() {
  // ===========================================================
  // Вычисление потоков через границы
  PetscReal u[DIM], Un;
  for (PetscInt e = 0; e < _User.NEdgesLocal; e++) {
    if (_User.UniqueEdgeIndex[e] != -1) {
      for (PetscInt i = 0; i < DIM; i++) {
        u[i] = 0.0;
        for (PetscInt k = 0; k < DIM; k++)
          u[i] +=
              PetscRealPart(_User.lX0[NVAR * _User.BorderVerts[e][k] + i + 1]);

        u[i] = u[i] / DIM;
      }
      Un = 0.0;
      for (PetscInt k = 0; k < DIM; k++) Un += u[k] * _User.BorderN[e][k];

      _User.MassFlux[_input->GetBorderIdx(_User.GloEdgeIdx[e])] +=
          Un * _User.BorderS[e] * _input->_dT;
    }
  }
}

PetscErrorCode Solver::Save(PetscInt p_time_iteration) {
  if (_saving_regime != Off) {
    // Save solution
    if ((p_time_iteration % _input->_output_solution_period == 0 &&
         p_time_iteration >= _input->_output_solution_start) ||
        p_time_iteration == _input->_NT)
      SaveSolution();

    // Save forces
    if (p_time_iteration % _input->_output_force_period == 0 ||
        p_time_iteration == _input->_NT)
      SaveForces();

    // Save perturbations energy
    if (p_time_iteration % _input->_output_energy_period == 0 ||
        p_time_iteration == _input->_NT)
      SaveEnergy();

    // Save probes
    if (p_time_iteration % _input->_output_probe_period == 0 ||
        p_time_iteration == _input->_NT)
      SaveProbes();
  }

  return 0;
}

PetscErrorCode Solver::SaveEnergy() {
  _monitor->StartTimeMeasurement(SaveResults);

  PetscReal velocity_max_re[3], velocity_max_im[3], energy;

  for (PetscInt mode = 0; mode < _User.FM; mode++) {
    energy = GetNormEnergy(&(_User.lX0[_User.lN * mode]));
    for (PetscInt i = 0; i < 3; i++)
      GetMaximumReIm(&(_User.lX0[_User.lN * mode]), i + 1,
                     &(velocity_max_re[i]), &(velocity_max_im[i]));
    if (_Rank == 0) {
      FILE *fptrSolution;
      char fSolution[512];

      if (_saving_regime == Usual ||
          (_saving_regime == OptimalPertubations && mode == 0))
        sprintf(fSolution, "%s/Energy-%ld.dat", _input->_Output, mode);
      else if (_saving_regime == OptimalPertubations)
        sprintf(fSolution, "%s/Energy-[%.10lf+i%.10lf]-%ld.dat",
                _input->_Output,
                PetscRealPart(
                    _eigenproblem[mode - 1]._eigenvalue[_eigenvalue_index]),
                PetscImaginaryPart(
                    _eigenproblem[mode - 1]._eigenvalue[_eigenvalue_index]),
                mode);
      else if (_saving_regime == KSIterations)
        sprintf(fSolution, "%s/KS_iterations/Energy-%ld.dat", _input->_Output,
                mode);

      fptrSolution = fopen(fSolution, "a");

      PetscFPrintf(PETSC_COMM_SELF, fptrSolution, "%.16lf", _User.T);

      PetscFPrintf(PETSC_COMM_SELF, fptrSolution,
                   "\t%.20lf\t%.20lf\t%.20lf\t%.20lf\t%.20lf\t%.20lf\t%.20lf",
                   energy, velocity_max_re[0], velocity_max_im[0],
                   velocity_max_re[1], velocity_max_im[1], velocity_max_re[2],
                   velocity_max_im[2]);

      PetscFPrintf(PETSC_COMM_SELF, fptrSolution, "\n");
      fclose(fptrSolution);
    }
  }

  if (_Rank == 0) printf(" -> Energy has been saved: t = %.6lf\n", _User.T);

  _monitor->EndTimeMeasurement(SaveResults);

  return 0;
}

PetscErrorCode Solver::SaveProbes() {
  _monitor->StartTimeMeasurement(SaveResults);

  for (PetscInt t = 0; t < _Size; t++) {
    if (_Rank == t) {
      for (PetscInt mode = 0; mode < _User.FM; mode++) {
        for (PetscInt probe_idx = 0;
             probe_idx < _input->_output_probe_count_local; probe_idx++) {
          FILE *fptrSolution;
          char fSolution[512];

          if (_saving_regime == Usual ||
              (_saving_regime == OptimalPertubations && mode == 0))
            sprintf(fSolution, "%s/Probe-(%.5lf, %.5lf)-%ld.dat",
                    _input->_Output, _input->_output_probe_x_local[probe_idx],
                    _input->_output_probe_y_local[probe_idx], mode);
          else if (_saving_regime == OptimalPertubations)
            sprintf(fSolution,
                    "%s/Probe-(%.5lf, %.5lf)-[%.10lf+i%.10lf]-%ld.dat",
                    _input->_Output, _input->_output_probe_x_local[probe_idx],
                    _input->_output_probe_y_local[probe_idx],
                    PetscRealPart(
                        _eigenproblem[mode - 1]._eigenvalue[_eigenvalue_index]),
                    PetscImaginaryPart(
                        _eigenproblem[mode - 1]._eigenvalue[_eigenvalue_index]),
                    mode);

          fptrSolution = fopen(fSolution, "a");

          fprintf(fptrSolution, "%.16lf", _User.T);

          fprintf(
              fptrSolution,
              "\t%.20lf\t%.20lf\t%.20lf\t%.20lf\t%.20lf\t%.20lf\t%.20lf\t%."
              "20lf\n",
              PetscRealPart(
                  _User.lX0[_User.lN * mode +
                            NVAR * _input->_output_probe_idx_local[probe_idx] +
                            0]),
              PetscImaginaryPart(
                  _User.lX0[_User.lN * mode +
                            NVAR * _input->_output_probe_idx_local[probe_idx] +
                            0]),
              PetscRealPart(
                  _User.lX0[_User.lN * mode +
                            NVAR * _input->_output_probe_idx_local[probe_idx] +
                            1]),
              PetscImaginaryPart(
                  _User.lX0[_User.lN * mode +
                            NVAR * _input->_output_probe_idx_local[probe_idx] +
                            1]),
              PetscRealPart(
                  _User.lX0[_User.lN * mode +
                            NVAR * _input->_output_probe_idx_local[probe_idx] +
                            2]),
              PetscImaginaryPart(
                  _User.lX0[_User.lN * mode +
                            NVAR * _input->_output_probe_idx_local[probe_idx] +
                            2]),
              PetscRealPart(
                  _User.lX0[_User.lN * mode +
                            NVAR * _input->_output_probe_idx_local[probe_idx] +
                            3]),
              PetscImaginaryPart(
                  _User.lX0[_User.lN * mode +
                            NVAR * _input->_output_probe_idx_local[probe_idx] +
                            3]));

          fclose(fptrSolution);
        }
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  if (_Rank == 0) printf(" -> Probes has been saved: t = %.6lf\n", _User.T);

  _monitor->EndTimeMeasurement(SaveResults);

  return 0;
}

PetscErrorCode Solver::SaveForces() {
  _monitor->StartTimeMeasurement(SaveResults);

  PetscInt bIdx, ElementIdx;
  PetscReal Y_x[DIM][NVAR];  // Y_x[i][j] - производная jй компоненты по i-му
                             // направлению
  PetscReal tau[DIM][DIM];

  PetscReal p, div;  // давление

  for (PetscInt b = 0; b < _input->_output_force_count; b++) {
    for (PetscInt k = 0; k < 2 * DIM; k++) _R[2 * DIM * b + k] = 0.0;

    for (PetscInt e = 0; e < _User.NEdgesLocal; e++) {
      if (_User.UniqueEdgeIndex[e] != -1) {
        bIdx = _input->GetBorderIdx(_User.GloEdgeIdx[e]);

        if (_input->_BoundaryNo[bIdx] == _input->_output_force_IDs[b]) {
          // --------------------------------------------------------------
          // Силы давления
          p = 0.0;
          for (PetscInt k = 0; k < DIM; k++)
            p += PetscRealPart(_User.lX0[NVAR * _User.BorderVerts[e][k]]);
          p = p / DIM;

          for (PetscInt k = 0; k < DIM; k++)
            _R[2 * DIM * b + 2 * k + 0] +=
                p * _User.BorderN[e][k] * _User.BorderS[e];

          // --------------------------------------------------------------
          // Вязкие силы

          ElementIdx = _User.BorderElement[e];
          // Расчет производных Y_x
          for (PetscInt d = 0; d < DIM; d++) {
            for (PetscInt i = 0; i < NVAR; i++) {
              Y_x[d][i] = 0.0;
              for (PetscInt k = 0; k < D1; k++)
                Y_x[d][i] +=
                    PetscRealPart(
                        _User.lX0[NVAR * _User.VByE[ElementIdx][k] + i]) *
                    _User.Hx[D1_D * ElementIdx + DIM * k + d];
            }
          }
          div = 0.0;
          for (PetscInt i = 0; i < DIM; i++) div += Y_x[i][i + 1];

          for (PetscInt i = 0; i < DIM; i++) {
            for (PetscInt j = 0; j < DIM; j++) {
              tau[i][j] = (Y_x[j][i + 1] + Y_x[i][j + 1]);
              if (i == j) tau[i][j] -= 2.0 * div / 3.0;
            }
          }

          for (PetscInt k = 0; k < DIM; k++) {
            for (PetscInt i = 0; i < DIM; i++) {
              _R[2 * DIM * b + 2 * k + 1] +=
                  -tau[k][i] * _User.BorderN[e][i] * _User.BorderS[e];
            }
          }
        }
      }
    }
    for (PetscInt k = 0; k < DIM; k++)
      _R[2 * DIM * b + 2 * k + 1] = _R[2 * DIM * b + 2 * k + 1] / _input->_Re;
  }
  MPI_Reduce(_R, _RGlobal, 2 * DIM * _input->_output_force_count, MPI_DOUBLE,
             MPI_SUM, 0, MPI_COMM_WORLD);

  // Сохранение
  if (_Rank == 0) {
    FILE *fptrSolution;
    char fSolution[512];
    for (PetscInt i = 0; i < _input->_output_force_count; i++) {
      if (_saving_regime == Usual || _saving_regime == OptimalPertubations)
        sprintf(fSolution, "%s/Forces-%ld.dat", _input->_Output,
                _input->_output_force_IDs[i]);
      else if (_saving_regime == KSIterations)
        sprintf(fSolution, "%s/KS_iterations/Forces-%ld.dat", _input->_Output,
                _input->_output_force_IDs[i]);

      fptrSolution = fopen(fSolution, "a");

      if (DIM == 2) {
        PetscFPrintf(
            PETSC_COMM_SELF, fptrSolution,
            "%.16lf\t%.16lf\t%.16lf\t%.16lf\t%.16lf\t%.16lf\t%.16lf\n", _User.T,
            _RGlobal[2 * DIM * i + 2 * 0], _RGlobal[2 * DIM * i + 2 * 0 + 1],
            _RGlobal[2 * DIM * i + 2 * 0] + _RGlobal[2 * DIM * i + 2 * 0 + 1],
            _RGlobal[2 * DIM * i + 2 * 1], _RGlobal[2 * DIM * i + 2 * 1 + 1],
            _RGlobal[2 * DIM * i + 2 * 1] + _RGlobal[2 * DIM * i + 2 * 1 + 1]);
      } else
        PetscFPrintf(
            PETSC_COMM_SELF, fptrSolution,
            "%.16lf\t%.16lf\t%.16lf\t%.16lf\t%.16lf\t%.16lf\t%.16lf\t%.16lf\t%."
            "16lf\t%.16lf\n",
            _User.T, _RGlobal[2 * DIM * i + 2 * 0],
            _RGlobal[2 * DIM * i + 2 * 0 + 1],
            _RGlobal[2 * DIM * i + 2 * 0] + _RGlobal[2 * DIM * i + 2 * 0 + 1],
            _RGlobal[2 * DIM * i + 2 * 1], _RGlobal[2 * DIM * i + 2 * 1 + 1],
            _RGlobal[2 * DIM * i + 2 * 1] + _RGlobal[2 * DIM * i + 2 * 1 + 1],
            _RGlobal[2 * DIM * i + 2 * 2], _RGlobal[2 * DIM * i + 2 * 2 + 1],
            _RGlobal[2 * DIM * i + 2 * 2] + _RGlobal[2 * DIM * i + 2 * 2 + 1]);

      fclose(fptrSolution);
    }
  }

  if (_Rank == 0) printf(" -> Forces have been saved: t = %.6lf\n", _User.T);

  _monitor->EndTimeMeasurement(SaveResults);

  return 0;
}

PetscErrorCode Solver::SaveSolution() {
  _monitor->StartTimeMeasurement(SaveResults);
  FILE *fptrSolution;
  char fSolutionReal[512];
  char fSolutionImg[512];

  for (PetscInt mode = 0; mode < _User.FM; mode++) {
    if (_saving_regime == Usual ||
        (_saving_regime == OptimalPertubations && mode == 0)) {
      sprintf(fSolutionReal, "%s/mode-%ld/Solution-%.10lf.dat", _input->_Output,
              mode, _User.T);
      sprintf(fSolutionImg, "%s/mode-%ld/ISolution-%.10lf.dat", _input->_Output,
              mode, _User.T);
    } else if (_saving_regime == OptimalPertubations) {
      sprintf(
          fSolutionReal, "%s/mode-%ld/Solution-[%.10lf+i%.10lf]-%.10lf.dat",
          _input->_Output, mode,
          PetscRealPart(_eigenproblem[mode - 1]._eigenvalue[_eigenvalue_index]),
          PetscImaginaryPart(
              _eigenproblem[mode - 1]._eigenvalue[_eigenvalue_index]),
          _User.T);
      sprintf(
          fSolutionImg, "%s/mode-%ld/ISolution-[%.10lf+i%.10lf]-%.10lf.dat",
          _input->_Output, mode,
          PetscRealPart(_eigenproblem[mode - 1]._eigenvalue[_eigenvalue_index]),
          PetscImaginaryPart(
              _eigenproblem[mode - 1]._eigenvalue[_eigenvalue_index]),
          _User.T);
    } else if (_saving_regime == KSIterations) {
      sprintf(fSolutionReal, "%s/KS_iterations/mode-%ld/Solution-%.10lf.dat",
              _input->_Output, mode, _User.T);
      sprintf(fSolutionImg, "%s/KS_iterations/mode-%ld/ISolution-%.10lf.dat",
              _input->_Output, mode, _User.T);
    }

    for (PetscInt t = 0; t < _Size; t++) {
      if (_Rank == t) {
        // real and imaginary parts
        for (PetscInt i = 0; i < NVAR * _User.Nvlocal; i++) {
          _auxLocalX_real[i] = PetscRealPart(_User.lX0[_User.lN * mode + i]);
          _auxLocalX_img[i] =
              PetscImaginaryPart(_User.lX0[_User.lN * mode + i]);
        }

        if (_Rank == 0) {
          fptrSolution = fopen(fSolutionReal, "wb");
        } else {
          fptrSolution = fopen(fSolutionReal, "ab");
        }
        fwrite(_auxLocalX_real, sizeof(PetscReal), NVAR * _User.Nvlocal,
               fptrSolution);
        fclose(fptrSolution);

        if (_Rank == 0) {
          fptrSolution = fopen(fSolutionImg, "wb");
        } else {
          fptrSolution = fopen(fSolutionImg, "ab");
        }
        fwrite(_auxLocalX_img, sizeof(PetscReal), NVAR * _User.Nvlocal,
               fptrSolution);
        fclose(fptrSolution);
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }

  if (_Rank == 0) printf(" -> Solution has been saved: t = %.6lf\n", _User.T);

  _monitor->EndTimeMeasurement(SaveResults);
  return 0;
}

PetscErrorCode Solver::SaveNormWeights() {
  _monitor->StartTimeMeasurement(SaveResults);
  FILE *fptr;
  char path[512];

  sprintf(path, "%s/norm_weights.dat", _input->_Output);

  for (PetscInt t = 0; t < _Size; t++) {
    if (_Rank == t) {
      if (_Rank == 0) {
        fptr = fopen(path, "wb");
      } else {
        fptr = fopen(path, "ab");
      }
      fwrite(_norm_weigths, sizeof(PetscReal), NVAR * _User.Nvlocal, fptr);
      fclose(fptr);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  _monitor->EndTimeMeasurement(SaveResults);
  return 0;
}

PetscErrorCode Solver::TestNullSpace(Mat pM, Vec pX) {
  PetscErrorCode ierr;
  Vec vecP;
  MatNullSpace nullspace;

  ierr = VecDuplicate(pX, &vecP);
  CHKERRQ(ierr);

  ierr = VecSet(vecP, 0.0);
  CHKERRQ(ierr);

  PetscInt istart, iend;
  VecGetOwnershipRange(vecP, &istart, &iend);

  PetscScalar val = 1.0;
  for (PetscInt i = istart; i < iend; i++)
    if (i % NVAR == 0) VecSetValues(vecP, 1, &i, &val, INSERT_VALUES);

  ierr = VecAssemblyBegin(vecP);
  CHKERRQ(ierr);
  ierr = VecAssemblyEnd(vecP);
  CHKERRQ(ierr);

  VecNormalize(vecP, NULL);
  MatNullSpaceCreate(MPI_COMM_WORLD, PETSC_FALSE, 1, &vecP, &nullspace);

  VecDestroy(&vecP);
  PetscBool result;
  MatNullSpaceTest(nullspace, pM, &result);

  if (result == PETSC_TRUE)
    printf("Nullspace is correct!\n");
  else
    printf("ERROR: Nullspace is not correct!\n");

  MatNullSpaceDestroy(&nullspace);

  return 0;
}
