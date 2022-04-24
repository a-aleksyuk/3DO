#include "input.h"

#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <fstream>
#include <iostream>
#include <string>

using namespace std;
using namespace O3D;

bool copyFile(const char *SRC, const char *DEST) {
  std::ifstream src(SRC, std::ios::binary);
  std::ofstream dest(DEST, std::ios::binary);
  dest << src.rdbuf();
  return src && dest;
}

Input::Input() { MPI_Comm_rank(MPI_COMM_WORLD, &_rank); }
Input::~Input() {
  PetscFree(_ic_solution);
  delete[] _ic_region;

  PetscFree(_BoundaryNo);
  PetscFree(_BorderIdxByVrtx);
  PetscFree(_BorderIdxByVrtx2);
  PetscFree(_BorderIdxByEdge);
  PetscFree(_output_force_IDs);

  if (_output_probe_count > 0) {
    PetscFree(_output_probe_x);
    PetscFree(_output_probe_y);
    PetscFree(_output_probe_idx_local);
    PetscFree(_output_probe_x_local);
    PetscFree(_output_probe_y_local);
  }

  for (PetscInt i = 0; i < 3; i++) {
    PetscFree(_ForceF[i]);
  }
  PetscFree(_ForceT);
  PetscFree(_ForceF);
  if (_ForceType == 'O') {
    PetscFree(_ForceA0);
    PetscFree(_ForceA1);
    PetscFree(_ForceF0);
    PetscFree(_ForceF1);
  }

  PetscFree(_solver_fem_tau_SUPG);
  PetscFree(_solver_fem_tau_PSPG);
  PetscFree(_solver_fem_tau_LSIC);
  PetscFree(_solver_snes_dirichlet_conditions_new);
  PetscFree(_solver_snes_near_null_space);
  delete[] _solver_snes_type;

  PetscFree(_time_scheme_stabilisation);
  PetscFree(_time_scheme_nonlinear_term);
  PetscFree(_time_scheme_acceleration);

  if (_base_flow_type != "None") delete[] _base_flow_times;

  delete[] _BC;
}

void Input::Initialize(char *p_profect_folder, char *p_input_file) {
  _ProjectDir = p_profect_folder;

  sprintf(_Input, "%s/input/", _ProjectDir);

  // ============================================
  // Read parameters
  if (_rank == 0) {
    printf(" # Read input...\n");
  }

  FILE *fRead;
  string file_name;
  string folder_name;
  char file_path[512];
  sprintf(file_path, "%s/%s", _Input, p_input_file);
  ifstream input_stream(file_path);

  input_stream >> _ipnut_json;
  input_stream.close();

  Version = _ipnut_json["Version"];

  // --------------------------------------------
  // Problem formulation
  if (_rank == 0) {
    printf("    | Problem formulation...\n");
  }
  _problem_type = _ipnut_json["Problem formulation"]["Type"];
  _Re = _ipnut_json["Problem formulation"]["Parameters"]["Re"];
  _Fr = _ipnut_json["Problem formulation"]["Parameters"]["Fr"];

  // Force
  if (_rank == 0) {
    printf("    | | Force...\n");
  }
  if (_ipnut_json["Problem formulation"]["Parameters"]["Force"]["Type"] ==
      "Constant") {
    _ForceType = 'C';
    _ForceNT = 1;
    PetscMalloc(3 * sizeof(PetscReal *), &(_ForceF));
    PetscMalloc(_ForceNT * sizeof(PetscReal), &(_ForceT));
    _ForceT[0] = 0;

    for (int i = 0; i < 3; i++) {
      PetscMalloc(_ForceNT * sizeof(PetscReal), &(_ForceF[i]));
      _ForceF[i][0] =
          _ipnut_json["Problem formulation"]["Parameters"]["Force"]["Value"][i];
    }
  } else if (_ipnut_json["Problem formulation"]["Parameters"]["Force"]
                        ["Type"] == "File") {
    _ForceType = 'F';
    PetscMalloc(3 * sizeof(PetscReal *), &(_ForceF));

    file_name =
        _ipnut_json["Problem formulation"]["Parameters"]["Force"]["File"];
    sprintf(file_path, "%s/%s", _Input, file_name.c_str());

    fRead = fopen(file_path, "r");
    if (!fRead) {
      printf("Could no open output file (Reading Function): %s\n", file_path);
      PressEnterToContinue();
    }
    fscanf(fRead, "%ld%*[^\n]\n", &_ForceNT);
    PetscMalloc(_ForceNT * sizeof(PetscReal), &(_ForceT));
    for (int i = 0; i < 3; i++)
      PetscMalloc(_ForceNT * sizeof(PetscReal), &(_ForceF[i]));

    for (PetscInt i = 0; i < _ForceNT; i++)
      fscanf(fRead, "%lf %lf %lf %lf%*[^\n]\n", &(_ForceT[i]), &(_ForceF[0][i]),
             &(_ForceF[1][i]), &(_ForceF[2][i]));

    fclose(fRead);
  } else if (_ipnut_json["Problem formulation"]["Parameters"]["Force"]
                        ["Type"] == "Oscillations") {
    _ForceType = 'O';
    PetscMalloc(3 * sizeof(PetscReal), &(_ForceA0));
    PetscMalloc(3 * sizeof(PetscReal), &(_ForceA1));
    PetscMalloc(3 * sizeof(PetscReal), &(_ForceF0));
    PetscMalloc(3 * sizeof(PetscReal), &(_ForceF1));

    for (int i = 0; i < 3; i++) {
      _ForceA0[i] =
          _ipnut_json["Problem formulation"]["Parameters"]["Force"]["A0"][i];
      _ForceA1[i] =
          _ipnut_json["Problem formulation"]["Parameters"]["Force"]["A1"][i];
      _ForceF0[i] =
          _ipnut_json["Problem formulation"]["Parameters"]["Force"]["F0"][i];
      _ForceF1[i] =
          _ipnut_json["Problem formulation"]["Parameters"]["Force"]["F1"][i];
    }
  } else {
    printf("ERROR: Input-Forces\n");
    PressEnterToContinue();
  }

  // Time
  if (_rank == 0) {
    printf("    | | Time...\n");
  }

  _T_start = _ipnut_json["Problem formulation"]["Time"]["t_0"];
  _T_end = _ipnut_json["Problem formulation"]["Time"]["t_1"];
  _dT = _ipnut_json["Problem formulation"]["Time"]["Step"];
  _NT = (PetscInt)std::round((_T_end - _T_start) / _dT);

  // Space
  if (_rank == 0) {
    printf("    | | Space...\n");
  }

  _SpaceType = _ipnut_json["Problem formulation"]["Space"]["Type"];
  file_name = _ipnut_json["Problem formulation"]["Space"]["Mesh"];

  sprintf(_MeshFile, "%s", file_name.c_str());
  sprintf(file_path, "%s/%s", _Input, _MeshFile);
  _Mesh.InitializeFromFile(file_path);
  _L = _ipnut_json["Problem formulation"]["Space"]["3D modes"]["z-period"];
  _FModes = _ipnut_json["Problem formulation"]["Space"]["3D modes"]["Count"];
  _FModes++;

  // Boundary conditions
  if (_rank == 0) {
    printf("    | | Boundary conditions...\n");
  }

  _BC = new BoundaryConditions[_FModes];
  for (PetscInt mode = 0; mode < _FModes; mode++) {
    if (mode == 0)
      _BC[mode].Initialize(
          _ipnut_json["Problem formulation"]["Boundary conditions"],
          "Base flow", _Input);
    else if (_ipnut_json["Problem formulation"]["Boundary conditions"].contains(
                 "Perturbations-" + to_string(mode)))
      _BC[mode].Initialize(
          _ipnut_json["Problem formulation"]["Boundary conditions"],
          "Perturbations-" + to_string(mode), _Input);
    else
      _BC[mode].Initialize(
          _ipnut_json["Problem formulation"]["Boundary conditions"],
          "Perturbations", _Input);
  }

  weak_form_type = 0;
  for (PetscInt bIdx = 0;
       bIdx <
       _ipnut_json["Problem formulation"]["Boundary conditions"]["Count"];
       bIdx++) {
    if (_ipnut_json["Problem formulation"]["Boundary conditions"]["Base flow"]
                   [bIdx]["Type"] == "Pseudo traction free")
      weak_form_type = 1;
  }

  if (_rank == 0 && weak_form_type == 0)
    printf("Weak form: 'traction free'\n");
  else if (_rank == 0 && weak_form_type == 1)
    printf("Weak form: 'pseudo traction free'\n");

  MPI_Barrier(MPI_COMM_WORLD);

  PetscInt bNo = -1, vrtx;

  _boundaries_count =
      _ipnut_json["Problem formulation"]["Boundary conditions"]["Count"];

  PetscMalloc(_boundaries_count * sizeof(PetscInt), &(_BoundaryNo));
  PetscMalloc(_Mesh._NVertex * sizeof(PetscInt), &_BorderIdxByVrtx);
  PetscMalloc(_Mesh._NVertex * sizeof(PetscInt), &_BorderIdxByVrtx2);
  PetscMalloc(_Mesh._NEdge * sizeof(PetscInt), &_BorderIdxByEdge);

  for (PetscInt k = 0; k < _boundaries_count; k++)
    _BoundaryNo[k] =
        _ipnut_json["Problem formulation"]["Boundary conditions"]["IDs"][k];

  MPI_Barrier(MPI_COMM_WORLD);

  for (vrtx = 0; vrtx < _Mesh._NVertex; vrtx++) {
    _BorderIdxByVrtx[vrtx] = -1;
    _BorderIdxByVrtx2[vrtx] = -1;
  }

  for (PetscInt i = 0; i < _Mesh._NEdge; i++) {
    bNo = -1;
    for (PetscInt k = 0; k < _boundaries_count; k++) {
      if (_BoundaryNo[k] == _Mesh._PhysicalEntity[i]) {
        bNo = k;
        _BorderIdxByEdge[i] = k;
        break;
      }
    }

    for (PetscInt j = 0; j < DIM; j++) {
      vrtx = _Mesh._VertsEdge[DIM * i + j];
      if (_BorderIdxByVrtx[vrtx] == -1)
        _BorderIdxByVrtx[vrtx] = bNo;
      else if (_BoundaryNo[bNo] != _BoundaryNo[_BorderIdxByVrtx[vrtx]]) {
        if (_rank == 0)
          printf(
              "### One point <-> two boundaries: PointIdx=%ld Boundary1=%ld "
              "Boundary2=%ld\n",
              vrtx, _BoundaryNo[_BorderIdxByVrtx[vrtx]], _BoundaryNo[bNo]);

        _BorderIdxByVrtx2[vrtx] = bNo;
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // Initial conditions
  if (_rank == 0) {
    printf("    | | Initial conditions...\n");
  }

  PetscMalloc(_FModes * _Mesh._NVertex * NVAR * sizeof(PetscScalar),
              &_ic_solution);
  _ic_region = new Region[_FModes];

  // Allocate additional memory for reading from file
  PetscInt *LocalIdx, *GlobalIdx, *LocalToGlobal;
  PetscReal *Solution, *ISolution;
  if (_ipnut_json["Problem formulation"]["Initial conditions"]["Base flow"]
                 ["Type"] == "File" ||
      _ipnut_json["Problem formulation"]["Initial conditions"]["Perturbations"]
                 ["Type"] == "File") {
    PetscMalloc(_Mesh._NVertex * sizeof(PetscInt), &LocalIdx);
    PetscMalloc(_Mesh._NVertex * sizeof(PetscInt), &GlobalIdx);
    PetscMalloc(_Mesh._NVertex * sizeof(PetscInt), &LocalToGlobal);

    PetscMalloc(_Mesh._NVertex * NVAR * sizeof(PetscReal), &Solution);
    PetscMalloc(_Mesh._NVertex * NVAR * sizeof(PetscReal), &ISolution);
  }

  string tag;
  // PetscInt mode_start, mode_end;
  for (PetscInt mode = 0; mode < _FModes; mode++) {
    if (mode == 0) {
      tag = "Base flow";
    } else if (_ipnut_json["Problem formulation"]["Initial conditions"]
                   .contains("Perturbations-" + to_string(mode))) {
      tag = "Perturbations-" + to_string(mode);
    } else {
      tag = "Perturbations";
    }

    if (_ipnut_json["Problem formulation"]["Initial conditions"][tag]["Type"] ==
        "File") {
      PetscInt FileSize;
      folder_name = _ipnut_json["Problem formulation"]["Initial conditions"]
                               [tag]["Folder"];
      file_name =
          _ipnut_json["Problem formulation"]["Initial conditions"][tag]["File"];

      // -------------------------------------------------------------
      // Local.dat
      sprintf(file_path, "%s/%s/LocalIdx.dat", _ProjectDir,
              folder_name.c_str());
      fRead = fopen(file_path, "rb");
      if (!fRead) {
        printf("Could not open input file: %s\n", file_path);
        PressEnterToContinue();
      }

      // CHECK: file size
      fseek(fRead, 0L, SEEK_END);
      FileSize = ftell(fRead);
      fseek(fRead, 0L, SEEK_SET);
      if (_Mesh._NVertex * sizeof(PetscInt) != FileSize) {
        printf(
            "ERROR: size of LocalIdx.dat (%ld) != Nv*sizeof(PetscInt) (%ld)\n",
            FileSize, _Mesh._NVertex * sizeof(PetscInt));
        PressEnterToContinue();
      }

      // Read LocalIdx
      fread(LocalIdx, sizeof(PetscInt), _Mesh._NVertex, fRead);
      fclose(fRead);

      // -------------------------------------------------------------
      // Global.dat
      sprintf(file_path, "%s/%s/GlobalIdx.dat", _ProjectDir,
              folder_name.c_str());
      fRead = fopen(file_path, "rb");
      if (!fRead) {
        printf("Could not open input file: %s\n", file_path);
        PressEnterToContinue();
      }

      // CHECK: file size
      fseek(fRead, 0L, SEEK_END);
      FileSize = ftell(fRead);
      fseek(fRead, 0L, SEEK_SET);
      if (_Mesh._NVertex * sizeof(PetscInt) != FileSize) {
        printf(
            "ERROR: size of GlobalIdx.dat (%ld) != Nv*sizeof(PetscInt) (%ld)\n",
            FileSize, _Mesh._NVertex * sizeof(PetscInt));
        PressEnterToContinue();
      }

      // Read GlobalIdx
      fread(GlobalIdx, sizeof(PetscInt), _Mesh._NVertex, fRead);
      fclose(fRead);

      // -------------------------------------------------------------
      // Create Local to Global Map
      for (PetscInt i = 0; i < _Mesh._NVertex; i++)
        LocalToGlobal[LocalIdx[i]] = GlobalIdx[i];

      // -------------------------------------------------------------
      // Read Solution
      // Read real part
      sprintf(file_path, "%s/%s/mode-%ld/%s", _ProjectDir, folder_name.c_str(),
              mode, file_name.c_str());
      fRead = fopen(file_path, "rb");
      if (!fRead) {
        printf("Could not open input file: %s\n", file_path);
        PressEnterToContinue();
      }
      fread(Solution, sizeof(PetscReal), _Mesh._NVertex * NVAR, fRead);
      fclose(fRead);

      // Read imaginary part
      sprintf(file_path, "%s/%s/mode-%ld/I%s", _ProjectDir, folder_name.c_str(),
              mode, file_name.c_str());
      fRead = fopen(file_path, "rb");
      if (!fRead) {
        printf("Could not open input file: %s\n", file_path);
        PressEnterToContinue();
      }
      fread(ISolution, sizeof(PetscReal), _Mesh._NVertex * NVAR, fRead);
      fclose(fRead);

      // Convert to complex numbers
      for (PetscInt vtx = 0; vtx < _Mesh._NVertex; vtx++)
        for (PetscInt k = 0; k < NVAR; k++)
          _ic_solution[NVAR * _Mesh._NVertex * mode +
                       NVAR * LocalToGlobal[vtx] + k] =
              Solution[NVAR * vtx + k] + PETSC_i * ISolution[NVAR * vtx + k];
    } else if (_ipnut_json["Problem formulation"]["Initial conditions"][tag]
                          ["Type"] == "Values") {
      SetInitialDistribution(
          0, mode,
          _ipnut_json["Problem formulation"]["Initial conditions"][tag]
                     ["Pressure"]);
      SetInitialDistribution(
          1, mode,
          _ipnut_json["Problem formulation"]["Initial conditions"][tag]
                     ["Velocity_1"]);
      SetInitialDistribution(
          2, mode,
          _ipnut_json["Problem formulation"]["Initial conditions"][tag]
                     ["Velocity_2"]);
      SetInitialDistribution(
          3, mode,
          _ipnut_json["Problem formulation"]["Initial conditions"][tag]
                     ["Velocity_3"]);
    } else if (_ipnut_json["Problem formulation"]["Initial conditions"][tag]
                          ["Type"] == "Random") {
      // Random initial perturbations to all 3D modes
      PetscScalar perturbation;
      PetscReal real_part, imaginary_part, norm;

      _ic_region[mode].Initialize(
          _ipnut_json["Problem formulation"]["Initial conditions"][tag]
                     ["Region"]);

      for (PetscInt k = 0; k < NVAR; k++) {
        real_part = _ipnut_json["Problem formulation"]["Initial conditions"]
                               [tag]["Direction"][k][0];
        imaginary_part =
            _ipnut_json["Problem formulation"]["Initial conditions"][tag]
                       ["Direction"][k][1];

        perturbation =
            ((PetscReal)_ipnut_json["Problem formulation"]["Initial conditions"]
                                   [tag]["Amplitude"][k]);
        perturbation = perturbation * (real_part + PETSC_i * imaginary_part) /
                       PetscSqrtReal(real_part * real_part +
                                     imaginary_part * imaginary_part);

        for (PetscInt vtx = 0; vtx < _Mesh._NVertex; vtx++) {
          if (_ic_region[mode].IsInRegion(_Mesh._X1[vtx], _Mesh._X2[vtx]))
            _ic_solution[NVAR * _Mesh._NVertex * mode + NVAR * vtx + k] =
                (2.0 * ((PetscReal)rand()) / ((PetscReal)RAND_MAX) - 1.0) *
                perturbation;
          else
            _ic_solution[NVAR * _Mesh._NVertex * mode + NVAR * vtx + k] = 0.0;
        }
      }
    }
  }

  if (_ipnut_json["Problem formulation"]["Initial conditions"]["Base flow"]
                 ["Type"] == "File" ||
      _ipnut_json["Problem formulation"]["Initial conditions"]["Perturbations"]
                 ["Type"] == "File") {
    // Free additional memory for reading from file
    PetscFree(LocalIdx);
    PetscFree(GlobalIdx);
    PetscFree(LocalToGlobal);
    PetscFree(Solution);
    PetscFree(ISolution);
  }

  // Base flow solution
  if (_rank == 0) {
    printf("    | | Base flow solution...\n");
  }

  if (_problem_type != "DNS" &&
      _ipnut_json["Problem formulation"].contains("Base flow solution")) {
    _base_flow_type =
        _ipnut_json["Problem formulation"]["Base flow solution"]["Type"];
    if (_base_flow_type == "None") {
      _base_flow_type = "None";
      _jacobi_matrix_interpolation = PETSC_FALSE;
    } else {
      _base_flow_folder =
          _ipnut_json["Problem formulation"]["Base flow solution"]["Folder"];
      _base_flow_times_file =
          _ipnut_json["Problem formulation"]["Base flow solution"]["File"];
      _base_flow_period =
          _ipnut_json["Problem formulation"]["Base flow solution"]["Period"];

      sprintf(file_path, "%s/%s", _Input, _base_flow_times_file.c_str());
      fRead = fopen(file_path, "r");
      if (!fRead) {
        printf("Could not open input file: %s\n", file_path);
        PressEnterToContinue();
      }

      fscanf(fRead, "%ld%*[^\n]\n", &_base_flow_times_count);

      _base_flow_modes_count = _base_flow_times_count / 2 + 1;
      _base_flow_times = new string[_base_flow_times_count];
      char buf[512];
      for (PetscInt i = 0; i < _base_flow_times_count; i++) {
        fscanf(fRead, "%s%*[^\n]\n", &(buf[0]));
        _base_flow_times[i] = string(buf);
      }
      fclose(fRead);
      if (_ipnut_json["Problem formulation"].contains("Base flow solution"))
        _jacobi_matrix_interpolation =
            ConvertBool(_ipnut_json["Problem formulation"]["Base flow solution"]
                                   ["Jacobian interpolation"]);
      else
        _jacobi_matrix_interpolation = PETSC_FALSE;
    }
  } else {
    _base_flow_type = "None";
    _jacobi_matrix_interpolation = PETSC_FALSE;
  }

  // Pressure constant
  if (_rank == 0) {
    printf("    | | Pressure constant...\n");
  }

  _pressure_constant_fixed = ConvertBool(
      _ipnut_json["Problem formulation"]["Pressure constant"]["Fixed"]);
  if (_pressure_constant_fixed == PETSC_TRUE) {
    _pressure_constant_value =
        _ipnut_json["Problem formulation"]["Pressure constant"]["Value"];

    PetscReal x = _pressure_constant_fixed =
        _ipnut_json["Problem formulation"]["Pressure constant"]["Point"][0];
    PetscReal y = _pressure_constant_fixed =
        _ipnut_json["Problem formulation"]["Pressure constant"]["Point"][1];

    // Closest node
    PetscReal curDist2, minDist2;
    for (PetscInt v = 0; v < _Mesh._NVertex; v++) {
      curDist2 = (_Mesh._X1[v] - x) * (_Mesh._X1[v] - x) +
                 (_Mesh._X2[v] - y) * (_Mesh._X2[v] - y);
      if (v == 0 || minDist2 > curDist2) {
        minDist2 = curDist2;
        _pressure_constant_node = v;
      }
    }
  } else
    _pressure_constant_node = -1;

  // --------------------------------------------
  // Output
  if (_rank == 0) {
    printf("    | Output...\n");
  }

  _output_force_period = _ipnut_json["Output"]["Force"]["Period"];
  if (_output_force_period < 1 || _output_force_period > _NT - 1) {
    _output_force_period = _NT;
  }

  _output_force_count = _ipnut_json["Output"]["Force"]["Count"];

  PetscMalloc(_output_force_count * sizeof(PetscInt), &_output_force_IDs);
  for (int i = 0; i < _output_force_count; i++)
    _output_force_IDs[i] = _ipnut_json["Output"]["Force"]["IDs"][i];

  _output_solution_period = _ipnut_json["Output"]["Solution"]["Period"];
  if (_output_solution_period < 1 || _output_solution_period > _NT - 1) {
    _output_solution_period = _NT;
  }

  _output_solution_start = _ipnut_json["Output"]["Solution"]["Start"];

  _output_monitor_period = _ipnut_json["Output"]["Monitor"]["Period"];
  if (_output_monitor_period < 1 || _output_monitor_period > _NT - 1) {
    _output_monitor_period = _NT;
  }

  _output_efficiency_enabled =
      ConvertBool(_ipnut_json["Output"]["Monitor"]["Efficiency"]);

  _output_energy_period = _ipnut_json["Output"]["Energy"]["Period"];

  if (_ipnut_json["Output"].contains("TSA"))
    _output_eigenvalues_monitor =
        _ipnut_json["Output"]["TSA"]["Eigenvalues to monitor"];
  else
    _output_eigenvalues_monitor = 2;

  if (_ipnut_json["Output"].contains("Probe")) {
    _output_probe_period = _ipnut_json["Output"]["Probe"]["Period"];
    _output_probe_count = _ipnut_json["Output"]["Probe"]["Count"];
    PetscMalloc(_output_probe_count * sizeof(PetscReal), &_output_probe_x);
    PetscMalloc(_output_probe_count * sizeof(PetscReal), &_output_probe_y);
    PetscMalloc(_output_probe_count * sizeof(PetscInt),
                &_output_probe_idx_local);
    PetscMalloc(_output_probe_count * sizeof(PetscReal),
                &_output_probe_x_local);
    PetscMalloc(_output_probe_count * sizeof(PetscReal),
                &_output_probe_y_local);

    for (PetscInt k = 0; k < _output_probe_count; k++) {
      _output_probe_x[k] = _ipnut_json["Output"]["Probe"]["(x, y)"][k][0];
      _output_probe_y[k] = _ipnut_json["Output"]["Probe"]["(x, y)"][k][1];
    }
  } else {
    _output_probe_count = 0;
    _output_probe_period = 100000;
  }

  // --------------------------------------------
  // Solver
  if (_rank == 0) {
    printf("    | Solver...\n");
  }

  Check_Input_Solver();

  if (_problem_type == "TSA")
    _solver_count = 3;
  else if (_ipnut_json["Solver"].contains("Perturbations"))
    _solver_count = 2;
  else
    _solver_count = 1;

  PetscMalloc(_FModes * sizeof(PetscReal), &_solver_fem_tau_SUPG);
  PetscMalloc(_FModes * sizeof(PetscReal), &_solver_fem_tau_PSPG);
  PetscMalloc(_FModes * sizeof(PetscReal), &_solver_fem_tau_LSIC);
  PetscMalloc(_FModes * sizeof(PetscBool),
              &_solver_snes_dirichlet_conditions_new);

  PetscMalloc(_FModes * sizeof(TimeSchemeFunction),
              &_time_scheme_stabilisation);
  PetscMalloc(_FModes * sizeof(TimeSchemeFunction),
              &_time_scheme_nonlinear_term);
  PetscMalloc(_FModes * sizeof(TimeSchemeAcceleration),
              &_time_scheme_acceleration);

  PetscMalloc(_solver_count * sizeof(PetscBool), &_solver_snes_near_null_space);
  _solver_snes_type = new string[_solver_count];

  for (PetscInt i = 0; i < _solver_count; i++) {
    if (i > 1) {
      _solver_snes_near_null_space[i] = _solver_snes_near_null_space[i - 1];
      _solver_snes_type[i] = _solver_snes_type[i - 1];
    } else {
      if (i == 0) {
        tag = "Base flow";
      } else if (i == 1) {
        tag = "Perturbations";
      }

      _solver_snes_type[i] = _ipnut_json["Solver"][tag]["SNES"]["Type"];
      _solver_snes_near_null_space[i] = ConvertBool(
          _ipnut_json["Solver"][tag]["SNES"]["J matrix"]["Near null space"]);
    }
  }

  for (PetscInt i = 0; i < _FModes; i++) {
    if (i > 1) {
      _solver_fem_tau_SUPG[i] = _solver_fem_tau_SUPG[i - 1];
      _solver_fem_tau_PSPG[i] = _solver_fem_tau_PSPG[i - 1];
      _solver_fem_tau_LSIC[i] = _solver_fem_tau_LSIC[i - 1];

      _solver_snes_dirichlet_conditions_new[i] =
          _solver_snes_dirichlet_conditions_new[i - 1];

      _time_scheme_acceleration[i] = _time_scheme_acceleration[i - 1];
      _time_scheme_stabilisation[i] = _time_scheme_stabilisation[i - 1];
      _time_scheme_nonlinear_term[i] = _time_scheme_nonlinear_term[i - 1];
    } else {
      if (i == 0) {
        tag = "Base flow";
      } else if (i == 1) {
        tag = "Perturbations";
      }

      _solver_fem_tau_SUPG[i] =
          _ipnut_json["Solver"][tag]["Finite element method"]["Stabilisation"]
                     ["SUPG"]["Coefficient"];
      _solver_fem_tau_PSPG[i] =
          _ipnut_json["Solver"][tag]["Finite element method"]["Stabilisation"]
                     ["PSPG"]["Coefficient"];
      _solver_fem_tau_LSIC[i] =
          _ipnut_json["Solver"][tag]["Finite element method"]["Stabilisation"]
                     ["LSIC"]["Coefficient"];

      _solver_snes_dirichlet_conditions_new[i] =
          _ipnut_json["Solver"][tag]["SNES"]["J matrix"]
                     ["Dirichlet conditions"] == "explicit"
              ? PETSC_TRUE
              : PETSC_FALSE;

      // Time Scheme
      // Function { Explicit, FunctionExtrapolation, ArgumentExtrapolation,
      // Implicit }; enum TimeSchemeAcceleration { BDF2, Euler };
      if (_ipnut_json["Solver"][tag].contains("Time scheme")) {
        if (_ipnut_json["Solver"][tag]["Time scheme"]["Time derivative"] ==
            "BDF2")
          _time_scheme_acceleration[i] = BDF2;
        else
          _time_scheme_acceleration[i] = Euler;

        if (_ipnut_json["Solver"][tag]["Time scheme"]["Stabilisation"] ==
            "explicit")
          _time_scheme_stabilisation[i] = Explicit;
        else if (_ipnut_json["Solver"][tag]["Time scheme"]["Stabilisation"] ==
                 "function extrapolation")
          _time_scheme_stabilisation[i] = FunctionExtrapolation;
        else if (_ipnut_json["Solver"][tag]["Time scheme"]["Stabilisation"] ==
                 "argument extrapolation")
          _time_scheme_stabilisation[i] = ArgumentExtrapolation;
        else if (_ipnut_json["Solver"][tag]["Time scheme"]["Stabilisation"] ==
                 "implicit")
          _time_scheme_stabilisation[i] = Implicit;

        if (_ipnut_json["Solver"][tag]["Time scheme"]["Nonlinear term"] ==
            "explicit")
          _time_scheme_nonlinear_term[i] = Explicit;
        else if (_ipnut_json["Solver"][tag]["Time scheme"]["Nonlinear term"] ==
                 "function extrapolation")
          _time_scheme_nonlinear_term[i] = FunctionExtrapolation;
        else if (_ipnut_json["Solver"][tag]["Time scheme"]["Nonlinear term"] ==
                 "argument extrapolation")
          _time_scheme_nonlinear_term[i] = ArgumentExtrapolation;
        else if (_ipnut_json["Solver"][tag]["Time scheme"]["Nonlinear term"] ==
                 "implicit")
          _time_scheme_nonlinear_term[i] = Implicit;
      } else {
        // Default
        _time_scheme_acceleration[i] = BDF2;

        // Nonlinear term
        if (_problem_type == "DNS" || i == 0)
          _time_scheme_nonlinear_term[i] = FunctionExtrapolation;
        else
          _time_scheme_nonlinear_term[i] = Implicit;

        // Stabilisation term
        if (i == 0)
          _time_scheme_stabilisation[i] = ArgumentExtrapolation;
        else
          _time_scheme_stabilisation[i] = Implicit;
      }
    }
  }

  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  // CHECKS
  for (PetscInt i = 0; i < _FModes; i++) {
    // Nonlinear term
    if ((i != 0 && _problem_type != "DNS" &&
         _time_scheme_nonlinear_term[i] != Implicit) ||
        (_problem_type == "DNS" &&
         _time_scheme_nonlinear_term[i] == Implicit)) {
      printf(
          "\n!!!!!!!!!!! Selected time scheme for nonlinear term is not "
          "supported !!!!!!!!!!!\n");
      printf(
          "\n!!!!!!!!!!! Parameter changed to default (Function Extrapolation "
          "- DNS (or i==0) | Implicit - otherwise) !!!!!!!!!!!\n");
      if (_problem_type == "DNS")
        _time_scheme_nonlinear_term[i] = FunctionExtrapolation;
      else
        _time_scheme_nonlinear_term[i] = Implicit;
    }

    // Stabilisation term
    if ((_time_scheme_stabilisation[i] == FunctionExtrapolation) ||
        (_time_scheme_stabilisation[i] == Implicit && i == 0)) {
      printf(
          "\n!!!!!!!!!!! 'Function extrapolation' time scheme for "
          "stabilisation terms is not supported !!!!!!!!!!!\n");
      printf(
          "\n!!!!!!!!!!! Parameter changed to default for base flow (Argument "
          "Extrapolation - base flow | Implicit - otherwise) !!!!!!!!!!!\n");
      if (i == 0)
        _time_scheme_stabilisation[i] = ArgumentExtrapolation;
      else
        _time_scheme_stabilisation[i] = Implicit;
    }
  }
  _time_scheme_nonlinear_term_extrapolation_f = PETSC_FALSE;
  _time_scheme_nonlinear_term_extrapolation_a = PETSC_FALSE;
  for (PetscInt i = 0; i < _FModes; i++) {
    if (_time_scheme_nonlinear_term[i] == FunctionExtrapolation) {
      _time_scheme_nonlinear_term_extrapolation_f = PETSC_TRUE;
      break;
    }
    if (_time_scheme_nonlinear_term[i] == ArgumentExtrapolation) {
      _time_scheme_nonlinear_term_extrapolation_a = PETSC_TRUE;
      break;
    }
  }
  if (_time_scheme_nonlinear_term_extrapolation_f == PETSC_TRUE &&
      _time_scheme_nonlinear_term_extrapolation_a == PETSC_TRUE) {
    printf(
        "\n!!!!!!!!!!! Simultaneous function and argument extrapolation for "
        "nonlinear term is not supported !!!!!!!!!!!\n");
    printf(
        "\n!!!!!!!!!!! Argument extrapolation changed to function "
        "extrapolation !!!!!!!!!!!\n");
    _time_scheme_nonlinear_term_extrapolation_a = PETSC_FALSE;
    for (PetscInt i = 0; i < _FModes; i++)
      if (_time_scheme_nonlinear_term[i] == ArgumentExtrapolation)
        _time_scheme_nonlinear_term[i] = FunctionExtrapolation;
  } /**/
  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  _solver_eigenproblem_KS_size =
      _ipnut_json["Solver"]["Eigenvalue problem"]["Size of Krylov subspace"];
  if (_ipnut_json["Solver"]["Eigenvalue problem"].contains(
          "Power method iterations"))
    _solver_eigenproblem_power_method_its =
        _ipnut_json["Solver"]["Eigenvalue problem"]["Power method iterations"];
  else
    _solver_eigenproblem_power_method_its = 0;

  _solver_fem_scale_pressure_type =
      _ipnut_json["Solver"]["Scales"]["Pressure"]["Type"];
  _solver_fem_scale_pressure =
      _ipnut_json["Solver"]["Scales"]["Pressure"]["Value"];

  _solver_fem_scale_mass_type =
      _ipnut_json["Solver"]["Scales"]["Mass conservation"]["Type"];
  _solver_fem_scale_mass =
      _ipnut_json["Solver"]["Scales"]["Mass conservation"]["Value"];

  _solver_fem_scale_momentum_type =
      _ipnut_json["Solver"]["Scales"]["Momentum conservation"]["Type"];
  _solver_fem_scale_momentum =
      _ipnut_json["Solver"]["Scales"]["Momentum conservation"]["Value"];

  _solver_rescale_enabled = PETSC_FALSE;
  _solver_rescale_threshold = 0.0;
  if (_ipnut_json["Solver"].contains("Rescale")) {
    if (_ipnut_json["Solver"]["Rescale"].contains("Threshold")) {
      _solver_rescale_enabled = PETSC_TRUE;
      _solver_rescale_threshold = _ipnut_json["Solver"]["Rescale"]["Threshold"];
    }
  }

  // Check
  if (_problem_type != "LSA" && _problem_type != "Floquet" &&
      _solver_rescale_enabled == PETSC_TRUE)
    printf(
        "\n!!!!!!!!!!! Rescale works only with LSA or Floquet problem types "
        "!!!!!!!!!!!\n");

  // --------------------------------------------
  // Debug
  if (_rank == 0) {
    printf("    | Debug...\n");
  }

  if (_ipnut_json.contains("Debug")) {
    _debug_enabled = ConvertBool(_ipnut_json["Debug"]["Enabled"]);
    _debug_thread = _ipnut_json["Debug"]["Monitor"]["Thread number"];
    _debug_jacobi_save = ConvertBool(_ipnut_json["Debug"]["J matrix"]["Save"]);
    _debug_tsa_KS_iterations_save =
        ConvertBool(_ipnut_json["Debug"]["TSA"]["Save KS-iterations"]);
    if (_ipnut_json["Debug"].contains("Test"))
      _debug_test = _ipnut_json["Debug"]["Test"];
    else
      _debug_test = 0;
  } else {
    _debug_enabled = PETSC_FALSE;
    _debug_thread = -2;
    _debug_jacobi_save = PETSC_FALSE;
    _debug_tsa_KS_iterations_save = PETSC_FALSE;
    _debug_test = 0;
  }

  // ============================================
  // Create output folders
  DIR *dir;
  char aux_path[1024];
  char aux_path1[1024];
  if (_FModes == 1)
    sprintf(aux_path, "%s/output_%s_2D/", _ProjectDir, _problem_type.c_str());
  else
    sprintf(aux_path, "%s/output_%s_3D/", _ProjectDir, _problem_type.c_str());
  if (_rank == 0) {
    mkdir(aux_path, 0755);
  }

  if (_problem_type == "DNS" || _problem_type == "LSA" ||
      _problem_type == "Floquet") {
    if (_FModes != 1)
      sprintf(aux_path1, "%s/l=%.6lf/", aux_path, _L);
    else
      sprintf(aux_path1, "%s", aux_path);
    if (_rank == 0) {
      mkdir(aux_path1, 0755);
    }

    for (PetscInt n = 0; n < 1000000; n++) {
      sprintf(_Output, "%s/run-%ld/", aux_path1, n);
      dir = opendir(_Output);
      if (dir)
        closedir(dir);
      else
        break;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (_rank == 0) {
      mkdir(_Output, 0755);
    }

    // Create folder for every Fourier mode
    for (PetscInt i = 0; i < _FModes; i++) {
      sprintf(aux_path, "%s/mode-%ld/", _Output, i);
      if (_rank == 0) {
        mkdir(aux_path, 0755);
      }
    }
  } else {
    if (_FModes != 1)
      sprintf(aux_path1, "%s/l=%.6lf_t0=%.6lf_t1=%.6lf_ks=%ld/", aux_path, _L,
              _T_start, _T_end, _solver_eigenproblem_KS_size);
    else
      sprintf(aux_path1, "%s/t0=%.6lf_t1=%.6lf_ks=%ld/", aux_path, _T_start,
              _T_end, _solver_eigenproblem_KS_size);
    if (_rank == 0) {
      mkdir(aux_path1, 0755);
    }

    for (PetscInt n = 0; n < 1000000; n++) {
      sprintf(_Output, "%s/run-%ld/", aux_path1, n);

      dir = opendir(_Output);
      if (dir)
        closedir(dir);
      else
        break;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (_rank == 0) {
      mkdir(_Output, 0755);
    }

    // Create folder for every Fourier mode
    for (PetscInt i = 0; i < _FModes; i++) {
      sprintf(aux_path, "%s/mode-%ld/", _Output, i);
      if (_rank == 0) {
        mkdir(aux_path, 0755);
      }
    }

    if (_debug_tsa_KS_iterations_save == PETSC_TRUE) {
      sprintf(aux_path, "%s/KS_iterations/", _Output);
      if (_rank == 0) {
        mkdir(aux_path, 0755);
      }

      // Create folder for every Fourier mode
      for (PetscInt i = 0; i < _FModes; i++) {
        sprintf(aux_path1, "%s/mode-%ld/", aux_path, i);
        if (_rank == 0) {
          mkdir(aux_path1, 0755);
        }
      }
    }
  }

  // Copy parameters
  char in_file_path[1024];
  char out_file_path[1024];
  sprintf(in_file_path, "%s/%s", _Input, p_input_file);
  sprintf(out_file_path, "%s/%s", _Output, p_input_file);
  copyFile(in_file_path, out_file_path);

  MPI_Barrier(MPI_COMM_WORLD);
}

void Input::SetInitialDistribution(PetscInt p_var_idx, PetscInt p_mode,
                                   json p_json) {
  if (p_json["Type"] == "Constant") {
    PetscReal real_part = p_json["Value"][0];
    PetscReal imaginary_part = p_json["Value"][1];
    for (PetscInt vtx = 0; vtx < _Mesh._NVertex; vtx++)
      _ic_solution[NVAR * _Mesh._NVertex * p_mode + NVAR * vtx + p_var_idx] =
          real_part + PETSC_i * imaginary_part;
  } else if (p_json["Type"] == "Zero") {
    for (PetscInt vtx = 0; vtx < _Mesh._NVertex; vtx++)
      _ic_solution[NVAR * _Mesh._NVertex * p_mode + NVAR * vtx + p_var_idx] =
          0.0;
  } else if (p_json["Type"] == "One") {
    for (PetscInt vtx = 0; vtx < _Mesh._NVertex; vtx++)
      _ic_solution[NVAR * _Mesh._NVertex * p_mode + NVAR * vtx + p_var_idx] =
          1.0;
  }
}

void Input::InitPeriodicBaseFlow(PetscInt plN, PetscInt *pGlobalIdx) {
  int rank, size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) printf("InitPeriodicBaseFlow-Start\n");

  // ---------------------------------
  // Allocate memory

  PetscReal *_ps_Solution;  // Solution in the physical space

  PetscMalloc(plN * _base_flow_times_count * sizeof(PetscReal),
              &(_ps_Solution));
  _base_flow_FS = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * plN *
                                              _base_flow_modes_count);

  // ---------------------------------
  // FFTW parameters
  int *inembed = NULL;
  int *onembed = NULL;
  int istride = 1;
  int ostride = 1;
  int n[1];
  n[0] = _base_flow_times_count;
  int howmany;

  int idist, odist;
  fftw_plan _FFTW_PlanR2C;  // FFTW

  howmany = plN;
  idist = _base_flow_times_count;
  odist = _base_flow_modes_count;
  _FFTW_PlanR2C = fftw_plan_many_dft_r2c(1, n, howmany, (double *)_ps_Solution,
                                         NULL, 1, idist, _base_flow_FS, NULL, 1,
                                         odist, FFTW_ESTIMATE);

  // ---------------------------------
  // Read solution
  FILE *fRead;
  char file_path[1024];

  // Read LocalIdx
  sprintf(file_path, "%s/%s/LocalIdx.dat", _ProjectDir,
          _base_flow_folder.c_str());
  fRead = fopen(file_path, "rb");
  if (!fRead) {
    printf("Could not open input file: %s\n", file_path);
    PressEnterToContinue();
  }

  PetscInt *LocalIdx;
  PetscMalloc(_Mesh._NVertex * sizeof(PetscInt), &LocalIdx);
  fread(LocalIdx, sizeof(PetscInt), _Mesh._NVertex, fRead);
  fclose(fRead);

  // -------------------------------------------------------------
  // Read GlobalIdx
  sprintf(file_path, "%s/%s/GlobalIdx.dat", _ProjectDir,
          _base_flow_folder.c_str());
  fRead = fopen(file_path, "rb");
  if (!fRead) {
    printf("Could not open input file: %s\n", file_path);
    PressEnterToContinue();
  }

  // CHECK
  fseek(fRead, 0L, SEEK_END);
  PetscInt FileSize = ftell(fRead);
  fseek(fRead, 0L, SEEK_SET);
  if (_Mesh._NVertex * sizeof(PetscInt) != FileSize) {
    printf("ERROR: size of GlobalIdx.dat (%ld) != Nv*sizeof(PetscInt) (%ld)\n",
           FileSize, _Mesh._NVertex * sizeof(PetscInt));
    PressEnterToContinue();
  }

  // Read
  PetscInt *GlobalIdx;
  PetscMalloc(_Mesh._NVertex * sizeof(PetscInt), &GlobalIdx);
  fread(GlobalIdx, sizeof(PetscInt), _Mesh._NVertex, fRead);
  fclose(fRead);

  // -------------------------------------------------------------
  // Create Local to Global Map
  PetscInt *LocalToGlobal;
  PetscMalloc(_Mesh._NVertex * sizeof(PetscInt), &LocalToGlobal);
  for (PetscInt i = 0; i < _Mesh._NVertex; i++)
    LocalToGlobal[LocalIdx[i]] = GlobalIdx[i];
  // -------------------------------------------------------------
  PetscReal *bufSolution, *bufSolution1;
  PetscMalloc(_Mesh._NVertex * NVAR * sizeof(PetscReal), &bufSolution);
  PetscMalloc(_Mesh._NVertex * NVAR * sizeof(PetscReal), &bufSolution1);

  for (PetscInt t = 0; t < _base_flow_times_count; t++) {
    sprintf(file_path, "%s/%s/mode-0/Solution-%s.dat", _ProjectDir,
            _base_flow_folder.c_str(), _base_flow_times[t].c_str());

    fRead = fopen(file_path, "rb");
    if (!fRead) {
      printf("Could not open input file: %s\n", file_path);
      PressEnterToContinue();
    }
    fread(bufSolution, sizeof(PetscReal), _Mesh._NVertex * NVAR, fRead);
    fclose(fRead);

    for (PetscInt i = 0; i < _Mesh._NVertex * NVAR; i++) {
      bufSolution1[NVAR * LocalToGlobal[i / NVAR] + i % NVAR] = bufSolution[i];
    }

    // Distribution over proc
    for (PetscInt i = 0; i < plN; i++)
      _ps_Solution[_base_flow_times_count * i + t] =
          bufSolution1[NVAR * pGlobalIdx[i / NVAR] + i % NVAR];
  }
  PetscFree(bufSolution);
  PetscFree(bufSolution1);
  PetscFree(GlobalIdx);
  PetscFree(LocalIdx);
  PetscFree(LocalToGlobal);

  // ---------------------------------
  // Fourier transform

  fftw_execute(_FFTW_PlanR2C);

  // normalisation
  for (PetscInt i = 0; i < plN * _base_flow_modes_count; i++) {
    _base_flow_FS[i][0] = _base_flow_FS[i][0] / _base_flow_times_count;
    _base_flow_FS[i][1] = _base_flow_FS[i][1] / _base_flow_times_count;

    if (!isfinite(_base_flow_FS[i][0]) || !isfinite(_base_flow_FS[i][1])) {
      printf("NAN: _fs_BaseFlow[%ld]\n", i);
      PressEnterToContinue();
    }
  }

  // ---------------------------------
  // Free memory
  PetscFree(_ps_Solution);
  fftw_destroy_plan(_FFTW_PlanR2C);

  if (rank == 0) printf("InitPeriodicBaseFlow-End\n");
}

void Input::FinalizePeriodicBaseFlow() { fftw_free(_base_flow_FS); }

PetscBool Input::ConvertBool(bool p_bool) {
  return p_bool == true ? PETSC_TRUE : PETSC_FALSE;
}

PetscScalar Input::GetXInitial(PetscInt pGlobalVrtxIdx, PetscInt pVarIdx,
                               PetscReal pX, PetscReal pY, PetscInt pModeIdx) {
  return _ic_solution[_Mesh._NVertex * NVAR * pModeIdx + pGlobalVrtxIdx * NVAR +
                      pVarIdx];
}

BCType Input::GetBCType(PetscInt pGlobalVrtxIdx, PetscInt pVarIdx,
                        PetscInt pModeIdx) {
  // Pressure constant fixation
  if (pVarIdx == 0 && pGlobalVrtxIdx == _pressure_constant_node &&
      pModeIdx == 0)
    return Dirichlet;

  if (_BorderIdxByVrtx[pGlobalVrtxIdx] == -1)
    return None;
  else if (_BorderIdxByVrtx2[pGlobalVrtxIdx] == -1) {
    return _BC[pModeIdx]
        ._BCType[_BorderIdxByVrtx[pGlobalVrtxIdx] * NVAR + pVarIdx];
  } else {
    if (_BC[pModeIdx]._BCType[_BorderIdxByVrtx[pGlobalVrtxIdx] * NVAR +
                              pVarIdx] == Dirichlet ||
        _BC[pModeIdx]._BCType[_BorderIdxByVrtx2[pGlobalVrtxIdx] * NVAR +
                              pVarIdx] == Dirichlet)
      return Dirichlet;
    else
      return _BC[pModeIdx]
          ._BCType[_BorderIdxByVrtx[pGlobalVrtxIdx] * NVAR + pVarIdx];
  }
}

PetscInt Input::GetBorderIdx(PetscInt pGlobalEdgeIdx) {
  return _BorderIdxByEdge[pGlobalEdgeIdx];
}

PetscInt Input::GetBorderIdxByVtx(PetscInt pGlobalVtxIdx) {
  return _BorderIdxByVrtx[pGlobalVtxIdx];
}

PetscScalar Input::GetBCValue(PetscInt pGlobalVrtxIdx, PetscInt pVarIdx,
                              PetscReal pTime, PetscReal pX, PetscReal pY,
                              PetscInt pModeIdx) {
  // Pressure constant fixation
  if (pVarIdx == 0 && pGlobalVrtxIdx == _pressure_constant_node &&
      pModeIdx == 0)
    return _pressure_constant_value;

  PetscInt bIdxByVrtx = _BorderIdxByVrtx[pGlobalVrtxIdx];
  if (_BC[pModeIdx]._BCType[_BorderIdxByVrtx[pGlobalVrtxIdx] * NVAR +
                            pVarIdx] != Dirichlet &&
      _BC[pModeIdx]._BCType[_BorderIdxByVrtx2[pGlobalVrtxIdx] * NVAR +
                            pVarIdx] == Dirichlet)
    bIdxByVrtx = _BorderIdxByVrtx2[pGlobalVrtxIdx];

  if (bIdxByVrtx == -1) {
    printf(
        "Error in GetBCValue: getting boundary conditions in inner vertex\n");
    PressEnterToContinue();
    return 0.0;
  } else if (_BC[pModeIdx]._BCValueType[bIdxByVrtx * NVAR + pVarIdx] ==
             Function) {
    PetscInt pIdx = bIdxByVrtx * NVAR + pVarIdx;
    return GetLinearInterpolation(_BC[pModeIdx]._fNValues[pIdx],
                                  _BC[pModeIdx]._fValuesT[pIdx],
                                  _BC[pModeIdx]._fValuesF[pIdx], pTime);
  } else if (_BC[pModeIdx]._BCValueType[bIdxByVrtx * NVAR + pVarIdx] ==
             Oscillations) {
    PetscInt pIdx = bIdxByVrtx * NVAR + pVarIdx;
    return _BC[pModeIdx]._OscA0[pIdx] +
           _BC[pModeIdx]._OscA1[pIdx] *
               cos(2 * PI *
                   (_BC[pModeIdx]._OscF0[pIdx] +
                    _BC[pModeIdx]._OscF1[pIdx] * pTime));
  } else if (_BC[pModeIdx]._BCValueType[bIdxByVrtx * NVAR + pVarIdx] ==
             Polynomial) {
    PetscInt pIdx = bIdxByVrtx * NVAR + pVarIdx;
    PetscReal x_polinomial = 0.0;
    PetscReal y_polinomial = 0.0;
    PetscReal power = 1.0;
    for (PetscInt i = _BC[pModeIdx]._polinomial_x_degree[pIdx]; i >= 0; i--) {
      x_polinomial += _BC[pModeIdx]._polinomial_x_coefficient[pIdx][i] * power;
      power *= pX;
    }

    power = 1.0;
    for (PetscInt i = _BC[pModeIdx]._polinomial_y_degree[pIdx]; i >= 0; i--) {
      y_polinomial += _BC[pModeIdx]._polinomial_y_coefficient[pIdx][i] * power;
      power *= pY;
    }

    return x_polinomial + y_polinomial;
  } else
    return _BC[pModeIdx]._Value[bIdxByVrtx * NVAR + pVarIdx];
}

PetscScalar Input::GetForceValue(PetscInt pIdx, PetscReal pTime, PetscReal pX,
                                 PetscReal pY, PetscInt pModeIdx) {
  if (_ForceType == 'O')
    return _ForceA0[pIdx] -
           2 * PI * _ForceF1[pIdx] * _ForceA1[pIdx] *
               sin(2 * PI * (_ForceF0[pIdx] + _ForceF1[pIdx] * pTime));
  else
    return GetLinearInterpolation(_ForceNT, _ForceT, _ForceF[pIdx], pTime);
}

PetscScalar Input::GetLinearInterpolation(PetscInt pN, PetscReal *pT,
                                          PetscScalar *pF, PetscReal pTime) {
  if (pTime <= pT[0]) return pF[0];
  if (pTime >= pT[pN - 1]) return pF[pN - 1];

  for (PetscInt t = 0; t < pN - 1; t++) {
    if (pTime >= pT[t] && pTime <= pT[t + 1]) {
      return pF[t] +
             (pTime - pT[t]) * (pF[t + 1] - pF[t]) / (pT[t + 1] - pT[t]);
    }
  }

  // В случае плохих данных
  printf("Error in GetLinearInterpolation: pTime=%lf\n", pTime);
  PressEnterToContinue();
  return 0.0;
}

PetscReal Input::GetLinearInterpolation(PetscInt pN, PetscReal *pT,
                                        PetscReal *pF, PetscReal pTime) {
  if (pTime <= pT[0]) return pF[0];
  if (pTime >= pT[pN - 1]) return pF[pN - 1];

  for (PetscInt t = 0; t < pN - 1; t++) {
    if (pTime >= pT[t] && pTime <= pT[t + 1]) {
      return pF[t] +
             (pTime - pT[t]) * (pF[t + 1] - pF[t]) / (pT[t + 1] - pT[t]);
    }
  }

  // В случае плохих данных
  printf("Error in GetLinearInterpolation: pTime=%lf\n", pTime);
  PressEnterToContinue();
  return 0.0;
}

void Input::Check_Input_Solver() {
  if (_ipnut_json["Solver"].contains("Base flow") == false) {
    if (_rank == 0) {
      printf(
          "ERROR [input]: 'Solver'->'Base flow' should be defined for 3D "
          "problem\n");
      PressEnterToContinue();
    }
    MPI_Barrier(MPI_COMM_WORLD);
  } else if (_FModes > 1 &&
             _ipnut_json["Solver"].contains("Perturbations") == false) {
    if (_rank == 0) {
      printf(
          "ERROR [input]: 'Solver'->'Perturbations' should be defined for 3D "
          "problem\n");
      PressEnterToContinue();
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
}
