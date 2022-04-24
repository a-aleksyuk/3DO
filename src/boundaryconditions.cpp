#include "boundaryconditions.h"

using namespace O3D;

BoundaryConditions::BoundaryConditions() {}
BoundaryConditions::~BoundaryConditions() {
  PetscFree(_BCType);
  PetscFree(_BCValueType);
  PetscFree(_Value);
  PetscFree(_OscA0);
  PetscFree(_OscA1);
  PetscFree(_OscF0);
  PetscFree(_OscF1);

  for (PetscInt i = 0; i < _NBoundaries * NVAR; i++) {
    if (_fNValues[i] != -1) {
      PetscFree(_fValuesT[i]);
      PetscFree(_fValuesF[i]);
    }

    if (_polinomial_x_degree[i] != -1) {
      PetscFree(_polinomial_x_coefficient[i]);
    }

    if (_polinomial_y_degree[i] != -1) {
      PetscFree(_polinomial_y_coefficient[i]);
    }
  }
  PetscFree(_fNValues);
  PetscFree(_fValuesT);
  PetscFree(_fValuesF);

  PetscFree(_polinomial_x_degree);
  PetscFree(_polinomial_y_degree);
  PetscFree(_polinomial_x_coefficient);
  PetscFree(_polinomial_y_coefficient);
}

void BoundaryConditions::Initialize(json p_json_input, string p_tag,
                                    char *p_project_folder) {
  _NBoundaries = p_json_input["Count"];

  // Allocate memory
  PetscMalloc(_NBoundaries * NVAR * sizeof(BCType), &(_BCType));
  PetscMalloc(_NBoundaries * NVAR * sizeof(BCValueType), &(_BCValueType));
  PetscMalloc(_NBoundaries * NVAR * sizeof(PetscScalar), &(_Value));
  PetscMalloc(_NBoundaries * NVAR * sizeof(PetscScalar), &(_OscA0));
  PetscMalloc(_NBoundaries * NVAR * sizeof(PetscScalar), &(_OscA1));
  PetscMalloc(_NBoundaries * NVAR * sizeof(PetscReal), &(_OscF0));
  PetscMalloc(_NBoundaries * NVAR * sizeof(PetscReal), &(_OscF1));
  PetscMalloc(_NBoundaries * NVAR * sizeof(PetscInt), &(_fNValues));
  PetscMalloc(_NBoundaries * NVAR * sizeof(PetscReal *), &(_fValuesT));
  PetscMalloc(_NBoundaries * NVAR * sizeof(PetscScalar *), &(_fValuesF));

  PetscMalloc(_NBoundaries * NVAR * sizeof(PetscInt), &(_polinomial_x_degree));
  PetscMalloc(_NBoundaries * NVAR * sizeof(PetscInt), &(_polinomial_y_degree));
  PetscMalloc(_NBoundaries * NVAR * sizeof(PetscReal *),
              &(_polinomial_x_coefficient));
  PetscMalloc(_NBoundaries * NVAR * sizeof(PetscReal *),
              &(_polinomial_y_coefficient));

  for (PetscInt i = 0; i < _NBoundaries * NVAR; i++) {
    _fNValues[i] = -1;
    _polinomial_x_degree[i] = -1;
    _polinomial_y_degree[i] = -1;
  }

  for (PetscInt bIdx = 0; bIdx < _NBoundaries; bIdx++) {
    if (p_json_input[p_tag][bIdx]["Type"] == "Custom") {
      SetBoundaryConditions(p_json_input[p_tag][bIdx]["Pressure"], bIdx, 0,
                            p_project_folder);
      SetBoundaryConditions(p_json_input[p_tag][bIdx]["Velocity_1"], bIdx, 1,
                            p_project_folder);
      SetBoundaryConditions(p_json_input[p_tag][bIdx]["Velocity_2"], bIdx, 2,
                            p_project_folder);
      SetBoundaryConditions(p_json_input[p_tag][bIdx]["Velocity_3"], bIdx, 3,
                            p_project_folder);
    } else if (p_json_input[p_tag][bIdx]["Type"] == "Zero") {
      for (PetscInt var = 0; var < NVAR; var++) {
        _BCType[bIdx * NVAR + var] = Dirichlet;
        _BCValueType[bIdx * NVAR + var] = Value;
        _Value[bIdx * NVAR + var] = 0.0;
      }
    } else if (p_json_input[p_tag][bIdx]["Type"] == "Wall") {
      for (PetscInt var = 0; var < NVAR; var++) {
        if (var == 0) {
          _BCType[bIdx * NVAR + var] = None;
        } else {
          _BCType[bIdx * NVAR + var] = Dirichlet;
          _BCValueType[bIdx * NVAR + var] = Value;
          _Value[bIdx * NVAR + var] = 0.0;
        }
      }
    } else if (p_json_input[p_tag][bIdx]["Type"] == "Traction free") {
      for (PetscInt var = 0; var < NVAR; var++) {
        if (var == 0) {
          _BCType[bIdx * NVAR + var] = None;
        } else {
          _BCType[bIdx * NVAR + var] = TractionFree;
        }
      }
    } else if (p_json_input[p_tag][bIdx]["Type"] == "Pseudo traction free") {
      for (PetscInt var = 0; var < NVAR; var++) {
        if (var == 0) {
          _BCType[bIdx * NVAR + var] = None;
        } else {
          _BCType[bIdx * NVAR + var] = PseudoTractionFree;
        }
      }
    }
  }
}

void BoundaryConditions::SetBoundaryConditions(json p_json_input,
                                               PetscInt p_boundary_idx,
                                               PetscInt p_var_idx,
                                               char *p_project_folder) {
  if (p_json_input["Type"] == "Dirichlet-Value") {
    _BCType[p_boundary_idx * NVAR + p_var_idx] = Dirichlet;
    _BCValueType[p_boundary_idx * NVAR + p_var_idx] = Value;
    PetscReal re_part = p_json_input["Value"][0];
    PetscReal im_part = p_json_input["Value"][1];
    _Value[p_boundary_idx * NVAR + p_var_idx] = re_part + PETSC_i * im_part;
  } else if (p_json_input["Type"] == "Dirichlet-Function") {
    _BCType[p_boundary_idx * NVAR + p_var_idx] = Dirichlet;
    _BCValueType[p_boundary_idx * NVAR + p_var_idx] = Function;

    char file_path[512];
    string file_name = p_json_input["File"];

    sprintf(file_path, "%s/%s", p_project_folder, file_name.c_str());

    FILE *fRead = fopen(file_path, "r");
    if (!fRead) {
      printf("Could no open file: %s\n", file_path);
      PressEnterToContinue();
    }
    fscanf(fRead, "%ld%*[^\n]\n",
           &(_fNValues[p_boundary_idx * NVAR + p_var_idx]));

    PetscReal real_part, imaginary_part;
    PetscMalloc(
        _fNValues[p_boundary_idx * NVAR + p_var_idx] * sizeof(PetscReal),
        &(_fValuesT[p_boundary_idx * NVAR + p_var_idx]));
    PetscMalloc(
        _fNValues[p_boundary_idx * NVAR + p_var_idx] * sizeof(PetscScalar),
        &(_fValuesF[p_boundary_idx * NVAR + p_var_idx]));
    for (PetscInt i = 0; i < _fNValues[p_boundary_idx * NVAR + p_var_idx];
         i++) {
      fscanf(fRead, "%lf %lf %lf%*[^\n]\n",
             &(_fValuesT[p_boundary_idx * NVAR + p_var_idx][i]), &real_part,
             &imaginary_part);
      _fValuesF[p_boundary_idx * NVAR + p_var_idx][i] =
          real_part + PETSC_i * imaginary_part;
    }
    fclose(fRead);
  } else if (p_json_input["Type"] == "Dirichlet-Oscillations") {
    _BCType[p_boundary_idx * NVAR + p_var_idx] = Dirichlet;
    _BCValueType[p_boundary_idx * NVAR + p_var_idx] = Oscillations;
    PetscReal re_part = p_json_input["A0"][0];
    PetscReal im_part = p_json_input["A0"][1];
    _OscA0[p_boundary_idx * NVAR + p_var_idx] = re_part + PETSC_i * im_part;
    re_part = p_json_input["A1"][0];
    im_part = p_json_input["A1"][1];
    _OscA1[p_boundary_idx * NVAR + p_var_idx] = re_part + PETSC_i * im_part;
    _OscF0[p_boundary_idx * NVAR + p_var_idx] = p_json_input["F0"];
    _OscF1[p_boundary_idx * NVAR + p_var_idx] = p_json_input["F1"];
  } else if (p_json_input["Type"] == "Dirichlet-Polynomial") {
    _BCType[p_boundary_idx * NVAR + p_var_idx] = Dirichlet;
    _BCValueType[p_boundary_idx * NVAR + p_var_idx] = Polynomial;

    _polinomial_x_degree[p_boundary_idx * NVAR + p_var_idx] =
        p_json_input["x-degree"];
    _polinomial_y_degree[p_boundary_idx * NVAR + p_var_idx] =
        p_json_input["y-degree"];

    PetscMalloc(
        (1 + _polinomial_x_degree[p_boundary_idx * NVAR + p_var_idx]) *
            sizeof(PetscReal),
        &(_polinomial_x_coefficient[p_boundary_idx * NVAR + p_var_idx]));
    PetscMalloc(
        (1 + _polinomial_y_degree[p_boundary_idx * NVAR + p_var_idx]) *
            sizeof(PetscReal),
        &(_polinomial_y_coefficient[p_boundary_idx * NVAR + p_var_idx]));

    for (PetscInt i = 0;
         i < 1 + _polinomial_x_degree[p_boundary_idx * NVAR + p_var_idx]; i++)
      _polinomial_x_coefficient[p_boundary_idx * NVAR + p_var_idx][i] =
          p_json_input["x-coefficients"][i];

    for (PetscInt i = 0;
         i < 1 + _polinomial_y_degree[p_boundary_idx * NVAR + p_var_idx]; i++)
      _polinomial_y_coefficient[p_boundary_idx * NVAR + p_var_idx][i] =
          p_json_input["y-coefficients"][i];
  } else {
    _BCType[p_boundary_idx * NVAR + p_var_idx] = None;
  }
}
