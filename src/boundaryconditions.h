#ifndef BOUNDARYCONDITIONS_H
#define BOUNDARYCONDITIONS_H

#include <iostream>

#include "json.hpp"
#include "structures.h"

using namespace std;
using json = nlohmann::json;

namespace O3D {

enum BCType { None, Dirichlet, TractionFree, PseudoTractionFree };
enum BCValueType { Value, Function, Oscillations, Polynomial };

class BoundaryConditions {
 public:
  PetscInt _NBoundaries;
  BCType *_BCType;
  BCValueType *_BCValueType;
  PetscScalar *_Value;
  PetscScalar *_OscA0;
  PetscScalar *_OscA1;
  PetscReal *_OscF0;
  PetscReal *_OscF1;
  PetscInt *_fNValues;
  PetscReal **_fValuesT;
  PetscScalar **_fValuesF;

  // Polinomial
  PetscInt *_polinomial_x_degree;
  PetscInt *_polinomial_y_degree;
  PetscReal **_polinomial_x_coefficient;
  PetscReal **_polinomial_y_coefficient;

 public:
  BoundaryConditions();
  virtual ~BoundaryConditions();

  void Initialize(json p_json_input, string p_tag, char *p_project_folder);

 private:
  void SetBoundaryConditions(json p_json_input, PetscInt p_boundary_idx,
                             PetscInt p_var_idx, char *p_project_folder);
};

}  // namespace O3D

#endif  // BOUNDARYCONDITIONS_H
