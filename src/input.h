#ifndef INPUT_H
#define INPUT_H

#include <fftw3.h>
#include <petsctime.h>

#include <iostream>

#include "boundaryconditions.h"
#include "json.hpp"
#include "mesh.h"
#include "region.h"
#include "structures.h"

using namespace std;
using json = nlohmann::json;

namespace O3D {

enum TimeSchemeFunction {
  Explicit,
  FunctionExtrapolation,
  ArgumentExtrapolation,
  Implicit
};

enum TimeSchemeAcceleration { BDF2, Euler };

class Input {
 public:
  PetscMPIInt _rank;

  char _Input[1024], _Output[1024];
  char *_ProjectDir;

  json _ipnut_json;
  // =================================================
  // From file
  PetscScalar Version;

  // --------------------------------------------
  // Problem formulation
  string _problem_type;
  PetscReal _Re;  // Reynolds number
  PetscReal _Fr;  // Froud number

  // Force
  char _ForceType;  // C - constant, F - file, O - oscillation
  PetscInt _ForceNT;
  PetscReal *_ForceT;
  PetscReal **_ForceF;
  PetscReal *_ForceA0, *_ForceA1, *_ForceF0, *_ForceF1;

  // Time
  PetscReal _T_start;
  PetscReal _T_end;
  PetscReal _dT;  // time step
  PetscInt _NT;   // amount of time steps

  // Space
  string _SpaceType;
  PetscReal _L;         // z-length of domain (for plane geometry)
  PetscInt _FModes;     // amount of Fourier modes in z direction ('1' for 2D
                        // simulations)
  char _MeshFile[524];  // mesh
  Mesh _Mesh;

  // Boundary conditions
  PetscInt _boundaries_count;
  BoundaryConditions *_BC;
  PetscInt *_BoundaryNo;
  PetscInt *_BorderIdxByVrtx;  // -1 - inner node
  PetscInt *_BorderIdxByVrtx2;
  PetscInt *_BorderIdxByEdge;

  // Initial conditions
  PetscScalar *_ic_solution;
  Region *_ic_region;

  // Base flow
  PetscInt weak_form_type;  // 0 - traction free; 1 - pseudo traction free

  // Base flow
  string _base_flow_type;
  string _base_flow_folder;
  string _base_flow_times_file;
  PetscInt _base_flow_times_count;
  string *_base_flow_times;
  PetscInt _base_flow_modes_count;
  PetscReal _base_flow_period;
  fftw_complex *_base_flow_FS;  // Solution in the Fourier space
  PetscBool _jacobi_matrix_interpolation;

  // Pressure constant
  PetscBool _pressure_constant_fixed;
  PetscInt _pressure_constant_node;
  PetscReal _pressure_constant_value;

  // --------------------------------------------
  // Output
  PetscInt _output_monitor_period;   // period of monitoring data printing
  PetscInt _output_solution_period;  // period of results saving
  PetscInt _output_solution_start;   // initial iteration for saving results

  PetscInt _output_force_count;   // amount of boundaries for force calculations
  PetscInt *_output_force_IDs;    // their physical numbers
  PetscInt _output_force_period;  // period of forces saving
  PetscInt _output_energy_period;
  PetscInt _output_eigenvalues_monitor;
  PetscBool _output_efficiency_enabled;

  PetscInt _output_probe_period;
  PetscInt _output_probe_count;
  PetscReal *_output_probe_x;
  PetscReal *_output_probe_y;
  PetscInt _output_probe_count_local;
  PetscInt *_output_probe_idx_local;
  PetscReal *_output_probe_x_local;
  PetscReal *_output_probe_y_local;

  // --------------------------------------------
  // Solver
  PetscInt
      _solver_count;  // max 3 solvers: base flow, perturbations, and adjoint
  PetscReal *_solver_fem_tau_SUPG;  // variant of tau_SUPG
  PetscReal *_solver_fem_tau_PSPG;  // variant of tau_PSPG
  PetscReal *_solver_fem_tau_LSIC;  // variant of tau_LSIC
  PetscBool
      *_solver_snes_dirichlet_conditions_new;  // Dirichlet condition in Jacobi
                                               // matrix: <0> - old algorithm;
                                               // <1> - new algorithm
  PetscBool *_solver_snes_near_null_space;     // Setup nullspace for Jacobi
                                               // matrix: <0> - no; <1> - yes
  string *_solver_snes_type;                   //

  PetscInt _solver_eigenproblem_KS_size;
  PetscInt _solver_eigenproblem_power_method_its;

  string _solver_fem_scale_pressure_type;
  string _solver_fem_scale_mass_type;
  string _solver_fem_scale_momentum_type;
  PetscReal _solver_fem_scale_pressure;  // Pressure scale: <0> - no (1); <k> -
                                         // k/tau_PSPG; <-1> - coupled scaling
  PetscReal
      _solver_fem_scale_mass;  // Scale of mass conservation equation: <0> - no
                               // (1); <k> - k/tau_PSPG; <-1> - coupled scaling
  PetscReal
      _solver_fem_scale_momentum;  // Scale of momentum conservation equation:
                                   // <0> - no (1); <k> - k(dT/h^2)

  PetscBool _solver_rescale_enabled;
  PetscReal _solver_rescale_threshold;

  TimeSchemeFunction *_time_scheme_stabilisation;
  TimeSchemeFunction *_time_scheme_nonlinear_term;
  TimeSchemeAcceleration *_time_scheme_acceleration;
  PetscBool _time_scheme_nonlinear_term_extrapolation_f;
  PetscBool _time_scheme_nonlinear_term_extrapolation_a;

  // --------------------------------------------
  // Debug
  PetscBool _debug_enabled;
  PetscInt _debug_thread;
  PetscInt _debug_test;
  PetscBool _debug_jacobi_save;
  PetscBool _debug_tsa_KS_iterations_save;

  // =================================================

 public:
  Input();
  virtual ~Input();
  void Initialize(char *p_profect_folder, char *p_input_file);
  void InitPeriodicBaseFlow(PetscInt plN, PetscInt *pGlobalIdx);
  void FinalizePeriodicBaseFlow();

  PetscBool ConvertBool(bool p_bool);

  void SetInitialDistribution(PetscInt p_var_idx, PetscInt p_mode, json p_json);

  PetscScalar GetXInitial(PetscInt pGlobalVrtxIdx, PetscInt pVarIdx,
                          PetscReal pX, PetscReal pY, PetscInt pModeIdx);
  PetscInt GetBorderIdx(PetscInt pGlobalEdgeIdx);
  PetscInt GetBorderIdxByVtx(PetscInt pGlobalEdgeIdx);
  BCType GetBCType(PetscInt pGlobalVrtxIdx, PetscInt pVarIdx,
                   PetscInt pModeIdx);
  PetscScalar GetBCValue(PetscInt pGlobalVrtxIdx, PetscInt pVarIdx,
                         PetscReal pTime, PetscReal pX, PetscReal pY,
                         PetscInt pModeIdx);

  PetscScalar GetForceValue(PetscInt pIdx, PetscReal pTime, PetscReal pX,
                            PetscReal pY, PetscInt pModeIdx);

  PetscScalar GetLinearInterpolation(PetscInt pN, PetscReal *pT,
                                     PetscScalar *pF, PetscReal pTime);
  PetscReal GetLinearInterpolation(PetscInt pN, PetscReal *pT, PetscReal *pF,
                                   PetscReal pTime);

 private:
  // CHECK: input data
  void Check_Input_Solver();
};
}  // namespace O3D
#endif  // INPUT_H
