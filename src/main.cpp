#include <stdlib.h>

#include <ctime>
#include <iostream>

#include "fftw3.h"
#include "input.h"
#include "meshpartition.h"
#include "navierstokes.h"
#include "solver.h"
#include "structures.h"

using namespace std;
using namespace O3D;

PetscErrorCode Run(char *project_folder, char *input_file) {
  // Get process rank
  PetscMPIInt rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Initialize input
  Input input;
  input.Initialize(project_folder, input_file);

  // Initialize monitor
  Monitor monitor;
  monitor.Initialize(&input);

  // Create mesh partition
  MeshPartition mesh_partition(MPI_COMM_WORLD, &input._Mesh);
  mesh_partition.CreatePartition();

  // Create solver
  NavierStokes equations(&input, &monitor);
  Solver solver(&input, &monitor, &equations, &mesh_partition);
  solver.Create();

  // Solve equations
  solver.Solve();

  return 0;
}

int main(int argc, char **argv) {
  // Get process rank and number of processes
  PetscMPIInt rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Initialize PETSc
  if (argc < 3) {
    if (rank == 0) printf("Project folder and input file are not defined.\n");
    return 0;
  }
  PetscInitialize(&argc, &argv, NULL, NULL);

  // Print date
  if (rank == 0) {
    char buffer[80];
    time_t seconds = time(NULL);
    strftime(buffer, 80, "%A, %B %d, %Y %H:%M:%S", localtime(&seconds));
    printf(" # Datetime: %s\n", buffer);
  }

  // Print the number of processes
  if (rank == 0) printf(" # Number of proc.: %d\n", size);

  // Print the folder of the project
  char *project_folder = argv[1];
  if (rank == 0) printf(" # Project folder: %s\n\n", project_folder);

  // Print the config file name
  char *input_file = argv[2];
  if (rank == 0) printf(" # Input file: %s\n\n", input_file);

  // Run a program
  PetscReal t0 = 0, t1 = 0;
  if (rank == 0) PetscTime(&t0);

  Run(project_folder, input_file);

  if (rank == 0) PetscTime(&t1);

  // Run time
  if (rank == 0) printf("Run time: %g sec.\n", t1 - t0);

  // Finalize
  PetscFinalize();

  return 0;
}

void O3D::PressEnterToContinue() {
  printf("\n\tPress Enter To Continue ... \n");
  if (!feof(stdin) && !ferror(stdin)) {
    int ch;
    do ch = getc(stdin);
    while (ch != '\n' && ch != EOF);
    clearerr(stdin);
  }
}
