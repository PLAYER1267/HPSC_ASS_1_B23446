# ============================================================
#  Makefile — 3D DEM Solver (HPSC 2026 Assignment 1)
# ============================================================

CXX      = g++
STD      = -std=c++17
WARN     = -Wall -Wextra -Wpedantic
OPT      = -O3 -march=native -ffast-math

TARGET_SERIAL = dem_serial
TARGET_OMP    = dem_omp
SRC           = dem_solver.cpp

.PHONY: all serial openmp clean \
        run_verification run_experiment run_scaling \
        run_neighbor run_science run_all plots

# ---- Build targets ----
all: serial openmp

serial:
	$(CXX) $(STD) $(WARN) $(OPT) -o $(TARGET_SERIAL) $(SRC)
	@echo "[OK] Built $(TARGET_SERIAL)"

openmp:
	$(CXX) $(STD) $(WARN) $(OPT) -fopenmp -o $(TARGET_OMP) $(SRC)
	@echo "[OK] Built $(TARGET_OMP)"

debug:
	$(CXX) $(STD) $(WARN) -g -O0 -fsanitize=address -o dem_debug $(SRC)

# ---- Run modes (serial) ----
run_verification: serial
	./$(TARGET_SERIAL) verification results

run_experiment: serial
	./$(TARGET_SERIAL) experiment results

# ---- Run modes (OpenMP) ----
run_scaling: openmp
	set OMP_NUM_THREADS=4 .\$(TARGET_OMP).exe scaling results

run_neighbor: openmp
	set OMP_NUM_THREADS=4 .\$(TARGET_OMP).exe neighbor_bonus results

run_science: openmp
	set OMP_NUM_THREADS=4 .\$(TARGET_OMP).exe science_bonus results

run_all: openmp run_verification run_experiment run_scaling run_neighbor run_science

# ---- Plots ----
plots:
	python plot_results.py

# ---- Clean ----
clean:
	rm -f $(TARGET_SERIAL) $(TARGET_OMP) dem_debug
	rm -rf results/

# ============================================================
# Quick reference:
#   make all                  Build serial + OpenMP
#   make run_verification     Verification tests
#   make run_scaling          Full scaling study (OMP)
#   make run_neighbor         Neighbour-grid vs all-pairs
#   make run_science          Damping + cloud settling
#   make plots                Generate all PDF figures
#   make clean                Remove binaries and results/
# ============================================================
