/*
 * dem_solver.cpp
 * ============================================================
 * 3D Discrete Element Method (DEM) Solver — HPSC 2026 Assignment 1
 * ============================================================
 *
 * Features:
 *   - Spring-dashpot (Kelvin-Voigt) contact model
 *   - Semi-implicit Euler time integration
 *   - Six-wall box domain
 *   - OpenMP parallelisation with thread-local force buffers
 *   - Cell-linked-list neighbour search (O(N) vs O(N^2))
 *   - RAII ScopedTimer profiling on every run
 *   - Organised output directories per run
 *   - Strong + weak scaling studies
 *   - Automated parallel correctness checker
 *   - Damping & timestep bonus investigations
 *
 * Build:
 *   Serial : g++ -std=c++17 -O3 -o dem_solver dem_solver.cpp
 *   OpenMP : g++ -std=c++17 -O3 -fopenmp -o dem_solver dem_solver.cpp
 *
 * Usage:
 *   ./dem_solver <mode> [output_dir]
 *   Modes: free_fall | constant_velocity | bounce | verification
 *          experiment | scaling | neighbor_bonus | science_bonus
 *          custom_case <N> <dt> <T> [outdir]
 * ============================================================
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace fs = std::filesystem;

// ============================================================
//  Vec3 — minimal 3D vector with full operator set
// ============================================================
struct Vec3 {
    double x = 0.0, y = 0.0, z = 0.0;

    Vec3& operator+=(const Vec3& r) { x+=r.x; y+=r.y; z+=r.z; return *this; }
    Vec3& operator-=(const Vec3& r) { x-=r.x; y-=r.y; z-=r.z; return *this; }
    Vec3& operator*=(double s)      { x*=s;   y*=s;   z*=s;   return *this; }
    Vec3& operator/=(double s)      { x/=s;   y/=s;   z/=s;   return *this; }
};

Vec3 operator+(Vec3 a, const Vec3& b) { return a += b; }
Vec3 operator-(Vec3 a, const Vec3& b) { return a -= b; }
Vec3 operator*(Vec3 a, double s)      { return a *= s; }
Vec3 operator*(double s, Vec3 a)      { return a *= s; }
Vec3 operator/(Vec3 a, double s)      { return a /= s; }
Vec3 operator-(const Vec3& a)         { return {-a.x, -a.y, -a.z}; }

double dot(const Vec3& a, const Vec3& b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
double norm(const Vec3& v)               { return std::sqrt(dot(v, v)); }
double norm2(const Vec3& v)              { return dot(v, v); }

// ============================================================
//  Core data structures
// ============================================================

struct Particle {
    Vec3   position;
    Vec3   velocity;
    Vec3   force;
    double radius = 0.01;
    double mass   = 1.0;
};

struct Domain {
    double lx = 1.0, ly = 1.0, lz = 1.0;
};

struct SimulationParams {
    double dt           = 1.0e-4;
    double total_time   = 1.0;
    double kn           = 2.0e4;   // Normal spring stiffness  [N/m]
    double gamma_n      = 20.0;    // Normal damping coefficient [kg/s]
    Vec3   gravity      = {0.0, 0.0, -9.81};
    int    output_every = 100;
    int    progress_every = 0;     // 0 = auto (10 prints per run)
    bool   use_neighbor_search = false;
    bool   use_openmp          = false;
    bool   clamp_contact_force = true;  // prevent tensile contact forces
};

// Profiling buckets — filled automatically by ScopedTimer on every run.
struct ProfileData {
    double zero_force        = 0.0;
    double gravity           = 0.0;
    double particle_contacts = 0.0;
    double wall_contacts     = 0.0;
    double integration       = 0.0;
    double diagnostics       = 0.0;

    double total() const {
        return zero_force + gravity + particle_contacts
             + wall_contacts + integration + diagnostics;
    }
    double contact_fraction() const {
        double t = total();
        return t > 0.0 ? (particle_contacts + wall_contacts) / t : 0.0;
    }
};

// Per-timestep contact statistics.
struct StepStats {
    std::size_t active_contacts = 0;
    std::size_t candidate_pairs = 0;
};

// Rich scalar summary written every output_every steps.
struct Diagnostics {
    double      time              = 0.0;
    double      kinetic_energy    = 0.0;
    double      potential_energy  = 0.0;  // m*g*z relative to floor
    double      max_speed         = 0.0;
    Vec3        center_of_mass    = {};
    double      max_height        = 0.0;
    double      min_height        = std::numeric_limits<double>::max();
    std::size_t active_contacts   = 0;
    std::size_t candidate_pairs   = 0;
};

// Everything returned after one complete simulation run.
struct RunSummary {
    double      wall_seconds      = 0.0;
    double      simulated_seconds = 0.0;
    Diagnostics last;
    ProfileData profile;
};

// Key-value map used for config snapshots (written to run_config.txt).
struct ConfigSnapshot {
    std::string                      mode;
    std::map<std::string,std::string> values;
};

// ============================================================
//  RAII timer — automatically adds elapsed time to a bucket
// ============================================================
class ScopedTimer {
    using Clock = std::chrono::steady_clock;
    double&           bucket_;
    Clock::time_point start_;
public:
    explicit ScopedTimer(double& b) : bucket_(b), start_(Clock::now()) {}
    ~ScopedTimer() {
        bucket_ += std::chrono::duration<double>(Clock::now() - start_).count();
    }
};

// ============================================================
//  Utility helpers
// ============================================================
static std::string dstr(double v) {
    std::ostringstream o; o << std::setprecision(16) << v; return o.str();
}

static void ensure_dir(const fs::path& p) { fs::create_directories(p); }

static void csv_line(std::ofstream& f, std::initializer_list<std::string> fields) {
    bool first = true;
    for (const auto& s : fields) { if (!first) f << ','; f << s; first = false; }
    f << '\n';
}

static double sphere_mass(double r, double rho) {
    constexpr double pi = 3.14159265358979323846;
    return rho * (4.0/3.0) * pi * r*r*r;
}

// ============================================================
//  Contact model
// ============================================================

// Returns the spring-dashpot force on particle b (force on a is negated).
// Returns zero Vec3 if no contact.
inline Vec3 contact_force(const Particle& a, const Particle& b,
                           double kn, double gn, bool clamp)
{
    const Vec3   rij  = b.position - a.position;
    const double d2   = norm2(rij);
    const double Rsum = a.radius + b.radius;

    if (d2 >= Rsum * Rsum || d2 < 1e-24) return {};

    const double d     = std::sqrt(d2);
    const Vec3   nij   = rij / d;
    const double delta = Rsum - d;
    const double vn    = dot(b.velocity - a.velocity, nij);
    double Fn = kn * delta - gn * vn;
    if (clamp && Fn < 0.0) Fn = 0.0;
    return Fn * nij;
}

// Single-wall spring-dashpot: outward normal is `wall_n`, overlap is `delta`.
inline Vec3 wall_force(double delta, const Vec3& wall_n, const Vec3& vel,
                       double kn, double gn, bool clamp)
{
    if (delta <= 0.0) return {};
    const double vn = dot(vel, wall_n);
    double Fn = kn * delta - gn * vn;
    if (clamp && Fn < 0.0) Fn = 0.0;
    return Fn * wall_n;
}

// ============================================================
//  Per-timestep force routines
// ============================================================

void zero_forces(std::vector<Particle>& ps) {
    for (auto& p : ps) p.force = {};
}

void add_gravity(std::vector<Particle>& ps, const Vec3& g) {
    for (auto& p : ps) p.force += p.mass * g;
}

// Checks all six walls and accumulates contact forces.
// Returns the number of active wall contacts.
std::size_t compute_wall_contacts(std::vector<Particle>& ps,
                                  const Domain& box,
                                  const SimulationParams& par)
{
    std::size_t cnt = 0;
    const double kn = par.kn, gn = par.gamma_n;
    const bool   cl = par.clamp_contact_force;

    for (auto& p : ps) {
        const double R = p.radius;
        // -x wall
        if (double d = R - p.position.x;       d>0) { p.force += wall_force(d,{ 1,0,0},p.velocity,kn,gn,cl); ++cnt; }
        // +x wall
        if (double d = p.position.x+R-box.lx;  d>0) { p.force += wall_force(d,{-1,0,0},p.velocity,kn,gn,cl); ++cnt; }
        // -y wall
        if (double d = R - p.position.y;       d>0) { p.force += wall_force(d,{ 0,1,0},p.velocity,kn,gn,cl); ++cnt; }
        // +y wall
        if (double d = p.position.y+R-box.ly;  d>0) { p.force += wall_force(d,{ 0,-1,0},p.velocity,kn,gn,cl); ++cnt; }
        // -z (floor)
        if (double d = R - p.position.z;       d>0) { p.force += wall_force(d,{ 0,0,1},p.velocity,kn,gn,cl); ++cnt; }
        // +z (ceiling)
        if (double d = p.position.z+R-box.lz;  d>0) { p.force += wall_force(d,{ 0,0,-1},p.velocity,kn,gn,cl); ++cnt; }
    }
    return cnt;
}

// ============================================================
//  Neighbour grid (cell-linked list)  — O(N) contact search
// ============================================================
struct CellIndex { int ix, iy, iz; };

class NeighborGrid {
public:
    NeighborGrid(const Domain& box, double cell_size)
      : box_(box), cs_(cell_size)
    {
        nx_ = std::max(1, (int)std::ceil(box_.lx / cs_));
        ny_ = std::max(1, (int)std::ceil(box_.ly / cs_));
        nz_ = std::max(1, (int)std::ceil(box_.lz / cs_));
        cells_.assign((std::size_t)(nx_*ny_*nz_), {});
    }

    void rebuild(const std::vector<Particle>& ps) {
        for (auto& c : cells_) c.clear();
        for (int i = 0; i < (int)ps.size(); ++i)
            cells_[linear(cell_of(ps[i].position))].push_back(i);
    }

    // Calls callback(i, j) for every candidate pair (i < j).
    // Returns number of candidate pairs examined.
    template<typename CB>
    std::size_t for_candidate_pairs(CB callback) const {
        std::size_t cands = 0;
        for (int iz=0; iz<nz_; ++iz)
        for (int iy=0; iy<ny_; ++iy)
        for (int ix=0; ix<nx_; ++ix) {
            const CellIndex ci{ix,iy,iz};
            const auto& A = cells_[linear(ci)];
            if (A.empty()) continue;

            for (int dz=-1; dz<=1; ++dz)
            for (int dy=-1; dy<=1; ++dy)
            for (int dx=-1; dx<=1; ++dx) {
                const CellIndex cj{ix+dx, iy+dy, iz+dz};
                if (!valid(cj)) continue;
                // Process each unordered neighbour-cell pair once.
                if (lex_before(cj, ci)) continue;

                const auto& B = cells_[linear(cj)];
                if (B.empty()) continue;
                const bool same = (dx==0 && dy==0 && dz==0);

                for (std::size_t a=0; a<A.size(); ++a) {
                    const std::size_t b0 = same ? a+1 : 0;
                    for (std::size_t b=b0; b<B.size(); ++b) {
                        ++cands;
                        callback(A[a], B[b]);
                    }
                }
            }
        }
        return cands;
    }

private:
    Domain box_;
    double cs_;
    int nx_, ny_, nz_;
    std::vector<std::vector<int>> cells_;

    CellIndex cell_of(const Vec3& p) const {
        auto cl = [](int v, int mx){ return std::max(0, std::min(v, mx-1)); };
        return { cl((int)(p.x/cs_), nx_),
                 cl((int)(p.y/cs_), ny_),
                 cl((int)(p.z/cs_), nz_) };
    }
    std::size_t linear(const CellIndex& c) const {
        return (std::size_t)(c.ix + nx_*(c.iy + ny_*c.iz));
    }
    bool valid(const CellIndex& c) const {
        return c.ix>=0 && c.ix<nx_ && c.iy>=0 && c.iy<ny_ && c.iz>=0 && c.iz<nz_;
    }
    static bool lex_before(const CellIndex& a, const CellIndex& b) {
        return std::tie(a.iz,a.iy,a.ix) < std::tie(b.iz,b.iy,b.ix);
    }
};

// ============================================================
//  Particle-particle contact detection
// ============================================================

// Serial brute-force O(N^2)
StepStats contacts_serial_bruteforce(std::vector<Particle>& ps,
                                     const SimulationParams& par)
{
    StepStats st;
    const int N = (int)ps.size();
    for (int i=0; i<N-1; ++i) {
        for (int j=i+1; j<N; ++j) {
            ++st.candidate_pairs;
            Vec3 f = contact_force(ps[i], ps[j], par.kn, par.gamma_n,
                                   par.clamp_contact_force);
            if (norm2(f) > 0.0) {
                ps[i].force -= f;
                ps[j].force += f;
                ++st.active_contacts;
            }
        }
    }
    return st;
}

// Serial with neighbour grid — O(N) for dilute systems
StepStats contacts_serial_grid(std::vector<Particle>& ps,
                               const SimulationParams& par,
                               NeighborGrid& grid)
{
    StepStats st;
    grid.rebuild(ps);
    st.candidate_pairs = grid.for_candidate_pairs([&](int i, int j){
        Vec3 f = contact_force(ps[i], ps[j], par.kn, par.gamma_n,
                               par.clamp_contact_force);
        if (norm2(f) > 0.0) {
            ps[i].force -= f;
            ps[j].force += f;
            ++st.active_contacts;
        }
    });
    return st;
}

// OpenMP parallel brute-force with per-thread force buffers.
// Each thread accumulates into its own buffer → no races → final reduction.
// Uses dynamic scheduling for the triangular loop (row i has N-i-1 entries).
StepStats contacts_parallel_omp(std::vector<Particle>& ps,
                                const SimulationParams& par)
{
    StepStats st;
#ifdef _OPENMP
    const int N = (int)ps.size();
    const int nthreads = omp_get_max_threads();

    // Per-thread force buffers — avoids all race conditions on ps[i].force
    std::vector<std::vector<Vec3>> bufs(
        (std::size_t)nthreads, std::vector<Vec3>((std::size_t)N));

    std::vector<std::size_t> thr_contacts((std::size_t)nthreads, 0);
    std::vector<std::size_t> thr_cands(  (std::size_t)nthreads, 0);

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        auto& buf     = bufs[(std::size_t)tid];
        std::size_t lc = 0, lcand = 0;

        // dynamic,32 gives better load balance for the triangular loop.
        #pragma omp for schedule(dynamic, 32) nowait
        for (int i = 0; i < N-1; ++i) {
            for (int j = i+1; j < N; ++j) {
                ++lcand;
                Vec3 f = contact_force(ps[i], ps[j],
                                       par.kn, par.gamma_n,
                                       par.clamp_contact_force);
                if (norm2(f) > 0.0) {
                    buf[i] -= f;
                    buf[j] += f;
                    ++lc;
                }
            }
        }
        thr_contacts[tid] = lc;
        thr_cands[tid]    = lcand;
    }

    // Reduction: merge all thread buffers into ps[i].force
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        for (int t = 0; t < nthreads; ++t)
            ps[i].force += bufs[t][i];
    }

    st.active_contacts = std::accumulate(thr_contacts.begin(), thr_contacts.end(), 0ULL);
    st.candidate_pairs = std::accumulate(thr_cands.begin(),    thr_cands.end(),    0ULL);
#else
    st = contacts_serial_bruteforce(ps, par);
#endif
    return st;
}

// ============================================================
//  Time integration (semi-implicit Euler)
// ============================================================
void integrate(std::vector<Particle>& ps, double dt) {
    for (auto& p : ps) {
        p.velocity += (p.force / p.mass) * dt;  // velocity updated first
        p.position += p.velocity * dt;           // then position uses new velocity
    }
}

// ============================================================
//  Diagnostics
// ============================================================
Diagnostics compute_diagnostics(const std::vector<Particle>& ps,
                                double time,
                                const StepStats& ss,
                                const Vec3& g)
{
    Diagnostics d;
    d.time           = time;
    d.active_contacts= ss.active_contacts;
    d.candidate_pairs= ss.candidate_pairs;
    d.min_height     = std::numeric_limits<double>::max();

    double mass_sum = 0.0;
    for (const auto& p : ps) {
        const double spd = norm(p.velocity);
        const double v2  = spd * spd;
        d.kinetic_energy   += 0.5 * p.mass * v2;
        d.potential_energy += p.mass * (-g.z) * p.position.z; // g.z is negative
        d.max_speed         = std::max(d.max_speed, spd);
        d.center_of_mass   += p.mass * p.position;
        d.max_height        = std::max(d.max_height, p.position.z);
        d.min_height        = std::min(d.min_height, p.position.z);
        mass_sum += p.mass;
    }
    if (mass_sum > 0.0) d.center_of_mass /= mass_sum;
    return d;
}

// ============================================================
//  I/O helpers
// ============================================================
void write_particles(const fs::path& path, const std::vector<Particle>& ps) {
    std::ofstream f(path);
    f << "id,x,y,z,vx,vy,vz,radius,mass\n";
    for (std::size_t i=0; i<ps.size(); ++i) {
        const auto& p = ps[i];
        f << i << ',' << std::setprecision(14)
          << p.position.x << ',' << p.position.y << ',' << p.position.z << ','
          << p.velocity.x << ',' << p.velocity.y << ',' << p.velocity.z << ','
          << p.radius << ',' << p.mass << '\n';
    }
}

void write_config(const fs::path& path, const ConfigSnapshot& cfg) {
    std::ofstream f(path);
    f << "mode=" << cfg.mode << '\n';
    for (const auto& [k,v] : cfg.values) f << k << '=' << v << '\n';
}

void write_diag_header(std::ofstream& f) {
    f << "time,kinetic_energy,potential_energy,total_energy,"
         "max_speed,com_x,com_y,com_z,"
         "max_height,min_height,active_contacts,candidate_pairs\n";
}

void append_diag(std::ofstream& f, const Diagnostics& d) {
    f << std::setprecision(14)
      << d.time << ','
      << d.kinetic_energy << ','
      << d.potential_energy << ','
      << d.kinetic_energy + d.potential_energy << ','
      << d.max_speed << ','
      << d.center_of_mass.x << ',' << d.center_of_mass.y << ',' << d.center_of_mass.z << ','
      << d.max_height << ','
      << (d.min_height > 1e29 ? 0.0 : d.min_height) << ','
      << d.active_contacts << ',' << d.candidate_pairs << '\n';
}

void write_profile(const fs::path& path, const ProfileData& pr) {
    std::ofstream f(path);
    f << std::fixed << std::setprecision(6);
    const double tot = pr.total();
    auto pct = [&](double v){ return tot > 0.0 ? 100.0*v/tot : 0.0; };
    f << "function,time_s,percent\n";
    f << "zero_force,"        << pr.zero_force        << ',' << pct(pr.zero_force)        << '\n';
    f << "gravity,"           << pr.gravity           << ',' << pct(pr.gravity)           << '\n';
    f << "particle_contacts," << pr.particle_contacts << ',' << pct(pr.particle_contacts) << '\n';
    f << "wall_contacts,"     << pr.wall_contacts     << ',' << pct(pr.wall_contacts)     << '\n';
    f << "integration,"       << pr.integration       << ',' << pct(pr.integration)       << '\n';
    f << "diagnostics,"       << pr.diagnostics       << ',' << pct(pr.diagnostics)       << '\n';
}

void write_summary(const fs::path& path, const RunSummary& s,
                   const std::string& label,
                   const std::vector<std::pair<std::string,std::string>>& extra = {})
{
    std::ofstream f(path);
    auto w = [&](const std::string& k, const std::string& v){
        f << std::left << std::setw(30) << k << ": " << v << '\n';
    };
    w("label",              label);
    w("wall_seconds",       dstr(s.wall_seconds));
    w("simulated_seconds",  dstr(s.simulated_seconds));
    w("final_KE",           dstr(s.last.kinetic_energy));
    w("final_PE",           dstr(s.last.potential_energy));
    w("final_E_total",      dstr(s.last.kinetic_energy + s.last.potential_energy));
    w("max_speed",          dstr(s.last.max_speed));
    w("active_contacts",    std::to_string(s.last.active_contacts));
    w("contact_frac_%",     dstr(100.0 * s.profile.contact_fraction()));
    for (const auto& [k,v] : extra) w(k, v);
}

void print_progress(const Diagnostics& d, int step, int total) {
    std::cout << std::fixed << std::setprecision(4)
              << "  step " << std::setw(7) << step << "/" << total
              << "  t=" << d.time
              << "  KE=" << std::scientific << std::setprecision(3) << d.kinetic_energy
              << std::fixed
              << "  COM_z=" << d.center_of_mass.z
              << "  contacts=" << d.active_contacts
              << '\n';
}

void print_summary(const RunSummary& s, const std::string& label) {
    std::cout << "\n[" << label << "]  wall=" << s.wall_seconds
              << "s  KE=" << s.last.kinetic_energy
              << "  contact_stage=" << s.profile.particle_contacts + s.profile.wall_contacts
              << "s (" << std::setprecision(1)
              << 100.0 * s.profile.contact_fraction() << "%)\n";
}

// ============================================================
//  Core simulation driver
// ============================================================
RunSummary run_simulation(std::vector<Particle>    particles,
                          const Domain&            box,
                          const SimulationParams&  par,
                          const fs::path&          out_dir,
                          const ConfigSnapshot&    cfg,
                          bool                     write_snapshots)
{
    ensure_dir(out_dir);
    write_config(out_dir / "run_config.txt", cfg);

    std::ofstream diag_f(out_dir / "diagnostics.csv");
    write_diag_header(diag_f);

    // Build neighbour grid if requested.
    std::optional<NeighborGrid> grid;
    if (par.use_neighbor_search) {
        double rmax = 0.0;
        for (const auto& p : particles) rmax = std::max(rmax, p.radius);
        grid.emplace(box, std::max(2.2 * rmax, 1e-3));
    }

    ProfileData profile;
    const int N     = (int)particles.size();
    const int steps = (int)std::ceil(par.total_time / par.dt);
    const int prog_stride = (par.progress_every > 0)
                          ? par.progress_every
                          : std::max(1, steps / 10);

    auto wall_start = std::chrono::steady_clock::now();
    StepStats last_stats;
    Diagnostics last = compute_diagnostics(particles, 0.0, last_stats, par.gravity);
    append_diag(diag_f, last);
    print_progress(last, 0, steps);
    if (write_snapshots)
        write_particles(out_dir / "particles_step_000000.csv", particles);

    for (int step = 1; step <= steps; ++step) {
        const double t = step * par.dt;

        { ScopedTimer T(profile.zero_force);
          zero_forces(particles); }

        { ScopedTimer T(profile.gravity);
          add_gravity(particles, par.gravity); }

        StepStats ss;
        { ScopedTimer T(profile.particle_contacts);
          if (par.use_openmp && !par.use_neighbor_search)
              ss = contacts_parallel_omp(particles, par);
          else if (par.use_neighbor_search && grid)
              ss = contacts_serial_grid(particles, par, *grid);
          else
              ss = contacts_serial_bruteforce(particles, par);
        }

        { ScopedTimer T(profile.wall_contacts);
          ss.active_contacts += compute_wall_contacts(particles, box, par); }

        { ScopedTimer T(profile.integration);
          integrate(particles, par.dt); }

        { ScopedTimer T(profile.diagnostics);
          last = compute_diagnostics(particles, t, ss, par.gravity);
          append_diag(diag_f, last);
          last_stats = ss;
        }

        if (step == steps || step % prog_stride == 0)
            print_progress(last, step, steps);

        if (write_snapshots && (step % par.output_every == 0 || step == steps)) {
            std::ostringstream nm;
            nm << "particles_step_" << std::setw(6) << std::setfill('0') << step << ".csv";
            write_particles(out_dir / nm.str(), particles);
        }
    }

    auto wall_end = std::chrono::steady_clock::now();
    write_profile(out_dir / "profile.csv", profile);

    RunSummary s;
    s.wall_seconds      = std::chrono::duration<double>(wall_end - wall_start).count();
    s.simulated_seconds = steps * par.dt;
    s.last              = last;
    s.profile           = profile;
    return s;
}

// ============================================================
//  Particle factory helpers
// ============================================================
std::vector<Particle> make_single(const Vec3& pos, const Vec3& vel,
                                   double R, double rho)
{
    Particle p;
    p.position = pos; p.velocity = vel;
    p.radius = R; p.mass = sphere_mass(R, rho);
    return {p};
}

// Random cloud — particles placed in upper half of box with small jitter velocity.
std::vector<Particle> make_cloud(int N, const Domain& box,
                                  double R, double rho, unsigned seed,
                                  bool zero_vel = false)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> rx(R, box.lx - R);
    std::uniform_real_distribution<double> ry(R, box.ly - R);
    std::uniform_real_distribution<double> rz(box.lz*0.45, box.lz - R);
    std::uniform_real_distribution<double> jit(-0.05, 0.05);

    std::vector<Particle> ps;
    ps.reserve((std::size_t)N);
    for (int i=0; i<N; ++i) {
        Particle p;
        p.position = {rx(rng), ry(rng), rz(rng)};
        p.velocity = zero_vel ? Vec3{} : Vec3{jit(rng), jit(rng), jit(rng)};
        p.radius   = R;
        p.mass     = sphere_mass(R, rho);
        ps.push_back(p);
    }
    return ps;
}

// ============================================================
//  Experiment functions
// ============================================================

// ------ Test 1: Free fall ------
void run_free_fall(const fs::path& root) {
    std::cout << "\n>>> free_fall\n";
    const Domain box{1.0, 1.0, 4.0};
    SimulationParams par;
    par.dt = 1e-4; par.total_time = 0.6;
    par.kn = 2e5;  par.gamma_n = 0.0;  // no damping for clean comparison
    par.output_every = 200;

    const double z0  = 3.0;
    const double g   = -par.gravity.z;  // = 9.81
    auto ps = make_single({0.5, 0.5, z0}, {}, 0.03, 2500.0);
    const fs::path od = root / "free_fall";

    ConfigSnapshot cfg;
    cfg.mode = "free_fall";
    cfg.values = {{"dt",dstr(par.dt)},{"z0",dstr(z0)},{"total_time",dstr(par.total_time)}};

    const RunSummary s = run_simulation(ps, box, par, od, cfg, true);

    // Analytical values at t = total_time (before floor contact)
    const double z_exact  = z0 - 0.5*g*par.total_time*par.total_time;
    const double vz_exact = -g*par.total_time;

    // Write analytical comparison
    {
        std::ofstream f(od / "analytical_comparison.csv");
        f << "t,z_numerical,z_analytical,vz_numerical,vz_analytical,error_z\n";
        // Re-run in memory to get per-step data
        Particle p0;
        p0.position = {0.5, 0.5, z0};
        p0.mass  = sphere_mass(0.03, 2500.0);
        p0.radius = 0.03;
        int steps = (int)std::ceil(par.total_time / par.dt);
        for (int i=0; i<=steps; ++i) {
            double t = i * par.dt;
            double z_a  = z0 - 0.5*g*t*t;
            double vz_a = -g*t;
            if (p0.position.z > p0.radius + 0.01) {
                f << std::setprecision(10)
                  << t << ',' << p0.position.z << ',' << z_a << ','
                  << p0.velocity.z << ',' << vz_a << ','
                  << std::abs(p0.position.z - z_a) << '\n';
            }
            p0.force = {0,0, -p0.mass * g};
            p0.velocity.z += (p0.force.z / p0.mass) * par.dt;
            p0.position.z += p0.velocity.z * par.dt;
        }
    }

    write_summary(od / "summary.txt", s, "free_fall",
        {{"z_exact_at_T",  dstr(z_exact)},
         {"vz_exact_at_T", dstr(vz_exact)}});
    print_summary(s, "free_fall");
}

// ------ Test 2: Constant velocity ------
void run_constant_velocity(const fs::path& root) {
    std::cout << "\n>>> constant_velocity\n";
    const Domain box{2.0, 2.0, 2.0};
    SimulationParams par;
    par.dt = 2e-4; par.total_time = 0.5;
    par.gravity = {}; par.output_every = 500;

    const Vec3 pos0{0.5, 0.6, 0.7};
    const Vec3 vel0{0.8, -0.35, 0.5};
    auto ps = make_single(pos0, vel0, 0.03, 2500.0);
    const fs::path od = root / "constant_velocity";

    ConfigSnapshot cfg;
    cfg.mode = "constant_velocity";
    cfg.values = {{"dt",dstr(par.dt)},{"total_time",dstr(par.total_time)}};

    const RunSummary s = run_simulation(ps, box, par, od, cfg, true);
    const Vec3 exact = pos0 + vel0 * par.total_time;

    write_summary(od / "summary.txt", s, "constant_velocity",
        {{"exact_x", dstr(exact.x)},
         {"exact_y", dstr(exact.y)},
         {"exact_z", dstr(exact.z)},
         {"com_x",   dstr(s.last.center_of_mass.x)},
         {"error_x", dstr(std::abs(s.last.center_of_mass.x - exact.x))}});
    print_summary(s, "constant_velocity");
}

// ------ Test 3: Bounce ------
void run_bounce(const fs::path& root, double gamma_n, const std::string& label) {
    std::cout << "\n>>> " << label << '\n';
    const Domain box{1.0, 1.0, 1.5};
    SimulationParams par;
    par.dt = 5e-5; par.total_time = 2.5;
    par.kn = 4e5; par.gamma_n = gamma_n;
    par.output_every = 1000;

    auto ps = make_single({0.5, 0.5, 0.9}, {}, 0.04, 2500.0);
    const fs::path od = root / label;
    ConfigSnapshot cfg;
    cfg.mode = "bounce";
    cfg.values = {{"gamma_n",dstr(gamma_n)},{"dt",dstr(par.dt)}};

    const RunSummary s = run_simulation(ps, box, par, od, cfg, true);
    write_summary(od / "summary.txt", s, label,
        {{"gamma_n", dstr(gamma_n)}, {"final_KE", dstr(s.last.kinetic_energy)}});
    print_summary(s, label);
}

// ------ Timestep sensitivity study ------
void run_dt_sensitivity(const fs::path& root) {
    std::cout << "\n>>> timestep_sensitivity\n";
    ensure_dir(root);
    std::ofstream out(root / "dt_scan.csv");
    out << "dt,error_z,error_vz,runtime_s\n";

    const double z0 = 3.0, T = 0.6, g = 9.81;
    const Domain box{1.0, 1.0, 4.0};

    for (double dt : {1e-2, 5e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5}) {
        // Integrate forward in memory (cheap, no file I/O)
        Particle p;
        p.position = {0.5, 0.5, z0};
        p.mass = sphere_mass(0.03, 2500.0); p.radius = 0.03;
        int steps = (int)std::ceil(T / dt);
        auto t0 = std::chrono::steady_clock::now();
        for (int s=0; s<steps; ++s) {
            p.force = {0,0, -p.mass * g};
            p.velocity.z += (p.force.z / p.mass) * dt;
            p.position.z += p.velocity.z * dt;
        }
        double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
        const double z_anal  = z0 - 0.5*g*T*T;
        const double vz_anal = -g*T;
        out << std::setprecision(10)
            << dt << ',' << std::abs(p.position.z - z_anal) << ','
            << std::abs(p.velocity.z - vz_anal) << ',' << elapsed << '\n';
    }
    std::cout << "  Wrote dt_scan.csv\n";
}

// ------ Multi-particle experiment ------
void run_experiment(const fs::path& root, int N,
                    bool use_omp, bool use_grid, int threads,
                    double total_time = 0.05)
{
    const Domain box{1.0, 1.0, 1.0};
    SimulationParams par;
    par.dt = 1e-4; par.total_time = total_time;
    par.kn = 2e4; par.gamma_n = 30.0;
    par.output_every = 999999;
    par.use_openmp = use_omp;
    par.use_neighbor_search = use_grid;

#ifdef _OPENMP
    if (threads > 0) omp_set_num_threads(threads);
#endif

    std::ostringstream nm;
    nm << "n" << N
       << (use_omp  ? "_omp"    : "_serial")
       << (use_grid ? "_grid"   : "_pairs");
    if (use_omp) nm << "_t" << threads;

    std::cout << "\n>>> " << nm.str() << '\n';

    auto ps = make_cloud(N, box, 0.015, 2400.0, (unsigned)(1234+N));
    ConfigSnapshot cfg;
    cfg.mode = "experiment";
    cfg.values = {{"N",std::to_string(N)},
                  {"omp", use_omp  ? "1":"0"},
                  {"grid",use_grid ? "1":"0"},
                  {"threads",std::to_string(threads)}};

    const RunSummary s = run_simulation(ps, box, par, root / nm.str(), cfg, false);

    write_summary(root / nm.str() / "summary.txt", s, nm.str(),
        {{"N", std::to_string(N)},
         {"threads", std::to_string(threads)},
         {"contact_%", dstr(100.0 * s.profile.contact_fraction())}});
    print_summary(s, nm.str());
}

// ------ Strong scaling ------
void run_strong_scaling(const fs::path& root, int N,
                        const std::vector<int>& thread_counts)
{
    ensure_dir(root);
    std::ofstream tbl(root / ("speedup_N" + std::to_string(N) + ".csv"));
    tbl << "threads,runtime_s,speedup,efficiency,"
           "contact_time_s,contact_pct\n";

    double baseline = 0.0;
    for (int t : thread_counts) {
        const Domain box{1.0, 1.0, 1.0};
        SimulationParams par;
        par.dt = 1e-4; par.total_time = 0.02;
        par.kn = 2e4; par.gamma_n = 30.0;
        par.output_every = 999999;
        par.use_openmp = (t > 1);

#ifdef _OPENMP
        omp_set_num_threads(t);
#endif
        auto ps = make_cloud(N, box, 0.015, 2400.0, 777u);
        std::ostringstream nm;
        nm << "strong_N" << N << "_t" << t;
        ConfigSnapshot cfg;
        cfg.mode = "strong_scaling";
        cfg.values = {{"N",std::to_string(N)},{"threads",std::to_string(t)}};

        std::cout << "\n>>> " << nm.str() << '\n';
        const RunSummary s = run_simulation(ps, box, par, root / nm.str(), cfg, false);

        if (t == thread_counts.front()) baseline = s.wall_seconds;
        const double speedup    = baseline / s.wall_seconds;
        const double efficiency = speedup / t;

        tbl << std::setprecision(6)
            << t << ',' << s.wall_seconds << ','
            << speedup << ',' << efficiency << ','
            << s.profile.particle_contacts + s.profile.wall_contacts << ','
            << 100.0 * s.profile.contact_fraction() << '\n';

        write_summary(root / nm.str() / "summary.txt", s, nm.str(),
            {{"speedup",    dstr(speedup)},
             {"efficiency", dstr(efficiency)}});
        print_summary(s, nm.str());
    }
}

// ------ Weak scaling ------
// N scales with thread count; box volume scales so density stays constant.
void run_weak_scaling(const fs::path& root, int base_N,
                      const std::vector<int>& thread_counts)
{
    ensure_dir(root);
    std::ofstream tbl(root / "weak_scaling.csv");
    tbl << "threads,N,box_side,runtime_s,weak_efficiency\n";
    double baseline = 0.0;

    for (int t : thread_counts) {
        const int    N    = base_N * t;
        const double side = std::cbrt((double)t);
        const Domain box{side, side, side};
        SimulationParams par;
        par.dt = 1e-4; par.total_time = 0.02;
        par.kn = 2e4; par.gamma_n = 30.0;
        par.output_every = 999999;
        par.use_openmp = (t > 1);

#ifdef _OPENMP
        omp_set_num_threads(t);
#endif
        auto ps = make_cloud(N, box, 0.015, 2400.0, (unsigned)(900+t));
        std::ostringstream nm;
        nm << "weak_t" << t << "_N" << N;
        ConfigSnapshot cfg;
        cfg.mode = "weak_scaling";
        cfg.values = {{"N",std::to_string(N)},{"threads",std::to_string(t)},
                      {"side",dstr(side)}};

        std::cout << "\n>>> " << nm.str() << '\n';
        const RunSummary s = run_simulation(ps, box, par, root / nm.str(), cfg, false);

        if (t == thread_counts.front()) baseline = s.wall_seconds;
        const double weak_eff = baseline / s.wall_seconds;
        tbl << std::setprecision(6)
            << t << ',' << N << ',' << side << ','
            << s.wall_seconds << ',' << weak_eff << '\n';
        print_summary(s, nm.str());
    }
}

// ------ Parallel correctness check ------
// Runs serial and parallel with identical seeds; reports max absolute differences.
void run_correctness_check(const fs::path& root, int N, int threads) {
    ensure_dir(root);
    std::cout << "\n>>> parallel_correctness N=" << N << " t=" << threads << '\n';

    const Domain box{1.0, 1.0, 1.0};
    SimulationParams base_par;
    base_par.dt = 1e-4; base_par.total_time = 0.02;
    base_par.kn = 2e4;  base_par.gamma_n = 30.0;
    base_par.output_every = 999999;

    SimulationParams ser_par = base_par; ser_par.use_openmp = false;
    SimulationParams omp_par = base_par; omp_par.use_openmp = true;

#ifdef _OPENMP
    omp_set_num_threads(threads);
#endif

    auto ps_ser = make_cloud(N, box, 0.015, 2400.0, 2026u);
    auto ps_omp = make_cloud(N, box, 0.015, 2400.0, 2026u);  // same seed!

    ConfigSnapshot cs, co;
    cs.mode = "correctness_serial"; cs.values = {{"N",std::to_string(N)}};
    co.mode = "correctness_omp";
    co.values = {{"N",std::to_string(N)},{"threads",std::to_string(threads)}};

    const RunSummary ss = run_simulation(ps_ser, box, ser_par,
                                         root/("serial_N"+std::to_string(N)), cs, false);
    const RunSummary so = run_simulation(ps_omp, box, omp_par,
                                         root/("omp_N"+std::to_string(N)), co, false);

    const double dKE   = std::abs(ss.last.kinetic_energy   - so.last.kinetic_energy);
    const double dSpeed= std::abs(ss.last.max_speed        - so.last.max_speed);
    const double dCOMz = std::abs(ss.last.center_of_mass.z - so.last.center_of_mass.z);

    // Append to shared correctness table
    const bool append = fs::exists(root / "correctness.csv");
    std::ofstream tbl(root / "correctness.csv",
                      append ? std::ios::app : std::ios::out);
    if (!append)
        tbl << "N,threads,dKE,dMaxSpeed,dCOMz,serial_KE,omp_KE\n";
    tbl << N << ',' << threads << ','
        << dKE << ',' << dSpeed << ',' << dCOMz << ','
        << ss.last.kinetic_energy << ',' << so.last.kinetic_energy << '\n';

    std::cout << "  dKE=" << dKE << "  dSpeed=" << dSpeed
              << "  dCOMz=" << dCOMz << '\n';
}

// ------ Neighbour grid benchmark ------
void run_neighbor_benchmark(const fs::path& root) {
    ensure_dir(root);
    std::ofstream tbl(root / "neighbor_comparison.csv");
    tbl << "N,mode,runtime_s,contact_stage_s,candidate_pairs,active_contacts\n";

    for (int N : {200, 1000, 5000}) {
        for (bool grid : {false, true}) {
            const Domain box{1.0, 1.0, 1.0};
            SimulationParams par;
            par.dt = 1e-4; par.total_time = 0.01;
            par.kn = 2e4; par.gamma_n = 25.0;
            par.use_neighbor_search = grid;
            par.output_every = 999999;

            auto ps = make_cloud(N, box, 0.012, 2400.0, (unsigned)(900+N));
            const std::string nm = std::string(grid?"grid_":"pairs_") + std::to_string(N);
            std::cout << "\n>>> " << nm << '\n';
            ConfigSnapshot cfg;
            cfg.mode = grid ? "neighbor_grid" : "all_pairs";
            cfg.values = {{"N",std::to_string(N)}};

            const RunSummary s = run_simulation(ps, box, par, root/nm, cfg, false);
            tbl << N << ',' << (grid?"grid":"all_pairs") << ','
                << s.wall_seconds << ','
                << s.profile.particle_contacts << ','
                << s.last.candidate_pairs << ','
                << s.last.active_contacts << '\n';
            print_summary(s, nm);
        }
    }
}

// ------ Damping investigation (Bonus science) ------
void run_damping_study(const fs::path& root) {
    std::cout << "\n>>> damping_study\n";
    ensure_dir(root);

    // --- Single particle bounce at various gamma_n ---
    {
        std::ofstream tbl(root / "bounce_damping_scan.csv");
        tbl << "gamma_n,final_KE,runtime_s\n";
        for (double gn : {5.0, 20.0, 50.0, 100.0, 200.0}) {
            const std::string lbl = "bounce_gn" + std::to_string((int)gn);
            run_bounce(root, gn, lbl);
            // Read back summary
            std::ifstream sf(root / lbl / "summary.txt");
            double fke = 0, rt = 0;
            std::string line;
            while (std::getline(sf, line)) {
                if (line.find("final_KE") != std::string::npos)
                    fke = std::stod(line.substr(line.find(':')+1));
                if (line.find("wall_seconds") != std::string::npos)
                    rt  = std::stod(line.substr(line.find(':')+1));
            }
            tbl << gn << ',' << fke << ',' << rt << '\n';
        }
    }

    // --- Cloud settling at various gamma_n ---
    {
        std::ofstream tbl(root / "cloud_damping_scan.csv");
        tbl << "gamma_n,final_KE,com_z,runtime_s\n";
        const Domain box{1.0, 1.0, 1.0};
        for (double gn : {5.0, 20.0, 50.0, 100.0}) {
            SimulationParams par;
            par.dt = 1e-4; par.total_time = 0.05;
            par.kn = 2e4; par.gamma_n = gn;
            par.use_neighbor_search = true;
            par.output_every = 999999;
            auto ps = make_cloud(500, box, 0.012, 2400.0, (unsigned)(1000+gn), true);
            const std::string lbl = "cloud_gn" + std::to_string((int)gn);
            ConfigSnapshot cfg;
            cfg.mode = "cloud_settling_damping";
            cfg.values = {{"gamma_n",dstr(gn)}};
            const RunSummary s = run_simulation(ps, box, par, root/lbl, cfg, false);
            tbl << gn << ',' << s.last.kinetic_energy << ','
                << s.last.center_of_mass.z << ',' << s.wall_seconds << '\n';
            print_summary(s, lbl);
        }
    }
}

// ------ Cloud settling demo ------
void run_cloud_settling(const fs::path& root) {
    std::cout << "\n>>> cloud_settling\n";
    const Domain box{1.0, 1.0, 1.0};
    SimulationParams par;
    par.dt = 1e-4; par.total_time = 0.1;
    par.kn = 2e4; par.gamma_n = 30.0;
    par.use_neighbor_search = true;
    par.output_every = 250;

    auto ps = make_cloud(1000, box, 0.012, 2400.0, 42u, true);
    ConfigSnapshot cfg;
    cfg.mode = "cloud_settling";
    cfg.values = {{"N","1000"}};
    const RunSummary s = run_simulation(ps, box, par, root / "cloud_settling", cfg, true);
    write_summary(root / "cloud_settling" / "summary.txt", s, "cloud_settling");
    print_summary(s, "cloud_settling");
}

// ============================================================
//  CLI dispatch
// ============================================================
void print_usage() {
    std::cout <<
        "Usage: dem_solver <mode> [output_dir]\n\n"
        "Modes:\n"
        "  free_fall           Single-particle free-fall verification\n"
        "  constant_velocity   Zero-force constant-velocity verification\n"
        "  bounce              Single-particle bounce verification\n"
        "  verification        All three verification tests + dt sensitivity\n"
        "  experiment          Multi-particle runs N=200,1000,5000\n"
        "  scaling             Full scaling study (serial profile + OMP strong+weak)\n"
        "  neighbor_bonus      Neighbor-grid vs all-pairs comparison\n"
        "  science_bonus       Damping investigation + cloud settling\n"
        "  custom_case N dt T [outdir]   Arbitrary single run\n";
}

int main(int argc, char** argv) {
    try {
        if (argc < 2) { print_usage(); return 1; }

        const std::string mode = argv[1];
        const fs::path root = (argc >= 3 && mode != "custom_case")
                            ? fs::path(argv[2]) : fs::path("results");

        if (mode == "free_fall") {
            run_free_fall(root);

        } else if (mode == "constant_velocity") {
            run_constant_velocity(root);

        } else if (mode == "bounce") {
            run_bounce(root, 40.0, "bounce_default");

        } else if (mode == "verification") {
            run_free_fall(root / "verification");
            run_constant_velocity(root / "verification");
            run_bounce(root / "verification", 20.0, "bounce_light");
            run_bounce(root / "verification", 70.0, "bounce_heavy");
            run_dt_sensitivity(root / "verification" / "dt_sensitivity");

        } else if (mode == "experiment") {
            const fs::path er = root / "experiments";
            for (int N : {200, 1000, 5000})
                run_experiment(er, N, false, false, 1, 0.05);

        } else if (mode == "scaling") {
            // --- Profiling (serial) ---
            const fs::path pr = root / "profiling";
            for (int N : {200, 1000, 5000})
                run_experiment(pr, N, false, false, 1, 0.03);

            // --- Strong scaling ---
            const std::vector<int> threads{1, 2, 4, 8};
            run_strong_scaling(root / "strong_scaling_N1000", 1000, threads);
            run_strong_scaling(root / "strong_scaling_N5000", 5000, threads);

            // --- Weak scaling ---
            run_weak_scaling(root / "weak_scaling", 500, threads);

            // --- Correctness check ---
            run_correctness_check(root / "correctness", 1000, 4);
            run_correctness_check(root / "correctness", 5000, 4);

        } else if (mode == "neighbor_bonus") {
            run_neighbor_benchmark(root / "neighbor_bonus");

        } else if (mode == "science_bonus") {
            run_damping_study(root / "science_bonus");
            run_cloud_settling(root / "science_bonus");

        } else if (mode == "custom_case") {
            if (argc < 5) { print_usage(); return 1; }
            const int    N  = std::stoi(argv[2]);
            const double dt = std::stod(argv[3]);
            const double T  = std::stod(argv[4]);
            const fs::path cr = (argc >= 6) ? fs::path(argv[5]) : fs::path("results");
            run_experiment(cr, N, false, false, 1, T);

        } else {
            print_usage(); return 1;
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 2;
    }
}
