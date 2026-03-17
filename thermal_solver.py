"""
thermal_solver.py
=================
Solver FDM (Finite Difference Method) para la ecuación de conducción
térmica estacionaria con convección superficial (condición de Robin/Newton).

Ecuación gobernante:
--------------------
  ∇·(k_eff ∇T) - h/t × (T - T_amb) + q = 0    en el dominio Ω (PCB)

En 2D discreto (Fourier 2D + convección linealizada como término de sink):

  k_eff/dx² × (T[i+1,j] + T[i-1,j] - 2T[i,j])
+ k_eff/dy² × (T[i,j+1] + T[i,j-1] - 2T[i,j])
- (2h)/(k_eff × t_pcb) × (T[i,j] - T_amb)
+ q[i,j] / (k_eff × t_pcb)
= 0

El término de convección se incluye en ambas caras (factor 2):
  sink = 2h/(k_eff × t_pcb)    [m⁻²]

Condiciones de contorno:
------------------------
  Bordes: condición de Neumann homogénea (∂T/∂n = 0, adiabático)
  Convección: modelada como term de reacción volumétrico (Robin implícito)

  Nota: La BC adiabática en bordes es conservadora. Para PCBs con
  montaje en rack se puede usar BC de temperatura fija en el borde
  de contacto (condición de Dirichlet).

Método numérico:
----------------
  SOR (Successive Over-Relaxation) con factor ω configurable.
  Alternativa implementada: scipy.sparse.linalg para mayor velocidad.

  El SOR converge en O(N²) para grillas regulares, pero la
  versión matricial (scipy) es significativamente más rápida:
  O(N) con BICGSTAB precondicioned con ILU.

Referencias:
  - Patankar, S.V., "Numerical Heat Transfer and Fluid Flow", 1980
  - Smith, G.D., "Numerical Solution of Partial Differential Equations:
    Finite Difference Methods", 3rd Ed., 1985
  - Anderson et al., "Computational Fluid Dynamics: The Basics with Applications",
    McGraw-Hill, 1995
  - Incropera & DeWitt, "Introduction to Heat Transfer", 6th Ed., 2011
    Cap. 4: Two-Dimensional Steady-State Conduction
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, bicgstab
from typing import Tuple, Optional, List
import time

from config import PCBConfig
from material_properties import ThermalProperties
from component_mapper import ThermalGrid


class FDMThermalSolver:
    """
    Solver FDM para conducción térmica 2D estacionaria en PCB.

    El sistema de ecuaciones resultante es lineal y simétrico definido positivo.
    Se puede resolver con:
      (a) Direct sparse solver (spsolve) — exacto, O(N^1.5) para grillas 2D
      (b) Iterativo BICGSTAB — O(N·iterations), paralelo
      (c) SOR iterativo — simple, diagnóstico fácil

    Para grillas 200×200 = 40,000 nodos: spsolve es más rápido.
    Para grillas 1000×1000 = 1,000,000 nodos: BICGSTAB es necesario.
    """

    def __init__(self, cfg: PCBConfig, props: ThermalProperties):
        self.cfg = cfg
        self.props = props
        self.nx = cfg.grid_nx
        self.ny = cfg.grid_ny

    def _build_system_matrix(self,
                              grid: ThermalGrid
                              ) -> Tuple[sparse.csr_matrix, np.ndarray]:
        """
        Construye el sistema lineal A·T = b usando discretización FDM de 5 puntos.

        Numeración de nodos: k = j * nx + i  (row-major, C-order)
        donde i ∈ [0, nx), j ∈ [0, ny)

        Coeficientes de la ecuación discreta para nodo interior (i,j):
          a_E = a_W = k_eff / dx²    [W/m²·K]
          a_N = a_S = k_eff / dy²    [W/m²·K]
          a_C = -(a_E + a_W + a_N + a_S) - 2h/t_pcb

        Término fuente:
          b[i,j] = -q[i,j] - (2h/t_pcb) × T_amb

        El factor 2h/t_pcb modela la convección en ambas caras (superior e inferior).

        Para nodos en borde (Neumann ∂T/∂n=0):
          Se omite la contribución del vecino exterior (coeficiente = 0)
          y se aumenta el central para mantener la simetría de la matriz.
          (Esto equivale a imagen espejo del nodo exterior.)

        Returns:
            (A, b): Matriz sparse CSR y vector RHS
        """
        nx, ny = self.nx, self.ny
        N = nx * ny

        k_eff = self.props.k_eff_xy_WmK
        dx = grid.dx_m
        dy = grid.dy_m
        h = self.props.h_surface_Wm2K
        t = self.props.pcb_thickness_m
        T_amb = self.props.T_ambient_C

        # Coeficientes de conducción
        a_EW = k_eff / (dx ** 2)   # [W/m³·K]
        a_NS = k_eff / (dy ** 2)   # [W/m³·K]

        # Término de convección (ambas caras): sink coefficient [W/m³·K]
        # Derivado de: q_conv = h·(T-T_amb) por cada cara → por volumen:
        # h·(T-T_amb)·dx·dy·2 / (t·dx·dy) = 2h/t × (T-T_amb)
        a_conv = 2.0 * h / t       # [W/m³·K]

        # Diagonal central: sum de todos los coeficientes de conducción + convección
        a_center_interior = -(2 * a_EW + 2 * a_NS + a_conv)

        # Arrays de coordenadas para construcción COO → CSR
        rows, cols, vals = [], [], []
        b = np.zeros(N, dtype=np.float64)

        # Flatten del mapa de densidad de potencia
        q_flat = grid.power_density.ravel()  # [W/m²] → necesitamos [W/m³]
        # q_vol [W/m³] = q_surface [W/m²] / t_pcb
        q_vol = q_flat / t

        for j in range(ny):
            for i in range(nx):
                k = j * nx + i

                # Determinar vecinos (con BC Neumann en bordes)
                n_east  = (i < nx - 1)
                n_west  = (i > 0)
                n_north = (j < ny - 1)
                n_south = (j > 0)

                # Coeficientes de conducción para este nodo
                a_E = a_EW if n_east  else 0.0
                a_W = a_EW if n_west  else 0.0
                a_N = a_NS if n_north else 0.0
                a_S = a_NS if n_south else 0.0

                # Diagonal central (negativo por convención A·T = b, T positivo)
                a_c = -(a_E + a_W + a_N + a_S + a_conv)

                # Ensamblar fila
                rows.append(k); cols.append(k); vals.append(a_c)

                if n_east:
                    rows.append(k); cols.append(k + 1);  vals.append(a_E)
                if n_west:
                    rows.append(k); cols.append(k - 1);  vals.append(a_W)
                if n_north:
                    rows.append(k); cols.append(k + nx); vals.append(a_N)
                if n_south:
                    rows.append(k); cols.append(k - nx); vals.append(a_S)

                # RHS: fuente térmica + término de convección
                b[k] = -q_vol[k] - a_conv * T_amb

        A = sparse.csr_matrix(
            (np.array(vals), (np.array(rows), np.array(cols))),
            shape=(N, N),
            dtype=np.float64
        )

        return A, b

    def solve_direct(self, grid: ThermalGrid) -> np.ndarray:
        """
        Resolución directa del sistema A·T = b con spsolve (SuperLU).

        Ventajas: Exacto (hasta precisión de máquina), sin parámetros de convergencia.
        Limitación: Memoria O(N^1.5) para grillas 2D regulares.
        Recomendado para N = nx*ny < 250,000 nodos.

        Returns:
            T_flat: Array 1D de temperaturas en °C, reshape a (ny, nx)
        """
        print(f"[Solver] Construyendo sistema FDM: {self.nx}×{self.ny} = "
              f"{self.nx*self.ny:,} nodos...")
        t0 = time.time()
        A, b = self._build_system_matrix(grid)
        t1 = time.time()
        print(f"  Ensamblaje: {t1-t0:.2f}s | "
              f"NNZ: {A.nnz:,} | Densidad: {A.nnz/(A.shape[0]**2)*100:.4f}%")

        print(f"[Solver] Resolviendo con spsolve (SuperLU)...")
        T_flat = spsolve(A, b)
        t2 = time.time()
        print(f"  Tiempo resolución: {t2-t1:.2f}s")

        return T_flat.reshape((self.ny, self.nx))

    def solve_iterative_bicgstab(self,
                                  grid: ThermalGrid,
                                  tol: float = 1e-6,
                                  maxiter: int = 2000
                                  ) -> np.ndarray:
        """
        Resolución iterativa con BICGSTAB precondicionado con ILU.

        Precondicionador ILU(0): Incomplete LU, llenado cero.
        Reduce el número de iteraciones de O(N) a O(√N) típicamente.

        Recomendado para grillas grandes (N > 250,000 nodos).

        Args:
            tol:     Tolerancia relativa [1e-6 es suficiente para térmica]
            maxiter: Iteraciones máximas

        Returns:
            T_flat: Array 2D de temperaturas [°C]
        """
        from scipy.sparse.linalg import spilu, LinearOperator

        print(f"[Solver] Construyendo sistema FDM: {self.nx}×{self.ny}...")
        A, b = self._build_system_matrix(grid)

        print(f"[Solver] Precondicionando con ILU(0)...")
        t0 = time.time()
        ilu = spilu(A.tocsc(), fill_factor=1.0)
        M = LinearOperator(A.shape, lambda x: ilu.solve(x))

        T0 = np.full(len(b), self.props.T_ambient_C)
        T_flat, info = bicgstab(A, b, x0=T0, M=M, tol=tol, maxiter=maxiter)

        t1 = time.time()
        if info == 0:
            print(f"  BICGSTAB convergió en {t1-t0:.2f}s")
        elif info > 0:
            print(f"  [WARN] BICGSTAB no convergió en {maxiter} iteraciones (tol={tol})")
        else:
            print(f"  [ERROR] BICGSTAB falló (info={info})")

        return T_flat.reshape((self.ny, self.nx))

    def solve_sor(self,
                  grid: ThermalGrid,
                  omega: float = None,
                  max_iter: int = None,
                  tol: float = None
                  ) -> Tuple[np.ndarray, List[float]]:
        """
        Solver SOR (Successive Over-Relaxation) iterativo puro NumPy.

        T_new[i,j] = (1-ω)·T_old[i,j] + ω/a_c × (b[i,j] - Σ a_k·T_k)

        Ventaja: Transparencia didáctica, control por iteración.
        Desventaja: Más lento que spsolve para la misma grilla.

        El factor ω óptimo teórico para una grilla rectangular:
          ω_opt ≈ 2 / (1 + sin(π/max(nx,ny)))

        Args:
            omega:    Factor de relajación (1=Gauss-Seidel, 1<ω<2=SOR)
            max_iter: Máximo de iteraciones
            tol:      Tolerancia de convergencia [°C]

        Returns:
            (T, residuals): Temperatura 2D y lista de residuos por iteración
        """
        from typing import List

        cfg = self.cfg
        props = self.props
        omega   = omega    or cfg.solver_omega
        max_iter = max_iter or cfg.solver_max_iter
        tol     = tol      or cfg.solver_tolerance

        nx, ny = self.nx, self.ny
        dx = grid.dx_m
        dy = grid.dy_m
        k  = props.k_eff_xy_WmK
        h  = props.h_surface_Wm2K
        t  = props.pcb_thickness_m
        T_amb = props.T_ambient_C

        a_EW  = k / dx**2
        a_NS  = k / dy**2
        a_conv = 2.0 * h / t
        q_vol = grid.power_density / t   # [W/m³]

        T = np.full((ny, nx), T_amb, dtype=np.float64)
        residuals = []

        print(f"[Solver-SOR] ω={omega:.3f} | max_iter={max_iter} | tol={tol}")
        t0 = time.time()

        for iteration in range(max_iter):
            T_old_max = T.max()
            T_prev = T.copy()

            for j in range(ny):
                for i in range(nx):
                    # Vecinos con BC Neumann (imagen espejo)
                    T_E = T[j, i+1] if i < nx-1 else T[j, i]
                    T_W = T[j, i-1] if i > 0    else T[j, i]
                    T_N = T[j+1, i] if j < ny-1 else T[j, i]
                    T_S = T[j-1, i] if j > 0    else T[j, i]

                    a_c = -(
                        (a_EW if i < nx-1 else 0) + (a_EW if i > 0 else 0) +
                        (a_NS if j < ny-1 else 0) + (a_NS if j > 0 else 0) +
                        a_conv
                    )

                    rhs = (a_EW * (T_E + T_W) + a_NS * (T_N + T_S)
                           - a_conv * T_amb - q_vol[j, i])

                    T_gs = -rhs / (-a_c)  # Gauss-Seidel step
                    T[j, i] = (1 - omega) * T[j, i] + omega * T_gs

            # Convergencia: norma L∞ del residuo
            residual = np.max(np.abs(T - T_prev))
            residuals.append(residual)

            if (iteration + 1) % 100 == 0:
                print(f"  iter {iteration+1:4d}: residual = {residual:.2e}°C | "
                      f"T_max = {T.max():.2f}°C")

            if residual < tol:
                print(f"  Convergido en {iteration+1} iteraciones, "
                      f"residual={residual:.2e}°C | t={time.time()-t0:.1f}s")
                break

        return T, residuals

    def solve(self,
              grid: ThermalGrid,
              method: str = 'direct'
              ) -> np.ndarray:
        """
        Interfaz unificada del solver.

        Args:
            grid:   ThermalGrid con power_density poblado
            method: 'direct' | 'bicgstab' | 'sor'

        Returns:
            T: Array 2D (ny, nx) de temperaturas en °C
        """
        print(f"\n{'='*60}")
        print(f"[Solver] Método: {method.upper()}")
        print(f"  k_eff_xy = {self.props.k_eff_xy_WmK:.4f} W/m·K")
        print(f"  h_conv   = {self.props.h_surface_Wm2K:.1f} W/m²·K")
        print(f"  T_amb    = {self.props.T_ambient_C:.1f} °C")
        print(f"  P_total  = {grid.total_power_W():.4f} W")
        print(f"{'='*60}")

        if method == 'direct':
            T = self.solve_direct(grid)
        elif method == 'bicgstab':
            T = self.solve_iterative_bicgstab(grid)
        elif method == 'sor':
            T, _ = self.solve_sor(grid)
        else:
            raise ValueError(f"Método desconocido: {method}. "
                             f"Usar 'direct', 'bicgstab' o 'sor'.")

        print(f"\n[Solver] Resultados:")
        print(f"  T_min = {T.min():.2f}°C")
        print(f"  T_max = {T.max():.2f}°C")
        print(f"  ΔT    = {T.max() - T.min():.2f}°C")
        print(f"  T_mean = {T.mean():.2f}°C")

        return T
