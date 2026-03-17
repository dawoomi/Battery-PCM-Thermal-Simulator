"""
config.py
=========
Parámetros globales de la PCB y configuración de la simulación térmica.

Modelo físico de referencia:
  - Conducción: Fourier, ∇·(k∇T) + q = 0
  - Convección superficial: q_conv = h·A·(T - T_amb)
  - k_eff calculada por regla de mezcla en dirección Z (in-plane vs through-plane)

Referencias:
  - IPC-2152: "Standard for Determining Current Carrying Capacity in Printed Board Design"
  - Bergman et al., "Fundamentals of Heat and Mass Transfer", 7th Ed.
  - AN2826 STMicroelectronics: "Thermal management for power devices"
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class PCBConfig:
    """
    Parámetros físicos y geométricos de la PCB.

    Propiedades térmicas FR4:
      k_xy  ≈ 0.3  W/m·K  (in-plane, dominado por fibra de vidrio)
      k_z   ≈ 0.29 W/m·K  (through-plane, similar al in-plane en FR4 estándar)

    Propiedades térmicas Cu (IPC-2152):
      k_cu  ≈ 385  W/m·K

    El k_eff in-plane se estima como mezcla volumétrica:
      k_eff = (k_fr4 * V_fr4 + k_cu * V_cu) / (V_fr4 + V_cu)
    donde V depende del copper fill factor.

    Convección natural en aire quieto:
      h ≈ 5–10 W/m²·K  (superficies horizontales sin forzado)
    Convección forzada:
      h ≈ 15–50 W/m²·K

    Temperatura de referencia:
      T_amb = 25°C (estándar JEDEC JESD51)
    """

    # --- Geometría ---
    pcb_width_mm: float = 100.0          # Ancho en X [mm]
    pcb_height_mm: float = 80.0          # Alto en Y [mm]
    pcb_thickness_mm: float = 1.6        # Espesor estándar IPC clase 2 [mm]

    # --- Stack-up de cobre ---
    num_copper_layers: int = 2           # 1, 2, 4, 6, 8 capas típicas
    copper_thickness_um: float = 35.0    # 1 oz/ft² ≈ 35 µm (estándar)
    copper_fill_factor: float = 0.35     # Fracción volumétrica de cobre (0–1)
                                         # 0.35 ≈ PCB mixed signal típica

    # --- Propiedades térmicas de materiales ---
    k_fr4_WmK: float = 0.3              # W/m·K - Conducción térmica FR4 (in-plane)
    k_copper_WmK: float = 385.0         # W/m·K - Conducción térmica cobre
    # k_eff se calcula en runtime (ver material_properties.py)

    # --- Condiciones de contorno ---
    h_convection_Wm2K: float = 8.0      # Coeficiente convección natural [W/m²·K]
                                         # Rango: 5–15 natural, 15–50 forzado
    T_ambient_C: float = 25.0           # Temperatura ambiente JEDEC [°C]

    # --- Grilla de simulación ---
    grid_nx: int = 300                  # Nodos en X
    grid_ny: int = 240                  # Nodos en Y
    # Resolución efectiva = pcb_width / grid_nx [mm/celda]

    # --- Solver ---
    solver_max_iter: int = 5000         # Máx iteraciones Gauss-Seidel
    solver_tolerance: float = 1e-4      # Criterio de convergencia [°C]
    solver_omega: float = 1.8           # Factor SOR (Successive Over-Relaxation)
                                         # ω=1 → Gauss-Seidel; 1<ω<2 → SOR
                                         # Óptimo teórico ≈ 2/(1+sin(π/N))

    # --- Visualización ---
    colormap: str = "inferno"           # matplotlib cmap: inferno, hot, plasma, jet
    output_dpi: int = 300               # DPI para exportación
    output_filename: str = "pcb_thermal_map.png"
    alpha_heatmap: float = 0.72         # Transparencia overlay heatmap

    # --- Paths ---
    step_file: str = "pcb_assembly.step"
    thermal_csv: str = "thermal_params.csv"

    @property
    def dx_m(self) -> float:
        """Tamaño de celda en X [metros]"""
        return (self.pcb_width_mm * 1e-3) / self.grid_nx

    @property
    def dy_m(self) -> float:
        """Tamaño de celda en Y [metros]"""
        return (self.pcb_height_mm * 1e-3) / self.grid_ny

    @property
    def pcb_area_m2(self) -> float:
        return (self.pcb_width_mm * 1e-3) * (self.pcb_height_mm * 1e-3)

    def summary(self) -> str:
        res_x = (self.pcb_width_mm / self.grid_nx)
        res_y = (self.pcb_height_mm / self.grid_ny)
        return (
            f"PCB: {self.pcb_width_mm}×{self.pcb_height_mm} mm | "
            f"Grid: {self.grid_nx}×{self.grid_ny} ({res_x:.2f}×{res_y:.2f} mm/cell) | "
            f"h={self.h_convection_Wm2K} W/m²K | T_amb={self.T_ambient_C}°C"
        )
