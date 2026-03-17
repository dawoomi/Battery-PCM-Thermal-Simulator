"""
material_properties.py
======================
Cálculo de propiedades térmicas efectivas de la PCB.

Modelo de conductividad efectiva (k_eff):
------------------------------------------
La PCB es un material compuesto anisótropo.

  In-plane (X,Y): las capas de cobre y FR4 están en paralelo térmico.
    k_eff_xy = (k_cu * t_cu_total + k_fr4 * t_fr4) / t_total

  Through-plane (Z): capas en serie térmico.
    1/k_eff_z = (t_cu_total/k_cu + t_fr4/k_fr4) / t_total

Para la simulación 2D in-plane usamos k_eff_xy.

Referencias:
  - Culham & Muzychka, "Optimization of Plate Fin Heat Sinks Using Entropy Generation Minimization"
  - IPC-2152 Appendix A: Thermal conductivity of multilayer PCBs
  - Simons, R.E., "Thermal Management of Electronics in Telecommunications Equipment"
    Electronics Cooling, Vol 10, No 2, 2004.
  - JEDEC JESD51-1: "Integrated Circuits Thermal Measurement Method – Electrical Test Method"
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict
from config import PCBConfig


# ─── Biblioteca de materiales comunes en PCB ──────────────────────────────────

MATERIAL_DATABASE: Dict[str, Dict] = {
    "FR4": {
        "k_WmK": 0.30,          # Conductividad térmica in-plane [W/m·K]
        "k_z_WmK": 0.29,        # Through-plane [W/m·K]
        "density_kgm3": 1850.0,  # Densidad [kg/m³]
        "Cp_JkgK": 1100.0,       # Calor específico [J/kg·K]
        "Tg_C": 130.0,           # Temperatura de transición vítrea [°C]
    },
    "Copper": {
        "k_WmK": 385.0,
        "k_z_WmK": 385.0,
        "density_kgm3": 8960.0,
        "Cp_JkgK": 385.0,
        "Tg_C": 1085.0,          # Punto de fusión
    },
    "Aluminum": {
        "k_WmK": 205.0,
        "k_z_WmK": 205.0,
        "density_kgm3": 2700.0,
        "Cp_JkgK": 900.0,
        "Tg_C": 660.0,
    },
    "Solder_SAC305": {
        # SAC305: Sn96.5/Ag3/Cu0.5 - estándar RoHS
        "k_WmK": 59.0,
        "k_z_WmK": 59.0,
        "density_kgm3": 7380.0,
        "Cp_JkgK": 230.0,
        "Tg_C": 217.0,
    },
    "Epoxy_Mold_Compound": {
        # Encapsulante típico IC
        "k_WmK": 0.8,
        "k_z_WmK": 0.8,
        "density_kgm3": 1900.0,
        "Cp_JkgK": 900.0,
        "Tg_C": 160.0,
    },
    "Air": {
        "k_WmK": 0.026,
        "k_z_WmK": 0.026,
        "density_kgm3": 1.205,
        "Cp_JkgK": 1006.0,
        "Tg_C": None,
    },
}


@dataclass
class ThermalProperties:
    """
    Propiedades térmicas efectivas calculadas para la PCB.
    Todas las propiedades son efectivas (equivalente del stack-up completo).
    """
    k_eff_xy_WmK: float       # Conductividad efectiva in-plane [W/m·K]
    k_eff_z_WmK: float        # Conductividad efectiva through-plane [W/m·K]
    h_surface_Wm2K: float     # Coef. convección superficial [W/m²·K]
    T_ambient_C: float        # Temperatura ambiente [°C]
    pcb_thickness_m: float    # Espesor total PCB [m]

    @property
    def Bi(self) -> float:
        """
        Número de Biot: Bi = h·L_c / k
        L_c = espesor/2 (placa plana, convección doble cara)
        Bi << 1 → temperatura uniforme en Z (válida aprox. lumped)
        Bi >> 1 → gradiente Z significativo
        """
        L_c = self.pcb_thickness_m / 2.0
        return (self.h_surface_Wm2K * L_c) / self.k_eff_z_WmK

    @property
    def thermal_length_m(self) -> float:
        """
        Longitud característica de difusión térmica [m].
        L_th = sqrt(k_eff * t_pcb / h)
        Indica el radio de influencia de una fuente puntual.
        """
        return np.sqrt(self.k_eff_xy_WmK * self.pcb_thickness_m / self.h_surface_Wm2K)


def compute_effective_conductivity(cfg: PCBConfig) -> ThermalProperties:
    """
    Calcula la conductividad térmica efectiva del stack-up de la PCB.

    Modelo:
      t_cu_total = num_layers × copper_thickness × fill_factor
      t_fr4 = pcb_thickness - t_cu_total

    In-plane (paralelo):
      k_eff_xy = (k_cu × t_cu + k_fr4 × t_fr4) / t_total

    Through-plane (serie):
      R_z = t_cu/k_cu + t_fr4/k_fr4   [m·K/W por unidad de área]
      k_eff_z = t_total / R_z

    Returns:
      ThermalProperties con todos los parámetros necesarios para el solver.
    """
    t_total_m = cfg.pcb_thickness_mm * 1e-3

    # Espesor total de cobre efectivo
    t_cu_m = (cfg.num_copper_layers
               * cfg.copper_thickness_um * 1e-6
               * cfg.copper_fill_factor)

    # Asegurar que t_cu no supere el espesor total (sanity check)
    t_cu_m = min(t_cu_m, 0.85 * t_total_m)
    t_fr4_m = t_total_m - t_cu_m

    k_cu = cfg.k_copper_WmK
    k_fr4 = cfg.k_fr4_WmK

    # Conductividad in-plane: modelo de mezcla en paralelo
    k_eff_xy = (k_cu * t_cu_m + k_fr4 * t_fr4_m) / t_total_m

    # Conductividad through-plane: modelo de resistencias en serie
    # 1/k_eff_z = t_total / (t_cu/k_cu + t_fr4/k_fr4) ... pero invertido:
    R_z = (t_cu_m / k_cu) + (t_fr4_m / k_fr4)   # [m·K/W]
    k_eff_z = t_total_m / R_z if R_z > 0 else k_fr4

    props = ThermalProperties(
        k_eff_xy_WmK=k_eff_xy,
        k_eff_z_WmK=k_eff_z,
        h_surface_Wm2K=cfg.h_convection_Wm2K,
        T_ambient_C=cfg.T_ambient_C,
        pcb_thickness_m=t_total_m,
    )

    print(f"[MaterialProperties] Stack-up thermal analysis:")
    print(f"  t_total  = {t_total_m*1e3:.2f} mm")
    print(f"  t_Cu     = {t_cu_m*1e6:.1f} µm  ({cfg.num_copper_layers} layers × "
          f"{cfg.copper_thickness_um}µm × fill={cfg.copper_fill_factor})")
    print(f"  t_FR4    = {t_fr4_m*1e3:.3f} mm")
    print(f"  k_eff_xy = {k_eff_xy:.4f} W/m·K  (in-plane)")
    print(f"  k_eff_z  = {k_eff_z:.4f} W/m·K  (through-plane)")
    print(f"  Bi       = {props.Bi:.4f}  {'(lumped válido)' if props.Bi < 0.1 else '(gradiente Z no despreciable)'}")
    print(f"  L_th     = {props.thermal_length_m*1e3:.2f} mm  (radio influencia térmica)")

    return props


def junction_temperature(P_W: float, theta_ja_CW: float, T_amb_C: float) -> float:
    """
    Temperatura de junction estimada según modelo JEDEC JESD51.

    T_j = T_amb + P × θ_JA

    donde θ_JA incluye resistencia junction-package + package-ambient.
    Esta es una estimación conservadora (worst case sin PCB spreading).

    Args:
        P_W:        Potencia disipada [W]
        theta_ja_CW: Resistencia térmica junction-to-ambient [°C/W]
        T_amb_C:    Temperatura ambiente [°C]

    Returns:
        T_junction [°C]
    """
    return T_amb_C + P_W * theta_ja_CW


def spreading_resistance(P_W: float,
                         source_area_m2: float,
                         props: ThermalProperties) -> float:
    """
    Resistencia térmica de spreading (Kennedy, 1960).
    Aproximación circular: fuente puntual sobre semiplano conductor.

      R_sp ≈ 1 / (2 × k_eff × sqrt(π × A_s))

    Válida cuando r_source << dimensión PCB.

    Ref: Kennedy, D.P., "Spreading Resistance in Cylindrical Semiconductor Devices",
         J. Appl. Phys., 31(8), 1960.

    Args:
        P_W:            Potencia [W]
        source_area_m2: Área del footprint [m²]
        props:          ThermalProperties

    Returns:
        ΔT por spreading [°C]
    """
    if source_area_m2 <= 0:
        return 0.0
    r_eq = np.sqrt(source_area_m2 / np.pi)  # Radio equivalente
    R_sp = 1.0 / (2.0 * props.k_eff_xy_WmK * r_eq * np.pi)  # Aproximación Carslaw
    return P_W * R_sp
