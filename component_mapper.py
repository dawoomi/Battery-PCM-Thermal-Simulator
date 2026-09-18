"""
component_mapper.py
===================
Mapeo de parámetros térmicos de componentes a la grilla de simulación.

Modelo de fuente térmica distribuida:
--------------------------------------
Cada componente se modela como una fuente volumétrica uniforme dentro
de su footprint proyectado en XY.

  q_source [W/m²] = P_component / A_footprint

La densidad de potencia se asigna a las celdas de la grilla que se
superponen con el footprint del componente.

Para componentes pequeños (< 1 celda): se asigna a la celda más cercana
preservando la potencia total (δ-fuente discreta).

Modelo de temperatura de junction:
  T_j = T_pcb_surface + P × θ_JC

donde θ_JC (junction-to-case) < θ_JA (junction-to-ambient).
Esta capa agrega la resistencia interna del encapsulado sobre la
temperatura calculada por el FDM en la superficie de la PCB.

Referencias:
  - JEDEC JESD51-1, JESD51-2: Definición y medición de θ_JA
  - Texas Instruments AN-1710: "Thermal Design by Insight, Not Hindsight"
  - Flotherm Application Notes: "Compact Thermal Models for IC Packages"
  - DELPHI/JEDEC Two-resistor thermal model (Jesd51-14)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from config import PCBConfig
from step_parser import ComponentGeometry


@dataclass
class ComponentThermal:
    """
    Unión de geometría 3D y parámetros térmicos de un componente.
    """
    geometry: ComponentGeometry
    power_W: float = 0.0           # Potencia disipada [W]
    theta_ja_CW: float = 100.0     # Resistencia térmica junction-to-ambient [°C/W]
    theta_jc_CW: float = 10.0      # Resistencia térmica junction-to-case [°C/W]
    description: str = ""

    @property
    def power_density_Wm2(self) -> float:
        """Densidad de potencia sobre el footprint [W/m²]"""
        area_m2 = self.geometry.footprint_area_mm2 * 1e-6
        if area_m2 < 1e-12:
            return 0.0
        return self.power_W / area_m2

    @property
    def t_rise_adiabatic_C(self) -> float:
        """
        Elevación de temperatura máxima teórica (θ_JA completo).
        T_j - T_amb = P × θ_JA
        """
        return self.power_W * self.theta_ja_CW


@dataclass
class ThermalGrid:
    """
    Grilla 2D de densidad de potencia y temperaturas.

    La grilla cubre exactamente las dimensiones de la PCB.
    Cada celda [i,j] representa un área dx×dy de la PCB.
    """
    nx: int
    ny: int
    pcb_width_mm: float
    pcb_height_mm: float

    # Arrays principales
    power_density: np.ndarray = field(default=None)  # [W/m²]
    temperature: np.ndarray = field(default=None)     # [°C]
    component_mask: np.ndarray = field(default=None)  # ID componente por celda

    def __post_init__(self):
        self.power_density = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.temperature = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.component_mask = np.full((self.ny, self.nx), -1, dtype=np.int32)

    @property
    def dx_mm(self) -> float:
        return self.pcb_width_mm / self.nx

    @property
    def dy_mm(self) -> float:
        return self.pcb_height_mm / self.ny

    @property
    def dx_m(self) -> float:
        return self.dx_mm * 1e-3

    @property
    def dy_m(self) -> float:
        return self.dy_mm * 1e-3

    def total_power_W(self) -> float:
        """Potencia total integrada sobre la grilla [W]"""
        cell_area_m2 = self.dx_m * self.dy_m
        return float(np.sum(self.power_density) * cell_area_m2)


def load_thermal_params(csv_path: str) -> pd.DataFrame:
    """
    Carga parámetros térmicos desde CSV o Excel.

    Formato CSV esperado:
      Component, Power_W, Theta_JA, Theta_JC, Description
      U1, 0.8, 40, 5, Buck regulator
      L1, 0.4, 30, 8, Inductor
      ...

    Theta_JC es opcional. Si no está, se estima como θ_JA / 8
    (relación empírica aproximada para packages típicos).

    Args:
        csv_path: Ruta al archivo CSV o XLSX

    Returns:
        DataFrame con las columnas estandarizadas.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Archivo de parámetros térmicos no encontrado: {csv_path}")

    if path.suffix.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(csv_path)
    else:
        # Detección automática de separador decimal (punto vs coma)
        # Excel español/Windows exporta numeros como "0,006" (entre comillas, decimal=,)
        import re as _re
        raw = open(csv_path, 'r', encoding='utf-8-sig', errors='replace').read()
        has_quoted_comma = bool(_re.search(r'"\d+,\d+"', raw))
        decimal_char = ',' if has_quoted_comma else '.'
        df = pd.read_csv(csv_path, skipinitialspace=True,
                         decimal=decimal_char, encoding='utf-8-sig')
        if decimal_char == ',':
            print("  [INFO] CSV con decimal=',' detectado (Excel español/europeo).")

    # Normalizar nombres de columnas
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

    required_cols = {'component', 'power_w', 'theta_ja'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Columnas requeridas faltantes: {missing}\n"
                         f"Columnas encontradas: {list(df.columns)}")

    # Theta_JC: si no existe, estimar como θ_JA / 8
    if 'theta_jc' not in df.columns:
        df['theta_jc'] = df['theta_ja'] / 8.0

    if 'description' not in df.columns:
        df['description'] = ''

    # Limpiar y validar
    # --- FIX: soporte separador decimal coma (locale europeo / Windows español) ---
    # pandas.to_numeric falla silenciosamente con "1,5" → NaN.
    # Normalizamos reemplazando coma decimal por punto en columnas numéricas.
    def _fix_decimal(series: pd.Series) -> pd.Series:
        """Reemplaza coma decimal por punto si el valor es string."""
        if series.dtype == object:
            series = series.astype(str).str.strip()
            # Solo reemplazar si el patrón es dígito,dígito (no separador de miles)
            series = series.str.replace(r'(?<=\d),(?=\d)', '.', regex=True)
        return series

    df['component'] = df['component'].astype(str).str.strip()
    df['power_w']   = pd.to_numeric(_fix_decimal(df['power_w']),   errors='coerce').fillna(0.0)
    df['theta_ja']  = pd.to_numeric(_fix_decimal(df['theta_ja']),  errors='coerce').fillna(100.0)
    df['theta_jc']  = pd.to_numeric(_fix_decimal(df['theta_jc']),  errors='coerce').fillna(10.0)

    # Avisar si quedaron ceros que venían como strings no parseables
    n_zero_power = (df['power_w'] == 0.0).sum()
    if n_zero_power == len(df):
        print(f"  [WARN] Todos los valores Power_W resultaron 0. "
              f"Verificar separador decimal en el CSV (usar punto '.' o coma ',').")

    print(f"[ComponentMapper] Parámetros térmicos cargados: {len(df)} componentes")
    print(f"  Potencia total declarada: {df['power_w'].sum():.3f} W")
    print(f"  Rango θ_JA: {df['theta_ja'].min():.0f}–{df['theta_ja'].max():.0f} °C/W")

    return df


def match_components(geometries: List[ComponentGeometry],
                     thermal_df: pd.DataFrame,
                     verbose: bool = True
                     ) -> List[ComponentThermal]:
    """
    Empareja geometrías (del STEP) con parámetros térmicos (del CSV).

    Estrategia de matching (en orden de prioridad):
      1. Match exacto por nombre (case-insensitive)
      2. Match parcial: el nombre del STEP contiene el nombre del CSV
      3. Match parcial inverso: el nombre del CSV contiene el del STEP
      4. Sin match: componente con P=0 (solo geometría)

    Args:
        geometries: Lista de ComponentGeometry del parser STEP
        thermal_df: DataFrame con parámetros térmicos

    Returns:
        Lista de ComponentThermal (fusión de geometría + térmica)
    """
    result = []
    matched = set()

    # Crear lookup dict case-insensitive
    thermal_lookup = {
        row['component'].lower(): row
        for _, row in thermal_df.iterrows()
    }

    for geom in geometries:
        name_lower = geom.name.lower()
        matched_row = None

        # Match exacto
        if name_lower in thermal_lookup:
            matched_row = thermal_lookup[name_lower]
        else:
            # Match parcial
            for key, row in thermal_lookup.items():
                if key in name_lower or name_lower in key:
                    matched_row = row
                    break

        if matched_row is not None:
            ct = ComponentThermal(
                geometry=geom,
                power_W=float(matched_row['power_w']),
                theta_ja_CW=float(matched_row['theta_ja']),
                theta_jc_CW=float(matched_row['theta_jc']),
                description=str(matched_row.get('description', '')),
            )
            matched.add(matched_row['component'])
        else:
            ct = ComponentThermal(geometry=geom, power_W=0.0)
            if verbose:
                print(f"  [WARN] Sin datos térmicos para: '{geom.name}' (P=0 W asumido)")

        result.append(ct)

    unmatched_thermal = set(thermal_df['component']) - matched
    if unmatched_thermal and verbose:
        print(f"  [WARN] Componentes en CSV sin geometría: {unmatched_thermal}")

    total_P = sum(c.power_W for c in result)
    print(f"[ComponentMapper] Matching completado: {len(result)} componentes, "
          f"P_total={total_P:.3f} W")

    return result


def build_power_grid(components: List[ComponentThermal],
                     cfg: PCBConfig) -> ThermalGrid:
    """
    Construye la grilla 2D de densidad de potencia térmica [W/m²].

    Para cada componente:
      1. Convertir footprint de mm a índices de grilla
      2. Calcular densidad de potencia: q = P / (A_footprint_m2)
      3. Asignar q a todas las celdas dentro del footprint

    Manejo de componentes sub-grilla (footprint < 1 celda):
      Se asigna la potencia como densidad en la celda que contiene el centroide,
      preservando la potencia total: q_cell = P / (dx*dy)

    Args:
        components: Lista de ComponentThermal
        cfg: PCBConfig con dimensiones de grilla

    Returns:
        ThermalGrid con power_density [W/m²] poblado.
    """
    grid = ThermalGrid(
        nx=cfg.grid_nx,
        ny=cfg.grid_ny,
        pcb_width_mm=cfg.pcb_width_mm,
        pcb_height_mm=cfg.pcb_height_mm,
    )

    dx_mm = grid.dx_mm
    dy_mm = grid.dy_mm
    cell_area_m2 = grid.dx_m * grid.dy_m

    for idx, comp in enumerate(components):
        if comp.power_W <= 0.0:
            continue

        geom = comp.geometry

        # Clip al footprint de la PCB
        x0 = max(0.0, geom.x_min)
        x1 = min(cfg.pcb_width_mm, geom.x_max)
        y0 = max(0.0, geom.y_min)
        y1 = min(cfg.pcb_height_mm, geom.y_max)

        if x1 <= x0 or y1 <= y0:
            # Preserve declared power even when malformed geometry lies outside the board.
            ix_c = int(np.clip(geom.cx / dx_mm, 0, cfg.grid_nx - 1))
            iy_c = int(np.clip(geom.cy / dy_mm, 0, cfg.grid_ny - 1))
            grid.power_density[iy_c, ix_c] += comp.power_W / cell_area_m2
            grid.component_mask[iy_c, ix_c] = idx
            continue

        # Convertir a índices de grilla (floor para inicio, ceil para fin)
        ix0 = int(np.floor(x0 / dx_mm))
        ix1 = int(np.ceil(x1 / dx_mm))
        iy0 = int(np.floor(y0 / dy_mm))
        iy1 = int(np.ceil(y1 / dy_mm))

        # Clamp a límites de la grilla
        ix0 = max(0, min(ix0, cfg.grid_nx - 1))
        ix1 = max(0, min(ix1, cfg.grid_nx))
        iy0 = max(0, min(iy0, cfg.grid_ny - 1))
        iy1 = max(0, min(iy1, cfg.grid_ny))

        # Distribute uniform footprint power by exact cell-overlap area.  Assigning
        # the full footprint density to every floor/ceil-selected cell overcounts
        # power whenever a footprint boundary cuts through a cell.
        area_footprint_m2 = (x1 - x0) * (y1 - y0) * 1e-6
        q_density = comp.power_W / area_footprint_m2  # [W/m²]

        for iy in range(iy0, iy1):
            cell_y0 = iy * dy_mm
            cell_y1 = cell_y0 + dy_mm
            overlap_y_mm = max(0.0, min(y1, cell_y1) - max(y0, cell_y0))

            for ix in range(ix0, ix1):
                cell_x0 = ix * dx_mm
                cell_x1 = cell_x0 + dx_mm
                overlap_x_mm = max(0.0, min(x1, cell_x1) - max(x0, cell_x0))
                overlap_area_m2 = overlap_x_mm * overlap_y_mm * 1e-6

                if overlap_area_m2 <= 0.0:
                    continue

                grid.power_density[iy, ix] += (
                    q_density * overlap_area_m2 / cell_area_m2
                )
                grid.component_mask[iy, ix] = idx

    integrated_power = grid.total_power_W()
    total_input_power = sum(c.power_W for c in components)

    print(f"[ComponentMapper] Grilla de potencia generada:")
    print(f"  Potencia total entrada:    {total_input_power:.4f} W")
    print(f"  Potencia integrada grilla: {integrated_power:.4f} W")
    print(f"  Error de discretización:   {abs(integrated_power - total_input_power)/max(total_input_power, 1e-9)*100:.2f}%")
    print(f"  q_max: {grid.power_density.max():.2f} W/m² | "
          f"q_mean (activo): {grid.power_density[grid.power_density>0].mean():.2f} W/m²"
          if grid.power_density.max() > 0 else "")

    return grid


def generate_demo_thermal_csv(output_path: str = "thermal_params.csv"):
    """
    Genera un CSV de ejemplo para demostración.
    Los valores de θ_JA son representativos de datasheets reales.
    """
    data = {
        'Component': ['U1_BuckReg', 'U2_RFTrx', 'U3_MCU', 'L1_Inductor',
                      'L2_Inductor', 'Q1_MOSFET', 'Q2_MOSFET', 'U4_LDO', 'XTAL_32M'],
        'Power_W':   [0.85,         0.35,        0.25,      0.40,
                      0.35,          0.55,         0.50,       0.20,  0.05],
        'Theta_JA':  [38,            45,           52,         30,
                      30,             25,            25,         60,   200],
        'Theta_JC':  [5,             8,             6,         None,
                      None,           3,             3,          10,   None],
        'Description': [
            'Synchronous Buck Regulator 3A',
            'Sub-GHz RF Transceiver',
            'ARM Cortex-M4 @ 168MHz',
            'Power inductor 10uH',
            'Power inductor 4.7uH',
            'N-Ch MOSFET load switch',
            'N-Ch MOSFET load switch',
            'LDO 3.3V 500mA',
            '32MHz TCXO Crystal',
        ]
    }
    df = pd.DataFrame(data)
    # Theta_JC estimada si no hay dato
    df['Theta_JC'] = df['Theta_JC'].fillna(df['Theta_JA'] / 8.0)
    df.to_csv(output_path, index=False)
    print(f"[Demo] CSV térmico generado: {output_path}")
    return output_path
