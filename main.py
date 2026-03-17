"""
main.py
=======
Pipeline principal del simulador térmico de PCB.

Flujo:
  1. Cargar configuración
  2. Calcular propiedades térmicas efectivas del stack-up
  3. Parsear geometría del STEP (o modo demo)
  4. Cargar parámetros térmicos del CSV
  5. Hacer matching componente-geometría
  6. Construir grilla de potencia
  7. Resolver ecuación térmica FDM
  8. Renderizar y exportar resultados

Uso:
  python main.py                         # Modo demo
  python main.py --step pcb.step         # Con STEP real
  python main.py --step pcb.step --csv thermal.csv --method bicgstab
  python main.py --demo --method direct  # Forzar modo demo

Argumentos CLI:
  --step FILE    Archivo STEP del ensamblado
  --csv FILE     CSV/XLS de parámetros térmicos
  --method STR   Solver: 'direct' | 'bicgstab' | 'sor' (default: direct)
  --out FILE     Imagen de salida (default: pcb_thermal_map.png)
  --nx INT       Grilla X (default: 300)
  --ny INT       Grilla Y (default: 240)
  --cmap STR     Colormap matplotlib (default: inferno)
  --h FLOAT      Coef. convección [W/m²K] (default: 8.0)
  --T_amb FLOAT  Temperatura ambiente [°C] (default: 25.0)
  --demo         Forzar modo demo con geometría sintética
  --no-show      No mostrar ventana (solo guardar imagen)
"""

import sys
import argparse
import numpy as np
import time
from pathlib import Path

# ── Importar módulos del proyecto ─────────────────────────────────────────────
from config import PCBConfig
from material_properties import compute_effective_conductivity
from step_parser import STEPParser, generate_synthetic_geometry
from step_parser_text import STEPTextParser
from component_mapper import (
    load_thermal_params,
    match_components,
    build_power_grid,
    generate_demo_thermal_csv,
)
from thermal_solver import FDMThermalSolver
from heatmap_renderer import render_thermal_map, render_junction_temperature_table


def parse_args():
    parser = argparse.ArgumentParser(
        description="PCB Thermal Simulator — FDM 2D Steady-State"
    )
    parser.add_argument('--step',   default=None,  help='Archivo STEP')
    parser.add_argument('--csv',    default=None,  help='CSV de parámetros térmicos')
    parser.add_argument('--method', default='direct',
                        choices=['direct', 'bicgstab', 'sor'],
                        help='Método del solver FDM')
    parser.add_argument('--out',    default='pcb_thermal_map.png',
                        help='Imagen de salida')
    parser.add_argument('--nx',     type=int,   default=300,  help='Grilla X')
    parser.add_argument('--ny',     type=int,   default=240,  help='Grilla Y')
    parser.add_argument('--cmap',   default='inferno', help='Colormap matplotlib')
    parser.add_argument('--h',      type=float, default=8.0,
                        help='Coeficiente de convección [W/m²K]')
    parser.add_argument('--T_amb',  type=float, default=25.0,
                        help='Temperatura ambiente [°C]')
    parser.add_argument('--demo',   action='store_true',
                        help='Forzar modo demo (geometría sintética)')
    parser.add_argument('--no-show', action='store_true',
                        help='No mostrar ventana (solo guardar)')
    return parser.parse_args()


def run_pipeline(cfg: PCBConfig,
                 step_file: str = None,
                 csv_file: str = None,
                 force_demo: bool = False,
                 solver_method: str = 'direct',
                 show: bool = True
                 ) -> dict:
    """
    Pipeline completo de simulación térmica.

    Args:
        cfg:            PCBConfig con todos los parámetros
        step_file:      Ruta al STEP (None → modo demo)
        csv_file:       Ruta al CSV de parámetros térmicos
        force_demo:     Si True, usa geometría sintética
        solver_method:  'direct', 'bicgstab' o 'sor'
        show:           Mostrar ventana matplotlib

    Returns:
        dict con: T (array), grid, components, props, fig
    """
    t_start = time.time()

    print("\n" + "═"*65)
    print("  PCB THERMAL SIMULATOR v1.0")
    print("  FDM 2D Steady-State Conduction + Newton Convection")
    print("═"*65)
    print(f"  {cfg.summary()}")
    print("═"*65 + "\n")

    # ── PASO 1: Propiedades térmicas efectivas ────────────────────────────────
    print("[PASO 1] Calculando propiedades térmicas del stack-up...")
    props = compute_effective_conductivity(cfg)

    # ── PASO 2: Geometría (STEP o demo) ───────────────────────────────────────
    print("\n[PASO 2] Extrayendo geometría...")

    use_demo = force_demo or (step_file is None) or (not Path(step_file).exists())

    if not use_demo:
        # Intentar OCC primero (máxima precisión), fallback a parser de texto puro
        try:
            from OCC.Core.STEPControl import STEPControl_Reader  # test importación
            parser = STEPParser(step_file, verbose=True)
            pcb_geom, geom_list = parser.extract_components()
            if pcb_geom is None or not geom_list:
                raise RuntimeError("OCC no extrajo geometría válida")
            print("  [INFO] Parser: pythonocc-core (OCC)")
        except (ImportError, RuntimeError):
            print("  [INFO] Parser: modo texto (sin pythonocc-core)")
            text_parser = STEPTextParser(step_file, verbose=True)
            pcb_geom, geom_list = text_parser.load()

        if pcb_geom is None or not geom_list:
            print("  [WARN] STEP no produjo geometría válida. Cambiando a modo demo.")
            use_demo = True

    if use_demo:
        print("  [DEMO] Usando geometría sintética representativa...")
        pcb_geom, geom_list = generate_synthetic_geometry(cfg)
        # En modo demo, actualizar cfg con las dimensiones del PCB detectado
        cfg.pcb_width_mm  = pcb_geom.width
        cfg.pcb_height_mm = pcb_geom.height

    # Actualizar cfg con dimensiones reales del PCB detectado
    cfg.pcb_width_mm  = pcb_geom.width
    cfg.pcb_height_mm = pcb_geom.height
    print(f"  PCB: {pcb_geom.width:.1f} × {pcb_geom.height:.1f} mm")
    print(f"  Componentes detectados: {len(geom_list)}")

    # ── PASO 3: Parámetros térmicos ───────────────────────────────────────────
    print("\n[PASO 3] Cargando parámetros térmicos...")

    if csv_file and Path(csv_file).exists():
        thermal_df = load_thermal_params(csv_file)
    else:
        print("  [DEMO] Generando CSV de parámetros térmicos de ejemplo...")
        demo_csv = "thermal_params_demo.csv"
        generate_demo_thermal_csv(demo_csv)
        thermal_df = load_thermal_params(demo_csv)

    # ── PASO 4: Matching componente-geometría ─────────────────────────────────
    print("\n[PASO 4] Emparejando geometría con parámetros térmicos...")
    components = match_components(geom_list, thermal_df)

    total_P = sum(c.power_W for c in components)
    print(f"  Potencia total: {total_P:.3f} W")

    # ── PASO 5: Grilla de potencia ────────────────────────────────────────────
    print("\n[PASO 5] Construyendo grilla de potencia térmica...")
    grid = build_power_grid(components, cfg)

    # ── PASO 6: Solver FDM ────────────────────────────────────────────────────
    print(f"\n[PASO 6] Resolviendo ecuación de conducción FDM...")
    solver = FDMThermalSolver(cfg, props)
    T = solver.solve(grid, method=solver_method)

    # Actualizar temperatura en la grilla
    grid.temperature = T

    # ── PASO 7: Post-proceso — T_junction ─────────────────────────────────────
    print("\n[PASO 7] Calculando temperaturas de junction...")
    tj_df = render_junction_temperature_table(
        components, T, grid, cfg,
        output_path="tj_analysis.png"
    )

    if tj_df is not None and len(tj_df) > 0:
        print("\n  Junction Temperature Summary:")
        print(f"  {'Component':<20} {'P[W]':>6} {'T_surf[°C]':>12} "
              f"{'Tj_FDM[°C]':>12} {'Tj_JEDEC[°C]':>13}")
        print("  " + "─"*65)
        for _, row in tj_df.iterrows():
            status = "⚠️ HOT" if row['T_j_FDM [°C]'] > 85 else "  OK"
            print(f"  {row['Component']:<20} {row['P [W]']:>6.3f} "
                  f"{row['T_surf_FDM [°C]']:>12.1f} "
                  f"{row['T_j_FDM [°C]']:>12.1f} "
                  f"{row['T_j_JEDEC [°C]']:>13.1f}  {status}")

    # ── PASO 8: Renderizar ────────────────────────────────────────────────────
    print(f"\n[PASO 8] Renderizando heatmap...")
    fig = render_thermal_map(
        T=T,
        grid=grid,
        components=components,
        cfg=cfg,
        props=props,
        output_path=cfg.output_filename,
        show=show,
    )

    # ── PASO 9: Exportar datos numéricos ─────────────────────────────────────
    np_out = cfg.output_filename.replace('.png', '_Tmatrix.npy')
    np.save(np_out, T)
    print(f"[PASO 9] Matriz de temperatura exportada: {np_out}")
    print(f"         Shape: {T.shape} | dtype: {T.dtype}")

    t_total = time.time() - t_start
    print(f"\n{'═'*65}")
    print(f"  Pipeline completado en {t_total:.2f}s")
    print(f"  Salidas:")
    print(f"    {cfg.output_filename}    (heatmap)")
    print(f"    tj_analysis.png          (tabla T_junction)")
    print(f"    {np_out}  (matriz numpy)")
    print("═"*65 + "\n")

    return {
        'T': T,
        'grid': grid,
        'components': components,
        'props': props,
        'fig': fig,
        'tj_df': tj_df,
    }


def main():
    args = parse_args()

    # Construir configuración desde argumentos CLI
    cfg = PCBConfig(
        grid_nx=args.nx,
        grid_ny=args.ny,
        colormap=args.cmap,
        h_convection_Wm2K=args.h,
        T_ambient_C=args.T_amb,
        output_filename=args.out,
        step_file=args.step or "pcb_assembly.step",
        thermal_csv=args.csv or "thermal_params.csv",
    )

    results = run_pipeline(
        cfg=cfg,
        step_file=args.step,
        csv_file=args.csv,
        force_demo=args.demo,
        solver_method=args.method,
        show=not args.no_show,
    )

    return results


if __name__ == "__main__":
    main()
