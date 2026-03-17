"""
heatmap_renderer.py
===================
Visualización del campo de temperatura sobre la PCB.

Genera una imagen de salida que combina:
  1. Fondo de la PCB (vista superior esquemática o imagen real si disponible)
  2. Overlay de heatmap con colormap tipo cámara térmica
  3. Footprints de componentes con etiquetas
  4. Barra de color (escala de temperatura)
  5. Anotaciones: T_junction estimada por componente

Colormaps recomendados para análisis térmico:
  - 'inferno':  Negro → violeta → rojo → amarillo (perceptualmente uniforme)
  - 'hot':      Negro → rojo → amarillo → blanco (clásico cámara térmica)
  - 'plasma':   Violeta → rojo → amarillo (perceptualmente uniforme, SOTA)
  - 'RdYlBu_r': Azul → amarillo → rojo (divergente, bueno para comparar)

  NOTA: Evitar 'jet' para publicaciones científicas (no perceptualmente
  uniforme, induce artefactos visuales). Referencia:
  Moreland K., "Diverging Color Maps for Scientific Visualization",
  ISVC 2009.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import FancyBboxPatch, Rectangle
from typing import List, Optional, Tuple
import warnings

from config import PCBConfig
from component_mapper import ComponentThermal, ThermalGrid
from material_properties import ThermalProperties


matplotlib.rcParams.update({
    'font.family': 'monospace',
    'font.size': 8,
    'axes.titlesize': 11,
    'figure.facecolor': '#0a0a0f',
    'axes.facecolor': '#0a0a0f',
    'text.color': '#e8e8e8',
    'axes.labelcolor': '#e8e8e8',
    'xtick.color': '#888888',
    'ytick.color': '#888888',
})


def render_thermal_map(T: np.ndarray,
                       grid: ThermalGrid,
                       components: List[ComponentThermal],
                       cfg: PCBConfig,
                       props: ThermalProperties,
                       output_path: str = None,
                       show: bool = True,
                       T_clamp_percentile: float = 99.0
                       ) -> plt.Figure:
    """
    Renderiza el mapa térmico completo de la PCB.

    Args:
        T:                   Array 2D (ny, nx) de temperaturas [°C]
        grid:                ThermalGrid con metadatos de la grilla
        components:          Lista de ComponentThermal para anotar footprints
        cfg:                 PCBConfig
        props:               ThermalProperties
        output_path:         Ruta de salida de la imagen
        show:                Mostrar ventana matplotlib
        T_clamp_percentile:  Percentil para clamping de T_max visual
                             (evita que un hotspot domine toda la escala)

    Returns:
        fig: matplotlib Figure
    """
    output_path = output_path or cfg.output_filename

    # ── Escala de temperatura ─────────────────────────────────────────────────
    T_min = T.min()
    T_max_data = T.max()
    # Clamping visual: el percentil 99 evita que un solo pixel caliente
    # "queme" toda la escala de colores. El hotspot sigue visible como saturación.
    T_max_visual = np.percentile(T, T_clamp_percentile)
    if T_max_visual <= T_min:
        T_max_visual = T_max_data

    norm = Normalize(vmin=T_min, vmax=T_max_visual)
    cmap = plt.cm.get_cmap(cfg.colormap)

    # ── Layout de la figura ───────────────────────────────────────────────────
    aspect_pcb = cfg.pcb_width_mm / cfg.pcb_height_mm
    fig_w = 14.0
    fig_h = fig_w / aspect_pcb * 0.72 + 3.5  # espacio para info panel

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=cfg.output_dpi // 3)
    fig.patch.set_facecolor('#08080e')

    # GridSpec: mapa principal | panel derecho
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(
        2, 2,
        figure=fig,
        width_ratios=[5, 1.2],
        height_ratios=[4, 1],
        hspace=0.04,
        wspace=0.08,
        left=0.05, right=0.95,
        top=0.93, bottom=0.06
    )

    ax_map   = fig.add_subplot(gs[0, 0])   # Mapa térmico principal
    ax_cbar  = fig.add_subplot(gs[0, 1])   # Colorbar + info
    ax_hist  = fig.add_subplot(gs[1, 0])   # Histograma de temperaturas
    ax_stats = fig.add_subplot(gs[1, 1])   # Estadísticas

    # ── Mapa principal ────────────────────────────────────────────────────────
    _draw_pcb_background(ax_map, cfg)
    _draw_heatmap(ax_map, T, norm, cmap, cfg, alpha=cfg.alpha_heatmap)
    _draw_component_footprints(ax_map, components, T, grid, cfg, props)
    _format_map_axes(ax_map, cfg)

    # ── Colorbar ──────────────────────────────────────────────────────────────
    _draw_colorbar(ax_cbar, norm, cmap, cfg, T_min, T_max_data, T_max_visual)

    # ── Histograma ────────────────────────────────────────────────────────────
    _draw_temperature_histogram(ax_hist, T, norm, cmap, cfg)

    # ── Estadísticas ─────────────────────────────────────────────────────────
    _draw_stats_panel(ax_stats, T, grid, components, cfg, props)

    # ── Título ───────────────────────────────────────────────────────────────
    fig.suptitle(
        f"PCB THERMAL ANALYSIS  ·  {cfg.pcb_width_mm:.0f}×{cfg.pcb_height_mm:.0f} mm  "
        f"·  P_total={grid.total_power_W():.2f}W  "
        f"·  T_amb={cfg.T_ambient_C:.0f}°C  "
        f"·  h={cfg.h_convection_Wm2K:.0f} W/m²K",
        color='#00ff88', fontsize=9, fontweight='bold',
        y=0.97, fontfamily='monospace'
    )

    if output_path:
        fig.savefig(output_path, dpi=cfg.output_dpi, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        print(f"[Renderer] Imagen guardada: {output_path} ({cfg.output_dpi} DPI)")

    if show:
        plt.show()

    return fig


def _draw_pcb_background(ax: plt.Axes, cfg: PCBConfig):
    """Dibuja el fondo esquemático del PCB."""
    # Fondo FR4 verde oscuro
    pcb_rect = FancyBboxPatch(
        (0, 0), cfg.pcb_width_mm, cfg.pcb_height_mm,
        boxstyle="round,pad=0.5",
        facecolor='#1a2f1a',
        edgecolor='#2a5a2a',
        linewidth=1.5,
        zorder=1
    )
    ax.add_patch(pcb_rect)

    # Solder mask texture (grid sutil)
    grid_spacing = max(5.0, cfg.pcb_width_mm / 20)
    for x in np.arange(0, cfg.pcb_width_mm, grid_spacing):
        ax.axvline(x, color='#1f3a1f', linewidth=0.3, alpha=0.4, zorder=2)
    for y in np.arange(0, cfg.pcb_height_mm, grid_spacing):
        ax.axhline(y, color='#1f3a1f', linewidth=0.3, alpha=0.4, zorder=2)

    # Fiduciales (esquinas)
    fid_offset = 3.0
    fid_r = 1.0
    for fx, fy in [(fid_offset, fid_offset),
                   (cfg.pcb_width_mm - fid_offset, fid_offset),
                   (fid_offset, cfg.pcb_height_mm - fid_offset),
                   (cfg.pcb_width_mm - fid_offset, cfg.pcb_height_mm - fid_offset)]:
        circle = plt.Circle((fx, fy), fid_r, color='#c8a000',
                             fill=False, linewidth=0.8, zorder=3)
        ax.add_patch(circle)
        dot = plt.Circle((fx, fy), 0.3, color='#c8a000', zorder=3)
        ax.add_patch(dot)


def _draw_heatmap(ax: plt.Axes, T: np.ndarray,
                  norm: Normalize, cmap,
                  cfg: PCBConfig, alpha: float = 0.75):
    """Superpone el heatmap de temperatura sobre la PCB."""
    # imshow con extent mapeado a mm del PCB
    im = ax.imshow(
        T,
        origin='lower',
        extent=[0, cfg.pcb_width_mm, 0, cfg.pcb_height_mm],
        norm=norm,
        cmap=cmap,
        alpha=alpha,
        interpolation='bilinear',   # bilinear para aspecto suave
        aspect='auto',
        zorder=4
    )
    return im


def _draw_component_footprints(ax: plt.Axes,
                                components: List[ComponentThermal],
                                T: np.ndarray,
                                grid: ThermalGrid,
                                cfg: PCBConfig,
                                props: ThermalProperties):
    """
    Dibuja footprints de componentes con temperatura de junction estimada.
    """
    for comp in components:
        if comp.power_W <= 0.0 and comp.geometry.footprint_area_mm2 < 1.0:
            continue

        g = comp.geometry
        # Color del borde según temperatura relativa
        is_hot = comp.power_W > 0.3

        edge_color = '#ff4444' if is_hot else '#44aaff'
        lw = 1.5 if is_hot else 0.8

        rect = Rectangle(
            (g.x_min, g.y_min), g.width, g.height,
            linewidth=lw,
            edgecolor=edge_color,
            facecolor='none',
            linestyle='--' if is_hot else ':',
            zorder=6
        )
        ax.add_patch(rect)

        # Etiqueta: nombre + potencia + T_junction estimada
        if g.footprint_area_mm2 > 20:  # Solo etiquetar componentes grandes
            # Obtener T de la superficie PCB en el footprint del componente
            ix0 = int(max(0, g.x_min / grid.dx_mm))
            ix1 = int(min(cfg.grid_nx - 1, g.x_max / grid.dx_mm))
            iy0 = int(max(0, g.y_min / grid.dy_mm))
            iy1 = int(min(cfg.grid_ny - 1, g.y_max / grid.dy_mm))

            T_surface_mean = T[iy0:iy1+1, ix0:ix1+1].mean() if iy1 > iy0 else T.mean()
            T_j = T_surface_mean + comp.power_W * comp.theta_jc_CW

            label = (f"{g.name.split('_')[0]}\n"
                     f"{comp.power_W:.2f}W\n"
                     f"Tj≈{T_j:.0f}°C")

            ax.text(
                g.cx, g.cy, label,
                ha='center', va='center',
                fontsize=5.5,
                color='#ffffff',
                fontweight='bold',
                zorder=7,
                bbox=dict(
                    boxstyle='round,pad=0.15',
                    facecolor='#000000',
                    edgecolor='none',
                    alpha=0.55
                )
            )


def _format_map_axes(ax: plt.Axes, cfg: PCBConfig):
    """Formatea los ejes del mapa principal."""
    ax.set_xlim(0, cfg.pcb_width_mm)
    ax.set_ylim(0, cfg.pcb_height_mm)
    ax.set_xlabel("X [mm]", color='#888888', fontsize=8)
    ax.set_ylabel("Y [mm]", color='#888888', fontsize=8)
    ax.set_facecolor('#08080e')
    ax.tick_params(colors='#666666', labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')


def _draw_colorbar(ax: plt.Axes, norm: Normalize, cmap,
                   cfg: PCBConfig,
                   T_min: float, T_max_data: float, T_max_visual: float):
    """Dibuja la barra de color de temperatura estilo cámara térmica."""
    # Crear imagen de gradiente vertical
    gradient = np.linspace(1, 0, 256).reshape(256, 1)
    ax.imshow(gradient, aspect='auto', cmap=cmap, norm=Normalize(0, 1),
              extent=[0, 1, T_min, T_max_visual])

    # Etiquetas
    n_ticks = 8
    ticks = np.linspace(T_min, T_max_visual, n_ticks)
    for tick in ticks:
        ax.axhline(tick, color='#ffffff', linewidth=0.5, alpha=0.4)
        ax.text(1.15, tick, f"{tick:.1f}°C",
                va='center', ha='left', fontsize=7, color='#cccccc')

    ax.set_xlim(0, 1)
    ax.set_ylim(T_min, T_max_visual)
    ax.set_xticks([])
    ax.set_ylabel("Temperatura [°C]", fontsize=8, color='#888888')
    ax.yaxis.set_label_position('right')
    ax.yaxis.tick_right()
    ax.tick_params(axis='y', labelsize=7, colors='#666666')
    ax.set_facecolor('#0a0a12')

    if T_max_data > T_max_visual:
        ax.text(0.5, 1.02,
                f"max={T_max_data:.1f}°C\n(clamped 99p)",
                ha='center', va='bottom', transform=ax.transAxes,
                fontsize=6.5, color='#ff8844')

    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_temperature_histogram(ax: plt.Axes, T: np.ndarray,
                                 norm: Normalize, cmap, cfg: PCBConfig):
    """Histograma de distribución de temperaturas."""
    T_flat = T.ravel()
    n_bins = 80
    counts, bin_edges = np.histogram(T_flat, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Colorear barras según temperatura
    for i, (bc, count) in enumerate(zip(bin_centers, counts)):
        color = cmap(norm(bc))
        ax.bar(bc, count, width=(bin_edges[1]-bin_edges[0]),
               color=color, alpha=0.85, linewidth=0)

    ax.set_xlabel("Temperatura [°C]", fontsize=7, color='#888888')
    ax.set_ylabel("Celdas", fontsize=7, color='#888888')
    ax.set_facecolor('#070710')
    ax.tick_params(colors='#555555', labelsize=6)
    for spine in ax.spines.values():
        spine.set_edgecolor('#222222')

    # Línea de T_amb
    ax.axvline(cfg.T_ambient_C, color='#00ff88', linewidth=1.0,
               linestyle='--', alpha=0.7, label=f'T_amb={cfg.T_ambient_C}°C')
    ax.legend(fontsize=6, facecolor='#111118', edgecolor='#333333',
              labelcolor='#aaaaaa')


def _draw_stats_panel(ax: plt.Axes, T: np.ndarray,
                       grid: ThermalGrid,
                       components: List[ComponentThermal],
                       cfg: PCBConfig,
                       props: ThermalProperties):
    """Panel de estadísticas numéricas."""
    ax.set_facecolor('#060610')
    ax.axis('off')

    total_P = grid.total_power_W()
    # Eficiencia de disipación: P_conv / P_input
    # En estado estacionario P_conv = P_input (todo lo que entra sale por convección)
    T_rise = T.mean() - cfg.T_ambient_C
    area_m2 = (cfg.pcb_width_mm * cfg.pcb_height_mm) * 1e-6
    P_conv_est = cfg.h_convection_Wm2K * area_m2 * T_rise * 2  # 2 caras

    stats_lines = [
        ("T_min",    f"{T.min():.2f}°C"),
        ("T_max",    f"{T.max():.2f}°C"),
        ("T_mean",   f"{T.mean():.2f}°C"),
        ("ΔT",       f"{T.max()-T.min():.2f}°C"),
        ("P_total",  f"{total_P:.3f} W"),
        ("P_conv≈",  f"{P_conv_est:.3f} W"),
        ("h_conv",   f"{cfg.h_convection_Wm2K} W/m²K"),
        ("k_eff",    f"{props.k_eff_xy_WmK:.3f} W/mK"),
        ("Grid",     f"{cfg.grid_nx}×{cfg.grid_ny}"),
        ("Bi_num",   f"{props.Bi:.4f}"),
    ]

    y_pos = 0.98
    for label, value in stats_lines:
        ax.text(0.05, y_pos, f"{label}:", transform=ax.transAxes,
                fontsize=7, color='#777777', va='top', fontfamily='monospace')
        ax.text(0.95, y_pos, value, transform=ax.transAxes,
                fontsize=7, color='#00ff88', va='top', ha='right',
                fontweight='bold', fontfamily='monospace')
        y_pos -= 0.095


def render_junction_temperature_table(components: List[ComponentThermal],
                                       T: np.ndarray,
                                       grid: ThermalGrid,
                                       cfg: PCBConfig,
                                       output_path: str = "tj_table.png"):
    """
    Genera una tabla separada con las temperaturas de junction estimadas
    para todos los componentes.

    T_j = T_pcb_surface (del FDM) + P × θ_JC

    Esta es la temperatura de junction estimada considerando:
      - La distribución térmica en la PCB (del solver FDM)
      - La resistencia junction-to-case del encapsulado (θ_JC del datasheet)

    Es más precisa que T_j = T_amb + P×θ_JA porque usa la temperatura
    real de la superficie de la PCB como punto de referencia.
    """
    rows = []
    for comp in sorted(components, key=lambda c: c.power_W, reverse=True):
        if comp.power_W <= 0:
            continue
        g = comp.geometry
        ix0 = int(max(0, g.x_min / grid.dx_mm))
        ix1 = int(min(cfg.grid_nx - 1, g.x_max / grid.dx_mm))
        iy0 = int(max(0, g.y_min / grid.dy_mm))
        iy1 = int(min(cfg.grid_ny - 1, g.y_max / grid.dy_mm))

        if iy1 > iy0 and ix1 > ix0:
            T_surf = T[iy0:iy1+1, ix0:ix1+1].max()
        else:
            T_surf = T.mean()

        T_j = T_surf + comp.power_W * comp.theta_jc_CW
        T_j_jedec = cfg.T_ambient_C + comp.power_W * comp.theta_ja_CW

        rows.append({
            'Component': g.name,
            'P [W]': comp.power_W,
            'θ_JA [°C/W]': comp.theta_ja_CW,
            'θ_JC [°C/W]': comp.theta_jc_CW,
            'T_surf_FDM [°C]': round(T_surf, 1),
            'T_j_FDM [°C]': round(T_j, 1),
            'T_j_JEDEC [°C]': round(T_j_jedec, 1),
            'Δ(FDM-JEDEC) [°C]': round(T_j - T_j_jedec, 1),
        })

    import pandas as pd
    df = pd.DataFrame(rows)

    # --- FIX: si no hay componentes con P>0, retornar DataFrame vacío sin crash ---
    if df.empty:
        print("  [WARN] render_junction_temperature_table: ningún componente con P>0. "
              "Verificar que Power_W en el CSV tenga valores distintos de cero.")
        return df

    # Render como tabla matplotlib
    fig, ax = plt.subplots(figsize=(14, max(3, len(rows) * 0.5 + 1.5)),
                            facecolor='#080810')
    ax.set_facecolor('#080810')
    ax.axis('off')

    colors_header = ['#1a3a5c'] * len(df.columns)
    cell_colors = []
    for i, row in df.iterrows():
        T_j = row['T_j_FDM [°C]']
        if T_j > 125:
            c = '#3a0808'  # Rojo: sobre límite típico JEDEC
        elif T_j > 85:
            c = '#3a2808'  # Amarillo: región de warning
        else:
            c = '#0a1a0a'  # Verde: OK
        cell_colors.append([c] * len(df.columns))

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        cellColours=cell_colors,
        colColours=colors_header,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    for (row, col), cell in table.get_celld().items():
        cell.set_text_props(color='#cccccc')
        cell.set_edgecolor('#222233')
        if row == 0:
            cell.set_text_props(color='#88ccff', fontweight='bold')

    fig.suptitle("Junction Temperature Analysis — FDM vs JEDEC θ_JA Model",
                 color='#00ff88', fontsize=10, fontfamily='monospace', y=0.98)

    fig.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='#080810')
    print(f"[Renderer] Tabla Tj guardada: {output_path}")
    plt.close(fig)

    return df
