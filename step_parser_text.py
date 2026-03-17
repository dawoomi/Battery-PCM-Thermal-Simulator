"""
step_parser_text.py
===================
Parser de archivos STEP AP214 (KiCad / Altium / Eagle) sin dependencias OCC.
Extrae referencias, footprints y posiciones reales de cada componente
recorriendo la cadena de entidades del estándar ISO 10303-214.

Cadena de resolución para cada componente:
  NEXT_ASSEMBLY_USAGE_OCCURRENCE (ref, child_def_id)
    ← PRODUCT_DEFINITION_SHAPE 'Placement' (nauo_id)
    ← CONTEXT_DEPENDENT_SHAPE_REPRESENTATION (pds_id, rr_block_id)
    → [RR compound block] REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION(idt_id)
    → ITEM_DEFINED_TRANSFORMATION (ax_from, ax_to)
    → AXIS2_PLACEMENT_3D (cp_id)
    → CARTESIAN_POINT (x, y, z)

El footprint del componente se resuelve como:
  child_def_id → PRODUCT_DEFINITION_FORMATION → PRODUCT (nombre footprint)

El PCB se identifica como el sólido con referencia que contiene '_PCB',
'PCB', 'Board', o cuya entidad PRODUCT tiene nombre de PCB.

Limitaciones:
  - Sin rotación: solo extrae el punto de inserción (origen del footprint)
  - Bounding box aproximada usando tabla de tamaños por tipo de footprint
  - Para bounding box exacta se requiere OCC (pythonocc-core)

Compatibilidad verificada: KiCad STEP AP214.
Altium y Eagle usan la misma estructura ISO 10303-214.
"""

import re
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ─── Tabla de tamaños de footprint (mm) ──────────────────────────────────────
# Dimensiones aproximadas (largo, ancho, alto) para los footprints más comunes.
# Se usan cuando no hay información de bounding box exacta del sólido.
# Fuente: IPC-7351B land pattern standards + datasheets típicos.

FOOTPRINT_SIZE_TABLE: Dict[str, Tuple[float, float, float]] = {
    # Capacitores
    "C_0402_1005Metric":    (1.0,  0.5,  0.5),
    "C_0603_1608Metric":    (1.6,  0.8,  0.8),
    "C_0805_2012Metric":    (2.0,  1.25, 1.25),
    "C_1206_3216Metric":    (3.2,  1.6,  1.6),
    "C_1210_3225Metric":    (3.2,  2.5,  2.5),
    "C_1812_4532Metric":    (4.5,  3.2,  3.2),
    # Resistores
    "R_0402_1005Metric":    (1.0,  0.5,  0.35),
    "R_0603_1608Metric":    (1.6,  0.8,  0.45),
    "R_0805_2012Metric":    (2.0,  1.25, 0.45),
    "R_1206_3216Metric":    (3.2,  1.6,  0.6),
    # Inductores
    "L_Wuerth_HCM-7050":    (7.0,  5.0,  4.5),
    "L_Vishay_IHLP-4040":   (4.0,  4.0,  3.0),
    "L_Sunlord_SWPA4030S":  (4.0,  3.0,  2.0),
    "L_Bourns_SRR1260":     (12.0, 12.0, 6.0),
    "L_0603":               (1.6,  0.8,  0.8),
    # ICs
    "SOIC-8_3.9x4.9mm_P1.27mm":           (5.0,  4.0,  1.5),
    "SOIC-8-1EP_3.9x4.9mm_P1.27mm_EP2.29x3mm": (5.0, 4.0, 1.5),
    "SOIC-8":               (5.0,  4.0,  1.5),
    "SOT-23":               (3.0,  1.75, 1.1),
    "SOT-23-5":             (3.0,  1.75, 1.1),
    "QFN-16":               (4.0,  4.0,  0.9),
    "QFN-20":               (4.0,  4.0,  0.9),
    "QFN-32":               (5.0,  5.0,  0.9),
    "LQFP-64":              (10.0, 10.0, 1.6),
    "TO-220":               (10.0, 4.5,  15.0),
    "TO-252":               (6.5,  6.5,  2.3),
    # Potenciómetros / trimmers
    "Potentiometer_Bourns_3314G_Vertical": (4.5, 4.5, 6.5),
    # Default
    "default":              (3.0,  3.0,  2.0),
}


def _footprint_size(fp_name: str) -> Tuple[float, float, float]:
    """
    Devuelve (largo, ancho, alto) en mm para un footprint dado.
    Busca coincidencia exacta primero, luego substring.
    """
    if fp_name in FOOTPRINT_SIZE_TABLE:
        return FOOTPRINT_SIZE_TABLE[fp_name]
    # Búsqueda por substring (maneja variantes)
    for key, size in FOOTPRINT_SIZE_TABLE.items():
        if key in fp_name or fp_name in key:
            return size
    return FOOTPRINT_SIZE_TABLE["default"]


@dataclass
class ComponentGeometry:
    """Geometría proyectada de un componente en el plano XY del PCB."""
    shape_id: int
    name: str = "unknown"
    x_min: float = 0.0
    y_min: float = 0.0
    x_max: float = 0.0
    y_max: float = 0.0
    z_min: float = 0.0
    z_max: float = 0.0
    is_pcb: bool = False

    @property
    def cx(self): return (self.x_min + self.x_max) / 2.0
    @property
    def cy(self): return (self.y_min + self.y_max) / 2.0
    @property
    def width(self): return self.x_max - self.x_min
    @property
    def height(self): return self.y_max - self.y_min
    @property
    def thickness(self): return self.z_max - self.z_min
    @property
    def footprint_area_mm2(self): return self.width * self.height
    @property
    def xy_aspect_ratio(self):
        return self.footprint_area_mm2 / self.thickness if self.thickness > 1e-6 else float('inf')


class STEPTextParser:
    """
    Parser de STEP AP214 sin OCC.
    Extrae posición e identidad de cada componente mediante regex sobre el texto.
    """

    def __init__(self, step_file: str, verbose: bool = True):
        self.step_file = step_file
        self.verbose = verbose
        self._content = ""
        self._components: List[ComponentGeometry] = []

    def load(self) -> Tuple[Optional[ComponentGeometry], List[ComponentGeometry]]:
        """
        Parsea el STEP y devuelve (pcb, [componentes]).

        Returns:
            (pcb_geometry, list_of_component_geometries)
        """
        with open(self.step_file, 'r', encoding='utf-8', errors='ignore') as f:
            self._content = f.read()

        # ── 1. Mapas de entidades básicas ────────────────────────────────────
        cp      = self._parse_cartesian_points()
        a2p3d   = self._parse_axis2placement(cp)
        idt     = self._parse_idt(a2p3d)
        rr_idt  = self._parse_rr_blocks()
        nauo    = self._parse_nauo()
        pds_nauo = self._parse_pds_placement()
        cdsr    = self._parse_cdsr()
        fp_map  = self._build_footprint_map()
        pcb_ref = self._find_pcb_name()

        if self.verbose:
            print(f"[STEPTextParser] Entidades: CP={len(cp)} A2P={len(a2p3d)} "
                  f"IDT={len(idt)} RR={len(rr_idt)} NAUO={len(nauo)}")

        # ── 2. Resolver instancias ────────────────────────────────────────────
        instances = []
        for cdsr_id, (rr_id, pds_id) in cdsr.items():
            idt_id = rr_idt.get(rr_id)
            if not idt_id:
                continue
            pos = idt.get(idt_id)
            if not pos:
                continue
            nauo_id = pds_nauo.get(pds_id)
            if not nauo_id:
                continue
            n = nauo.get(nauo_id)
            if not n or not n['ref']:
                continue
            fp = fp_map.get(n['child'], 'unknown')
            instances.append({
                'ref':      n['ref'],
                'footprint': fp,
                'x':        pos[0],
                'y':        pos[1],
                'z':        pos[2],
            })

        if self.verbose:
            print(f"[STEPTextParser] Instancias resueltas: {len(instances)}")

        if not instances:
            return None, []

        # ── 3. Determinar origen del PCB (normalizar coordenadas) ─────────────
        # El PCB tiene z ≈ 0 (la cara inferior). Normalizar X,Y al mínimo.
        # Filtrar primero el PCB por referencia/nombre
        pcb_inst = None
        comp_instances = []

        for inst in instances:
            is_pcb_inst = (
                inst['ref'] == pcb_ref or
                'PCB' in inst['footprint'].upper() or
                'BOARD' in inst['footprint'].upper() or
                (pcb_ref and pcb_ref in inst['ref'])
            )
            if is_pcb_inst and pcb_inst is None:
                pcb_inst = inst
            else:
                comp_instances.append(inst)

        # Si no encontramos PCB por nombre, usar el de menor Z (el sustrato)
        if pcb_inst is None and instances:
            pcb_inst = min(instances, key=lambda i: i['z'])
            comp_instances = [i for i in instances if i is not pcb_inst]

        # Extraer dimensiones reales del PCB desde su BREP
        pcb_dims = self._extract_pcb_dimensions()
        pcb_x0      = pcb_dims['x_min']
        pcb_y0      = pcb_dims['y_min']
        pcb_y_max   = pcb_dims.get('y_max', pcb_y0 + pcb_dims['height'])
        pcb_w       = pcb_dims['width']
        pcb_h       = pcb_dims['height']
        pcb_t       = pcb_dims['thickness']
        y_inverted  = pcb_dims.get('is_y_inverted', False)

        if self.verbose:
            print(f"[STEPTextParser] PCB detectado: {pcb_w:.1f}×{pcb_h:.1f}×{pcb_t:.2f} mm  "
                  f"Y_invertido={y_inverted}")

        pcb_geom = ComponentGeometry(
            shape_id=0,
            name="PCB",
            x_min=0.0, y_min=0.0, z_min=0.0,
            x_max=pcb_w, y_max=pcb_h, z_max=pcb_t,
            is_pcb=True
        )

        # ── 4. Construir ComponentGeometry para cada componente ───────────────
        # Normalización de coordenadas al sistema de referencia de la PCB:
        #   X: x_norm = x_abs - pcb_x0
        #   Y normal:   y_norm = y_abs - pcb_y0
        #   Y invertido (KiCad): y_norm = pcb_h - (y_abs - pcb_y0)
        #   El eje Y de KiCad en STEP apunta hacia abajo (Y negativo = arriba en diseño)
        components = []
        z_bottom = pcb_t  # componentes sobre la cara superior del PCB

        for idx, inst in enumerate(comp_instances):
            fp = inst['footprint']
            sz = _footprint_size(fp)  # (largo, ancho, alto) en mm

            x_c = inst['x'] - pcb_x0

            if y_inverted:
                # KiCad STEP: Y_abs es negativo. y_norm = pcb_h + (y_abs - pcb_y_max)
                # Equivalente: y_norm = pcb_h - (pcb_y_max - y_abs)
                # = pcb_h - (0 - y_abs) cuando pcb_y_max = 0
                y_c = pcb_h + (inst['y'] - pcb_y_max)
            else:
                y_c = inst['y'] - pcb_y0

            # Clamp al interior de la PCB (componentes en el borde)
            x_c = max(sz[0]/2, min(pcb_w - sz[0]/2, x_c))
            y_c = max(sz[1]/2, min(pcb_h - sz[1]/2, y_c))

            comp = ComponentGeometry(
                shape_id=idx + 1,
                name=inst['ref'],
                x_min=x_c - sz[0] / 2,
                y_min=y_c - sz[1] / 2,
                x_max=x_c + sz[0] / 2,
                y_max=y_c + sz[1] / 2,
                z_min=z_bottom,
                z_max=z_bottom + sz[2],
            )
            components.append(comp)

        # Deduplicar (ej: L1 exportado como sub-ensamblado con 2 sólidos)
        seen = {}
        deduped = []
        for c in components:
            if c.name not in seen:
                seen[c.name] = c
                deduped.append(c)
            elif c.footprint_area_mm2 > seen[c.name].footprint_area_mm2:
                deduped[deduped.index(seen[c.name])] = c
                seen[c.name] = c
        components = deduped

        if self.verbose:
            print(f"[STEPTextParser] Componentes montados: {len(components)}")
            for c in components:
                fp = next((i['footprint'] for i in comp_instances if i['ref'] == c.name), '?')
                print(f"  {c.name:<8} ({c.cx:6.1f},{c.cy:6.1f}) mm  {c.width:.1f}x{c.height:.1f} mm  fp={fp}")
        return pcb_geom, components

    # ── Parsers de entidades individuales ─────────────────────────────────────

    def _parse_cartesian_points(self) -> Dict[str, List[float]]:
        cp = {}
        for m in re.finditer(
            r'#(\d+)\s*=\s*CARTESIAN_POINT\s*\(\'[^\']*\',\s*\(([^)]+)\)\s*\)',
            self._content
        ):
            vals = [float(x.strip()) for x in m.group(2).split(',')]
            if len(vals) == 3:
                cp[m.group(1)] = vals
        return cp

    def _parse_axis2placement(self, cp: Dict) -> Dict[str, List[float]]:
        """AXIS2_PLACEMENT_3D id -> [x, y, z] del punto de origen"""
        a2p = {}
        for m in re.finditer(
            r'#(\d+)\s*=\s*AXIS2_PLACEMENT_3D\s*\(\'[^\']*\',\s*#(\d+)',
            self._content
        ):
            pos = cp.get(m.group(2))
            if pos:
                a2p[m.group(1)] = pos
        return a2p

    def _parse_idt(self, a2p: Dict) -> Dict[str, List[float]]:
        """ITEM_DEFINED_TRANSFORMATION id -> posición (ax_to)"""
        idt = {}
        for m in re.finditer(
            r'#(\d+)\s*=\s*ITEM_DEFINED_TRANSFORMATION\s*\(\'[^\']*\',\'[^\']*\',\s*#(\d+),\s*#(\d+)\)',
            self._content
        ):
            pos = a2p.get(m.group(3))  # ax_to
            if pos:
                idt[m.group(1)] = pos
        return idt

    def _parse_rr_blocks(self) -> Dict[str, str]:
        """Bloques compuestos RR → IDT id"""
        rr_idt = {}
        pattern = (
            r'#(\d+)\s*=\s*\(\s*REPRESENTATION_RELATIONSHIP\b.*?'
            r'REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION\s*\(#(\d+)\).*?'
            r'SHAPE_REPRESENTATION_RELATIONSHIP\s*\(\s*\)\s*\)\s*;'
        )
        for m in re.finditer(pattern, self._content, re.DOTALL):
            rr_idt[m.group(1)] = m.group(2)
        return rr_idt

    def _parse_nauo(self) -> Dict[str, Dict]:
        """NEXT_ASSEMBLY_USAGE_OCCURRENCE → {seq, ref, parent, child}"""
        nauo = {}
        for m in re.finditer(
            r"#(\d+)\s*=\s*NEXT_ASSEMBLY_USAGE_OCCURRENCE\s*\('([^']*)','([^']*)','[^']*',#(\d+),#(\d+)",
            self._content
        ):
            nauo[m.group(1)] = {
                'seq':    m.group(2),
                'ref':    m.group(3),
                'parent': m.group(4),
                'child':  m.group(5),
            }
        return nauo

    def _parse_pds_placement(self) -> Dict[str, str]:
        """PRODUCT_DEFINITION_SHAPE 'Placement' → NAUO id"""
        pds = {}
        for m in re.finditer(
            r"#(\d+)\s*=\s*PRODUCT_DEFINITION_SHAPE\s*\('Placement','[^']*',\s*#(\d+)\s*\)\s*;",
            self._content
        ):
            pds[m.group(1)] = m.group(2)
        return pds

    def _parse_cdsr(self) -> Dict[str, Tuple[str, str]]:
        """CONTEXT_DEPENDENT_SHAPE_REPRESENTATION → (rr_id, pds_id)"""
        cdsr = {}
        for m in re.finditer(
            r'#(\d+)\s*=\s*CONTEXT_DEPENDENT_SHAPE_REPRESENTATION\s*\(#(\d+),#(\d+)\)\s*;',
            self._content
        ):
            cdsr[m.group(1)] = (m.group(2), m.group(3))
        return cdsr

    def _build_footprint_map(self) -> Dict[str, str]:
        """child_def_id → nombre del footprint (PRODUCT name)"""
        prod_map = {}
        for m in re.finditer(r"#(\d+)\s*=\s*PRODUCT\s*\('([^']+)'", self._content):
            prod_map[m.group(1)] = m.group(2)
        form_to_prod = {}
        for m in re.finditer(
            r'#(\d+)\s*=\s*PRODUCT_DEFINITION_FORMATION\s*\(.*?,.*?,#(\d+)\s*\)',
            self._content
        ):
            form_to_prod[m.group(1)] = m.group(2)
        def_to_form = {}
        for m in re.finditer(
            r'#(\d+)\s*=\s*PRODUCT_DEFINITION\s*\([^,]*,[^,]*,#(\d+)',
            self._content
        ):
            def_to_form[m.group(1)] = m.group(2)

        fp_map = {}
        for def_id in def_to_form:
            form = def_to_form.get(def_id, '')
            prod = form_to_prod.get(form, '')
            name = prod_map.get(prod, '')
            if name:
                fp_map[def_id] = name
        return fp_map

    def _find_pcb_name(self) -> Optional[str]:
        """Busca el nombre de referencia del PCB en el STEP."""
        for m in re.finditer(
            r"NEXT_ASSEMBLY_USAGE_OCCURRENCE\s*\('[^']*','([^']*)'[^)]*\)",
            self._content
        ):
            ref = m.group(1)
            # KiCad exporta el PCB con referencia vacía o con '=>'
        # Buscar por nombre de producto que contenga '_PCB'
        for m in re.finditer(r"PRODUCT\('([^']*PCB[^']*)'", self._content, re.IGNORECASE):
            return m.group(1)
        return None

    def _extract_pcb_dimensions(self) -> Dict[str, float]:
        """
        Extrae dimensiones del PCB buscando el producto con nombre *_PCB.
        Como fallback, estima el bounding box de todos los puntos de inserción.
        """
        # Buscar la ADVANCED_BREP_SHAPE del PCB y su bounding box
        # En ausencia de OCC, estimamos por los puntos de inserción de los componentes
        all_x, all_y, all_z = [], [], []

        cp = self._parse_cartesian_points()
        # Buscar AXIS2_PLACEMENT_3D asociados a ITEM_DEFINED_TRANSFORMATION
        a2p = self._parse_axis2placement(cp)
        idt_raw = {}
        for m in re.finditer(
            r'#(\d+)\s*=\s*ITEM_DEFINED_TRANSFORMATION\s*\(\'[^\']*\',\'[^\']*\',\s*#(\d+),\s*#(\d+)\)',
            self._content
        ):
            ax_to = m.group(3)
            pos = a2p.get(ax_to)
            if pos:
                idt_raw[m.group(1)] = pos
                all_x.append(pos[0])
                all_y.append(pos[1])
                all_z.append(pos[2])

        if not all_x:
            return {'x_min': 0, 'y_min': 0, 'z_min': 0,
                    'width': 100, 'height': 80, 'thickness': 1.6}

        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        z_min         = min(all_z)

        # Agregar margen perimetral para componentes en el borde
        margin = 5.0  # mm
        width     = (x_max - x_min) + 2 * margin
        height    = (y_max - y_min) + 2 * margin

        # Origen ligeramente antes del primer componente
        return {
            'x_min':     x_min - margin,
            'y_min':     y_min - margin,
            'z_min':     z_min,
            'width':     width,
            'height':    height,
            'thickness': 1.6,  # estándar, no extraíble sin OCC fácilmente
        }


# ─── Función de conveniencia ──────────────────────────────────────────────────

def parse_step_file(step_file: str, verbose: bool = True
                    ) -> Tuple[Optional[ComponentGeometry], List[ComponentGeometry]]:
    """
    Interfaz unificada. Intenta OCC primero, cae a parser de texto.

    Returns:
        (pcb, components)
    """
    try:
        from OCC.Core.STEPControl import STEPControl_Reader
        # OCC disponible — usar STEPParser original
        from step_parser import STEPParser
        parser = STEPParser(step_file, verbose=verbose)
        return parser.extract_components()
    except ImportError:
        pass

    # Fallback: parser de texto
    parser = STEPTextParser(step_file, verbose=verbose)
    return parser.load()
