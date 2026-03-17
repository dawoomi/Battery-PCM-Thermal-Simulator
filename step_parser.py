"""
step_parser.py
==============
Extracción de geometría 3D desde archivos STEP usando pythonocc-core (OCC).

Pipeline:
  1. Leer STEP con STEPControl_Reader
  2. Transferir shapes al modelo
  3. Iterar sobre sólidos individuales (TopoDS_Solid)
  4. Calcular Bounding Box de cada sólido (Bnd_Box)
  5. Identificar el PCB por criterios de geometría (plano, mayor área XY)
  6. Proyectar footprints de componentes al plano XY del PCB

Nota sobre identificación del PCB:
  El FR4 board es el sólido con mayor extensión XY y menor extensión Z.
  Criterio: aspect_ratio = (dx * dy) / dz > threshold

  Si los sólidos tienen nombres (STEP Product Name), se puede buscar
  por nombre. En ausencia de nombres, se usa criterio geométrico.

Referencia API:
  - pythonocc-core: https://github.com/tpaviot/pythonocc-core
  - STEP AP214 ISO 10303-214
  - OCC Documentation: https://dev.opencascade.org/doc/overview/html/
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import warnings


@dataclass
class ComponentGeometry:
    """
    Geometría proyectada de un componente en el plano XY de la PCB.

    Coordenadas en mm referenciadas al origen del PCB (esquina inferior izquierda).
    """
    shape_id: int                    # Índice del sólido en el modelo STEP
    name: str = "unknown"            # Nombre del producto (si disponible en STEP)
    # Bounding Box en coordenadas STEP (mm)
    x_min: float = 0.0
    y_min: float = 0.0
    x_max: float = 0.0
    y_max: float = 0.0
    z_min: float = 0.0
    z_max: float = 0.0
    # Propiedades derivadas
    is_pcb: bool = False

    @property
    def cx(self) -> float:
        """Centro X [mm]"""
        return (self.x_min + self.x_max) / 2.0

    @property
    def cy(self) -> float:
        """Centro Y [mm]"""
        return (self.y_min + self.y_max) / 2.0

    @property
    def width(self) -> float:
        """Ancho [mm]"""
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        """Alto [mm]"""
        return self.y_max - self.y_min

    @property
    def thickness(self) -> float:
        """Espesor Z [mm]"""
        return self.z_max - self.z_min

    @property
    def footprint_area_mm2(self) -> float:
        return self.width * self.height

    @property
    def xy_aspect_ratio(self) -> float:
        """Relación (dx*dy)/dz — alto → plano (PCB), bajo → componente vertical"""
        if self.thickness < 1e-6:
            return float('inf')
        return self.footprint_area_mm2 / self.thickness


class STEPParser:
    """
    Parser de archivos STEP para extracción de geometría de PCBs.

    Uso:
        parser = STEPParser("board.step")
        components = parser.extract_components()
        pcb = parser.get_pcb()
        components_normalized = parser.normalize_to_pcb(components, pcb)
    """

    def __init__(self, step_file: str, verbose: bool = True):
        self.step_file = step_file
        self.verbose = verbose
        self._shapes = []
        self._components: List[ComponentGeometry] = []
        self._occ_available = self._check_occ()

    def _check_occ(self) -> bool:
        try:
            from OCC.Core.STEPControl import STEPControl_Reader
            from OCC.Core.IFSelect import IFSelect_RetDone
            from OCC.Core.BRep import BRep_Builder
            from OCC.Core.BRepBndLib import brepbndlib_Add
            from OCC.Core.Bnd import Bnd_Box
            from OCC.Core.TopExp import TopExp_Explorer
            from OCC.Core.TopAbs import TopAbs_SOLID
            return True
        except ImportError:
            warnings.warn(
                "[STEPParser] pythonocc-core no disponible. "
                "Instalar con: conda install -c conda-forge pythonocc-core\n"
                "Usando modo de demostración con geometría sintética.",
                RuntimeWarning
            )
            return False

    def load(self) -> bool:
        """
        Carga el archivo STEP y extrae todos los sólidos con sus bounding boxes.

        Returns:
            True si la carga fue exitosa.
        """
        if not self._occ_available:
            print("[STEPParser] Usando geometría sintética (modo demo).")
            return False

        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.BRepBndLib import brepbndlib_Add
        from OCC.Core.Bnd import Bnd_Box
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_SOLID, TopAbs_COMPOUND
        from OCC.Core.BRep import BRep_Builder
        from OCC.Core.TopoDS import TopoDS_Compound
        from OCC.Core.STEPCAFControl import STEPCAFControl_Reader
        from OCC.Core.XCAFDoc import XCAFDoc_DocumentTool
        from OCC.Core.TDocStd import TDocStd_Document

        reader = STEPControl_Reader()
        status = reader.ReadFile(self.step_file)

        if status != IFSelect_RetDone:
            raise RuntimeError(f"Error leyendo STEP: {self.step_file}")

        reader.TransferRoots()
        n_shapes = reader.NbShapes()

        if self.verbose:
            print(f"[STEPParser] STEP cargado: {n_shapes} shape(s) raíz")

        # Expandir todos los sólidos
        solid_id = 0
        for i in range(1, n_shapes + 1):
            shape = reader.Shape(i)
            explorer = TopExp_Explorer(shape, TopAbs_SOLID)
            while explorer.More():
                solid = explorer.Current()
                bbox = Bnd_Box()
                bbox.SetGap(0.0)
                brepbndlib_Add(solid, bbox)

                if not bbox.IsVoid():
                    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
                    comp = ComponentGeometry(
                        shape_id=solid_id,
                        name=f"solid_{solid_id:03d}",
                        x_min=xmin, y_min=ymin, z_min=zmin,
                        x_max=xmax, y_max=ymax, z_max=zmax,
                    )
                    self._components.append(comp)
                    solid_id += 1
                explorer.Next()

        if self.verbose:
            print(f"[STEPParser] Sólidos extraídos: {len(self._components)}")

        return len(self._components) > 0

    def identify_pcb(self,
                     xy_aspect_threshold: float = 10.0,
                     pcb_thickness_range_mm: Tuple[float, float] = (0.5, 3.2)
                     ) -> Optional[ComponentGeometry]:
        """
        Identifica el sólido correspondiente al PCB.

        Criterios (en orden de prioridad):
          1. Nombre contiene 'pcb', 'board', 'fr4' (case-insensitive)
          2. Aspect ratio (dx*dy)/dz > threshold Y espesor dentro del rango IPC
          3. Mayor área XY

        Args:
            xy_aspect_threshold: Mínimo aspect ratio para candidato PCB
            pcb_thickness_range_mm: Rango válido de espesor [mm] (IPC estándar: 0.8–3.2)

        Returns:
            ComponentGeometry del PCB, o None si no se puede identificar.
        """
        if not self._components:
            return None

        # Criterio 1: búsqueda por nombre
        keywords = ['pcb', 'board', 'fr4', 'substrate', 'pwb']
        for comp in self._components:
            if any(kw in comp.name.lower() for kw in keywords):
                comp.is_pcb = True
                if self.verbose:
                    print(f"[STEPParser] PCB identificado por nombre: '{comp.name}'")
                return comp

        # Criterio 2 y 3: geométrico
        candidates = [
            c for c in self._components
            if (c.xy_aspect_ratio > xy_aspect_threshold
                and pcb_thickness_range_mm[0] <= c.thickness <= pcb_thickness_range_mm[1])
        ]

        if not candidates:
            # Fallback: mayor área XY
            candidates = self._components

        pcb = max(candidates, key=lambda c: c.footprint_area_mm2)
        pcb.is_pcb = True

        if self.verbose:
            print(f"[STEPParser] PCB identificado por geometría: "
                  f"{pcb.width:.1f}×{pcb.height:.1f}×{pcb.thickness:.2f} mm | "
                  f"aspect={pcb.xy_aspect_ratio:.1f}")
        return pcb

    def normalize_to_pcb(self,
                          pcb: ComponentGeometry
                          ) -> List[ComponentGeometry]:
        """
        Normaliza las coordenadas de todos los componentes al sistema de referencia
        del PCB (origen en la esquina inferior izquierda del PCB).

        Solo devuelve componentes que estén dentro del footprint XY del PCB
        y por encima del plano Z superior del PCB (componentes montados).

        Args:
            pcb: ComponentGeometry del PCB

        Returns:
            Lista de ComponentGeometry normalizados (excluye el PCB).
        """
        normalized = []
        z_pcb_top = pcb.z_max  # Cara superior del PCB

        for comp in self._components:
            if comp.is_pcb:
                continue

            # Componente debe estar montado sobre el PCB (z_min ≥ z_top_pcb - tolerancia)
            tolerance_z = 0.5  # mm - tolerancia para gap de soldadura
            if comp.z_min < z_pcb_top - tolerance_z:
                continue  # Componente debajo del PCB (hardware, conector trasero)

            # Verificar intersección XY con el PCB
            if (comp.x_max < pcb.x_min or comp.x_min > pcb.x_max or
                    comp.y_max < pcb.y_min or comp.y_min > pcb.y_max):
                continue  # Fuera del PCB en XY

            # Normalizar al origen del PCB
            norm = ComponentGeometry(
                shape_id=comp.shape_id,
                name=comp.name,
                x_min=comp.x_min - pcb.x_min,
                y_min=comp.y_min - pcb.y_min,
                z_min=comp.z_min - pcb.z_min,
                x_max=comp.x_max - pcb.x_min,
                y_max=comp.y_max - pcb.y_min,
                z_max=comp.z_max - pcb.z_min,
            )
            normalized.append(norm)

        if self.verbose:
            print(f"[STEPParser] Componentes sobre PCB: {len(normalized)}")

        return normalized

    def extract_components(self) -> Tuple[Optional[ComponentGeometry],
                                           List[ComponentGeometry]]:
        """
        Pipeline completo: carga STEP, identifica PCB, normaliza componentes.

        Returns:
            (pcb, components) — ambos en coordenadas normalizadas al PCB.
        """
        success = self.load()
        if not success:
            return None, []

        pcb = self.identify_pcb()
        if pcb is None:
            raise RuntimeError("[STEPParser] No se pudo identificar el PCB.")

        components = self.normalize_to_pcb(pcb)
        return pcb, components


def generate_synthetic_geometry(cfg) -> Tuple['ComponentGeometry',
                                               List['ComponentGeometry']]:
    """
    Genera geometría sintética para demo/testing cuando OCC no está disponible.
    Simula una PCB con distribución realista de componentes.

    Args:
        cfg: PCBConfig

    Returns:
        (pcb, components) con coordenadas en mm
    """
    pcb = ComponentGeometry(
        shape_id=0,
        name="PCB_FR4",
        x_min=0, y_min=0, z_min=0,
        x_max=cfg.pcb_width_mm,
        y_max=cfg.pcb_height_mm,
        z_max=cfg.pcb_thickness_mm,
        is_pcb=True,
    )

    def _c(sid, name, x0, y0, x1, y1, z1=5.0):
        return ComponentGeometry(
            shape_id=sid, name=name,
            x_min=x0, y_min=y0, x_max=x1, y_max=y1,
            z_min=1.6, z_max=z1
        )

    synthetic_components = [
        _c(1,  "U1_BuckReg",   15, 10, 28, 22, 5.0),
        _c(2,  "U2_RFTrx",     40, 33, 54, 47, 3.5),
        _c(3,  "U3_MCU",       58, 12, 77, 33, 2.5),
        _c(4,  "L1_Inductor",  28,  6, 36, 18, 4.5),
        _c(5,  "L2_Inductor",  28, 20, 36, 30, 4.5),
        _c(6,  "C1_BulkCap",    8, 36, 15, 48, 7.0),
        _c(7,  "C2_BulkCap",    8, 50, 15, 62, 7.0),
        _c(8,  "Q1_MOSFET",    55, 47, 64, 58, 3.0),
        _c(9,  "Q2_MOSFET",    55, 60, 64, 70, 3.0),
        _c(10, "R_Array1",     80,  8, 93, 14, 0.5),
        _c(11, "U4_LDO",       18, 52, 30, 65, 2.0),
        _c(12, "XTAL_32M",     70, 37, 81, 50, 3.0),
    ]

    return pcb, synthetic_components
