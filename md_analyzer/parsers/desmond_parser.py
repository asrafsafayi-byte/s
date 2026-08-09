import re
import os
import logging
from typing import Optional
from md_analyzer.models import SimulationConfig

logger = logging.getLogger(__name__)


class DesmondConfigParser:
    """فقط می‌خواند و پارس می‌کند. هیچ‌چیز نمی‌سازد."""

    def parse(self, filepath: str) -> SimulationConfig:
        cfg = SimulationConfig()
        
        # اعتبارسنجی ورودی
        if not filepath or not isinstance(filepath, str):
            logger.error("مسیر فایل نامعتبر است")
            cfg.raw_metadata["خطا"] = "مسیر فایل نامعتبر است"
            return cfg
        
        if not os.path.exists(filepath):
            logger.warning(f"فایل یافت نشد: {filepath}")
            cfg.raw_metadata["خطا"] = f"فایل یافت نشد: {filepath}"
            return cfg

        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception as e:
            logger.error(f"خطا در خواندن فایل: {e}")
            cfg.raw_metadata["خطای خواندن فایل"] = str(e)
            return cfg

        # تعداد اتم‌ها
        if m := re.search(r'#\s*N\s+atoms\s*=\s*(\d+)', content, re.I):
            cfg.atom_count = int(m.group(1))

        # ensemble
        if m := re.search(r'ensemble\s*=\s*\{[^}]*class\s*=\s*"(\w+)"', content, re.S):
            cfg.ensemble = m.group(1)
        elif m := re.search(r'class\s*=\s*"(NVT|NPT|NPAT|NPH)"', content, re.I):
            cfg.ensemble = m.group(1)

        # thermostat method
        if m := re.search(r'method\s*=\s*"(Nose-Hoover|NH|Langevin|Berendsen)"', content, re.I):
            cfg.thermostat = m.group(1)

        # temperature
        if m := re.search(r'temperature\s*=\s*\[\[\s*"([\d.]+)"', content, re.I):
            cfg.target_temp_k = float(m.group(1))

        # pressure
        if m := re.search(r'pressure\s*=\s*"([\d.]+)"', content, re.I):
            cfg.target_pressure_bar = float(m.group(1))

        # timestep
        if m := re.search(r'timestep\s*=\s*\[\s*"([\d.]+)"', content, re.I):
            cfg.timestep_fs = float(m.group(1)) * 1000.0

        # simulation time
        if m := re.search(r'time\s*=\s*"([\d.]+)"', content, re.I):
            cfg.simulation_time_ns = float(m.group(1)) / 1000.0

        # box dimensions (3x3 matrix flattened) - الگوی انعطاف‌پذیرتر
        box_pat = r'box\s*=\s*\[\s*"([\d.eE+-]+)"\s*"[\d.eE+-]+"\s*"[\d.eE+-]+"\s*"[\d.eE+-]+"\s*"([\d.eE+-]+)"\s*"[\d.eE+-]+"\s*"[\d.eE+-]+"\s*"[\d.eE+-]+"\s*"([\d.eE+-]+)"'
        if m := re.search(box_pat, content, re.I):
            try:
                cfg.box_dimensions_angstrom = (
                    float(m.group(1)), 
                    float(m.group(2)), 
                    float(m.group(3))
                )
            except ValueError:
                logger.warning("خطا در پارس کردن ابعاد باکس")

        return cfg
