import os
import re
import logging
import pandas as pd
from typing import List, Optional
from md_analyzer.models import EnergyData

logger = logging.getLogger(__name__)


class EnergyFileParser:
    def parse(self, filepath: str) -> Optional[EnergyData]:
        # اعتبارسنجی ورودی
        if not filepath or not isinstance(filepath, str):
            logger.error("مسیر فایل نامعتبر است")
            return None
        
        if not os.path.exists(filepath):
            logger.warning(f"فایل انرژی یافت نشد: {filepath}")
            return None

        # بررسی پسوند فایل
        if not filepath.lower().endswith('.ene'):
            logger.warning(f"فرمت فایل ممکن است اشتباه باشد: {filepath}")

        headers: List[str] = []
        try:
            with open(filepath, "r") as f:
                for line in f:
                    if line.startswith("#"):
                        parts = line.replace("#", "").strip().split()
                        if len(parts) > 2:
                            headers = parts
                    else:
                        break
        except Exception as e:
            logger.error(f"خطا در خواندن فایل انرژی: {e}")
            return None

        try:
            df = pd.read_csv(filepath, sep=r'\s+', comment='#', header=None, engine='python')
        except Exception as e:
            logger.error(f"خطا در پارس کردن داده‌های انرژی: {e}")
            return None

        # Assign columns
        if headers and len(headers) == df.shape[1]:
            df.columns = [str(h) for h in headers]
        elif df.shape[1] >= 6:
            default = ["Time", "E_tot", "E_p", "E_k", "Temp", "Press"]
            df.columns = default + [f"Col_{i}" for i in range(6, df.shape[1])]
        else:
            df.columns = [f"Col_{i}" for i in range(df.shape[1])]

        # Coerce numeric
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        time_col = "Time" if "Time" in df.columns else df.columns[0]
        available = [c for c in df.columns if str(c).lower() not in ("time", "frame")]

        total_ns = None
        if time_col == "Time" and not df[time_col].isnull().all() and len(df) > 0:
            total_ns = (df[time_col].iloc[-1] - df[time_col].iloc[0]) / 1000.0

        return EnergyData(
            df=df,
            available_metrics=available,
            time_column=time_col,
            total_time_ns=total_ns,
        )
