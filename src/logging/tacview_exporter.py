"""Tacview ACMI file exporter for JSBSim air combat data.

Produces .txt.acmi files compatible with Tacview Advanced / Tacview Starter.
"""

import math
from datetime import datetime, timezone
from typing import List


class TacviewExporter:
    """Export accumulated simulation frames to Tacview ACMI format."""

    def __init__(self, filepath: str, base_lat: float = 30.0, base_lon: float = 120.0):
        self.filepath = filepath
        self.base_lat = base_lat
        self.base_lon = base_lon

        # Meters per degree at reference latitude
        self._m_per_deg_lat = 111320.0
        self._m_per_deg_lon = self._m_per_deg_lat * math.cos(math.radians(base_lat))

    def write(self, frames: List[dict]) -> None:
        """Write all frames to the ACMI file.

        Args:
            frames: List of frame dicts with keys:
                time: float (seconds)
                attacker: {lon_deg, lat_deg, alt_ft, roll_deg, pitch_deg, yaw_deg}
                evader:   {lon_deg, lat_deg, alt_ft, roll_deg, pitch_deg, yaw_deg}
        """
        with open(self.filepath, "w", encoding="utf-8") as f:
            # Header
            ref_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            f.write("FileType=text/acmi/tacview\n")
            f.write("FileVersion=2.1\n")
            f.write(f"0,ReferenceTime={ref_time}\n")
            f.write(f"0,ReferenceLongitude={self.base_lon}\n")
            f.write(f"0,ReferenceLatitude={self.base_lat}\n")

            # Object registration
            f.write(f"101,T={self.base_lon}|{self.base_lat}|0|0|0|0,Type=Air+FixedWing,Name=Attacker,Color=Red\n")
            f.write(f"102,T={self.base_lon}|{self.base_lat}|0|0|0|0,Type=Air+FixedWing,Name=Evader,Color=Blue\n")

            # Frames
            for frame in frames:
                t = frame["time"]
                f.write(f"#{t:.3f}\n")
                self._write_object(f, "101", frame["attacker"])
                self._write_object(f, "102", frame["evader"])

    def _write_object(self, f, obj_id: str, state: dict) -> None:
        """Write one object's state for the current frame."""
        lon = state["lon_deg"]
        lat = state["lat_deg"]
        alt_ft = state["alt_ft"]
        roll = state["roll_deg"]
        pitch = state["pitch_deg"]
        yaw = state["yaw_deg"]

        # Convert PyBullet yaw (0=east, CCW) to Tacview yaw (0=north, CW)
        tacview_yaw = (90.0 - yaw) % 360.0

        f.write(f"{obj_id},T={lon:.7f}|{lat:.7f}|{alt_ft:.1f}|{roll:.1f}|{pitch:.1f}|{tacview_yaw:.1f}\n")
