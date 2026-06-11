"""Unit conversion utilities for JSBSim (imperial) <-> SI (metric)."""

# Length
FT_TO_M = 0.3048
M_TO_FT = 1.0 / FT_TO_M
NM_TO_M = 1852.0
M_TO_NM = 1.0 / NM_TO_M

# Speed
FPS_TO_MPS = FT_TO_M
MPS_TO_FPS = M_TO_FT
KTS_TO_MPS = NM_TO_M / 3600.0
MPS_TO_KTS = 3600.0 / NM_TO_M

# Angle
DEG_TO_RAD = 3.141592653589793 / 180.0
RAD_TO_DEG = 180.0 / 3.141592653589793

# Acceleration
G_TO_MPS2 = 9.80665
FTPS2_TO_MPS2 = FT_TO_M


def ft_to_m(ft: float) -> float:
    return ft * FT_TO_M


def m_to_ft(m: float) -> float:
    return m * M_TO_FT


def kts_to_mps(kts: float) -> float:
    return kts * KTS_TO_MPS


def mps_to_kts(mps: float) -> float:
    return mps * MPS_TO_KTS


def deg_to_rad(deg: float) -> float:
    return deg * DEG_TO_RAD


def rad_to_deg(rad: float) -> float:
    return rad * RAD_TO_DEG
