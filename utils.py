import json
import time
import calendar
from pathlib import Path
import numpy as np

def _ensure_cache_dir():
    Path("/tmp/cache").mkdir(parents=True, exist_ok=True)

def cache_data(data, filename):
    _ensure_cache_dir()
    with open('/tmp/cache/' + str(get_current_timestamp()) + '_' + filename + '.json', 'w') as f:
        json.dump(data, f, indent=2)

def get_current_timestamp():
    return calendar.timegm(time.gmtime())

def need_to_cache(filename, max_age_seconds):
    filename = get_cached_filename(filename)
    if '.json' not in filename:
        return True
    
    created_time = int(filename.split('_')[0])

    if (get_current_timestamp() - created_time) >= int(max_age_seconds):
        return True

    return False

def get_cached_data(filename):
    filename = get_cached_filename(filename)
    if '.json' in filename:
        with open('/tmp/cache/' + filename, 'r') as f:
            return json.load(f)
    return ""

def get_cached_filename(filename):
    folder_path = Path("/tmp/cache")
    if not folder_path.exists():
        return ""
    for f in folder_path.iterdir():
        if filename in f.name and f.is_file():
            return f.name
    return ""

def hex_to_rgb(hex_value):
    hex_value = hex_value.lstrip('#')
    return [int(hex_value[i:i+2], 16) / 255.0 for i in (0, 2, 4)]

def rgb_to_lab(rgb):
    # sRGB sang Tuyến tính
    rgb_linear = [((c + 0.055) / 1.055) ** 2.4 if c > 0.04045 else c / 12.92 for c in rgb]
    # Tuyến tính sang XYZ (D65 White Point)
    X = rgb_linear[0] * 0.4124564 + rgb_linear[1] * 0.3575761 + rgb_linear[2] * 0.1804375
    Y = rgb_linear[0] * 0.2126729 + rgb_linear[1] * 0.7151522 + rgb_linear[2] * 0.0721750
    Z = rgb_linear[0] * 0.0193339 + rgb_linear[1] * 0.1191920 + rgb_linear[2] * 0.9503041
    # XYZ sang CIE L*a*b*
    X_n, Y_n, Z_n = 0.95047, 1.00000, 1.08883
    xr, yr, zr = X/X_n, Y/Y_n, Z/Z_n
    fx = xr ** (1/3) if xr > 0.008856 else (7.787 * xr) + (16/116)
    fy = yr ** (1/3) if yr > 0.008856 else (7.787 * yr) + (16/116)
    fz = zr ** (1/3) if zr > 0.008856 else (7.787 * fz) + (16/116)
    L = (116 * fy) - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.array([L, a, b])