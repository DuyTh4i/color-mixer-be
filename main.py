import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI
from pydantic import BaseModel
from supabase import Client, create_client

from fastapi import FastAPI, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from fastapi.middleware.cors import CORSMiddleware

from utils import *
import numpy as np
from itertools import combinations
from scipy.optimize import minimize
import math

load_dotenv()

INTERNAL_API_KEY = os.environ["INTERNAL_API_KEY"]
ALLOW_ORIGIN = os.environ["ALLOW_ORIGIN"]

supabase: Client = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SECRET_KEY"],
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(lifespan=lifespan)
router = APIRouter()

limiter = Limiter(key_func=get_remote_address)

origins = [
    ALLOW_ORIGIN
]

app.state.limiter = limiter

app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)


@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    if request.method == "OPTIONS":
        return await call_next(request)

    if request.url.path.startswith("/api/"):
        api_key = request.headers.get("X-API-KEY")
        if api_key != INTERNAL_API_KEY:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid request"},
            )
    response = await call_next(request)
    return response

@router.get("/brands")
@limiter.limit("30/minute")
async def get_brands(request: Request):
    if need_to_cache("brands", os.environ["CACHE_LIFETIME"]):
        response = supabase.table("brands").select("id, name, logo , subcollections(id, name, product_img, colors(count))").execute()
        cache_data(response.data, "brands")
        return response.data
    return get_cached_data("brands")


class RequestingRecipes(BaseModel):
    hex_value: str
    subcollection_ids: list[str]


# ============================================================
#  Color space conversion utilities
# ============================================================

def hex_to_rgb(hex_str: str):
    """Convert hex color string (e.g. '#FF0000') to RGB tuple (0-255)."""
    hex_str = hex_str.lstrip('#')
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_lab(rgb: tuple):
    """Convert RGB (0-255) to CIE Lab color space."""

    def _rgb_to_xyz(r, g, b):
        r, g, b = r / 255.0, g / 255.0, b / 255.0

        def _linearize(c):
            if c > 0.04045:
                return ((c + 0.055) / 1.055) ** 2.4
            return c / 12.92

        r = _linearize(r) * 100.0
        g = _linearize(g) * 100.0
        b = _linearize(b) * 100.0

        x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
        y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
        z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

        return x, y, z

    def _xyz_to_lab(x, y, z):
        # D65 reference white
        xn, yn, zn = 95.047, 100.000, 108.883

        def _f(t):
            if t > 0.008856:
                return t ** (1.0 / 3.0)
            return (7.787 * t) + (16.0 / 116.0)

        fx = _f(x / xn)
        fy = _f(y / yn)
        fz = _f(z / zn)

        L = (116.0 * fy) - 16.0
        a = 500.0 * (fx - fy)
        b_val = 200.0 * (fy - fz)

        return np.array([L, a, b_val])

    x, y, z = _rgb_to_xyz(*rgb)
    return _xyz_to_lab(x, y, z)


def delta_e(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """CIE76 Delta E (Euclidean distance in Lab space)."""
    return float(np.sqrt(np.sum((lab1 - lab2) ** 2)))


def lab_to_rgb(lab: np.ndarray):
    """Convert CIE Lab to RGB (0-255)."""
    L, a, b_val = lab[0], lab[1], lab[2]

    # Lab -> XYZ (D65)
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b_val / 200.0

    def _f_inv(t):
        if t > 0.008856:
            return t ** 3.0
        return (t - 16.0 / 116.0) / 7.787

    xn, yn, zn = 95.047, 100.000, 108.883
    x = _f_inv(fx) * xn
    y = _f_inv(fy) * yn
    z = _f_inv(fz) * zn

    # XYZ -> linear RGB
    r_lin = x * 3.2404542 + y * -1.5371385 + z * -0.4985314
    g_lin = x * -0.9692660 + y * 1.8760108 + z * 0.0415560
    b_lin = x * 0.0556434 + y * -0.2040259 + z * 1.0572252

    # Gamma correction
    def _gamma(c):
        if c > 0.0031308:
            return 1.055 * (c ** (1.0 / 2.4)) - 0.055
        return 12.92 * c

    r = _gamma(r_lin)
    g = _gamma(g_lin)
    b = _gamma(b_lin)

    # Clamp 0-255
    return (
        max(0, min(255, int(round(r * 255)))),
        max(0, min(255, int(round(g * 255)))),
        max(0, min(255, int(round(b * 255)))),
    )


def rgb_to_hex(rgb: tuple) -> str:
    """Convert RGB (0-255) to hex string e.g. '#FF0000'."""
    return "#{:02X}{:02X}{:02X}".format(*rgb)


# ============================================================
#  Mixing algorithm
# ============================================================

def get_suggestions(hex_value: str, subcollection_ids: list[str], data: list) -> list:
    """Return colors from data whose hex matches hex_value AND belong to subcollection_ids."""
    return [
        item for item in data
        if item.get("hex_value", "").upper() == hex_value.upper()
        and item.get("subcollections", {}).get("id", "") in subcollection_ids
    ]


def slsqp_mix(target_lab: np.ndarray, comp_labs: list):
    """
    Optimize weights for N color components using SLSQP.
    Returns continuous weights (sum to 1.0) and delta_e.
    """
    n = len(comp_labs)

    def objective(w):
        mix_lab = np.zeros(3)
        for i in range(n):
            mix_lab += w[i] * comp_labs[i]
        return float(np.sum((mix_lab - target_lab) ** 2))

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = [(0.0, 1.0) for _ in range(n)]
    w0 = np.ones(n) / n

    res = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 200, 'ftol': 1e-12})
    delta_e_val = float(np.sqrt(res.fun))
    return res.x, delta_e_val


def continuous_to_integer_ratios(weights: np.ndarray, min_r: int = 1, max_r: int = 10) -> list:
    """
    Convert continuous weights (sum=1) to integer drop counts (1-10)
    that best preserve the relative proportions.

    Tries all possible total drops T (from N*min_r to N*max_r),
    finds the integer distribution closest to the ideal weights.
    """
    n = len(weights)
    if max(weights) == 0:
        return [min_r] * n

    best_ratios = None
    best_error = float('inf')

    for total_r in range(n * min_r, n * max_r + 1):
        ideal = [w * total_r for w in weights]
        rounded = [max(min_r, min(max_r, int(round(v)))) for v in ideal]
        current_sum = sum(rounded)
        diff = total_r - current_sum

        if diff == 0:
            pass
        elif diff > 0:
            for _ in range(diff):
                candidates = [i for i in range(n) if rounded[i] < max_r]
                if not candidates:
                    break
                i = max(candidates, key=lambda x: ideal[x] - rounded[x])
                rounded[i] += 1
        else:
            for _ in range(-diff):
                candidates = [i for i in range(n) if rounded[i] > min_r]
                if not candidates:
                    break
                i = min(candidates, key=lambda x: ideal[x] - rounded[x])
                rounded[i] -= 1

        if sum(rounded) != total_r:
            continue

        error = sum(((rounded[i] / total_r) - weights[i]) ** 2 for i in range(n))
        if error < best_error:
            best_error = error
            best_ratios = rounded

    if best_ratios is None:
        best_ratios = [max(min_r, min(max_r, int(round(w * max_r)))) for w in weights]

    return best_ratios


def lm_mix(target_lab: np.ndarray, comp_labs: list):
    """
    Optimize weights for N color components using Levenberg-Marquardt.
    Uses softmax parametrization for sum(w)=1, w_i >= 0.
    Returns continuous weights and delta_e.
    """
    n = len(comp_labs)
    labs_matrix = np.array(comp_labs)  # shape (n, 3)

    def residuals(z):
        # z: unconstrained params, w = softmax(z)
        exp_z = np.exp(z - np.max(z))  # stabilize
        w = exp_z / np.sum(exp_z)
        mix_lab = w @ labs_matrix  # (3,)
        return mix_lab - target_lab  # residual vector (3,)

    z0 = np.zeros(n)
    res = minimize(
        lambda z: np.sum(residuals(z) ** 2),
        z0,
        method='L-BFGS-B',
        options={'maxiter': 500, 'ftol': 1e-15}
    )
    # Also try least_squares with LM
    try:
        from scipy.optimize import least_squares
        res_ls = least_squares(residuals, z0, method='lm', max_nfev=500)
        z_opt = res_ls.x
    except Exception:
        z_opt = res.x

    exp_z = np.exp(z_opt - np.max(z_opt))
    w_opt = exp_z / np.sum(exp_z)
    mix_lab = w_opt @ labs_matrix
    de = delta_e(mix_lab, target_lab)

    return w_opt, de


def find_best_mix(target_hex: str, available_colors: list, exclude_ids: set, subcollection_ids: list):
    """
    2-step pipeline to find best mix formula (2-4 colors):
      Step 1: Compute single-color Delta E, keep Top 20 closest colors.
      Step 2: Try combos of size 2,3,4 from Top 20 using Levenberg-Marquardt.

    Only uses colors whose subcollection id is in subcollection_ids.
    Excludes colors whose ids are in exclude_ids.

    Priority: lowest delta_e first, then fewest colors as tiebreaker.

    Returns:
        dict with 'recipes', 'result_hex', 'match_results', or None.
    """
    target_lab = rgb_to_lab(hex_to_rgb(target_hex))

    # Build available list with single-color delta_e, filtered
    available = []
    for item in available_colors:
        if item["id"] in exclude_ids:
            continue
        sub_id = item.get("subcollections", {}).get("id", "")
        if sub_id not in subcollection_ids:
            continue
        lab = rgb_to_lab(hex_to_rgb(item["hex_value"]))
        de = delta_e(target_lab, lab)
        available.append({
            "id": item["id"],
            "color_name": item["color_name"],
            "hex_value": item["hex_value"],
            "subcollections": item.get("subcollections", {}),
            "lab": lab,
            "single_de": de,
        })

    if len(available) < 2:
        print(f"[MIX] Not enough colors ({len(available)}), need at least 2")
        return None

    # ---- Step 1: Keep top 20 closest single colors ----
    available.sort(key=lambda x: x["single_de"])
    top_n = min(20, len(available))
    top_colors = available[:top_n]

    print(f"[MIX] Target: {target_hex}, Available: {len(available)}, Top {top_n} selected for combos")
    total_combos = sum(
        len(list(combinations(top_colors, n)))
        for n in range(2, min(5, len(top_colors) + 1))
    )
    print(f"[MIX] Total combos to evaluate: {total_combos}")
    evaluated = 0

    best_result = None
    best_priority = (float('inf'), float('inf'))  # (delta_e, num_colors)

    # ---- Step 2: Levenberg-Marquardt for combos of size 2,3,4 ----
    for num_colors in range(2, min(5, len(top_colors) + 1)):
        num_combos = len(list(combinations(top_colors, num_colors)))
        print(f"[MIX] Trying {num_colors}-color combos: {num_combos} combinations")
        for combo in combinations(top_colors, num_colors):
            comp_labs = [c["lab"] for c in combo]
            weights_cont, de_lm = lm_mix(target_lab, comp_labs)

            if de_lm is None or math.isnan(de_lm):
                continue

            # Convert to integer ratios
            int_ratios = continuous_to_integer_ratios(weights_cont, min_r=1, max_r=10)

            # Compute actual delta_e with integer ratios
            total_w = sum(int_ratios)
            mix_lab = np.zeros(3)
            for i in range(num_colors):
                mix_lab += (int_ratios[i] / total_w) * comp_labs[i]
            final_de = delta_e(mix_lab, target_lab)

            evaluated += 1

            priority = (final_de, num_colors)
            if priority < best_priority:
                best_priority = priority
                best_result = (combo, int_ratios, final_de)
                combo_names = [c["color_name"] for c in combo]
                print(f"[MIX] New best! #{evaluated} | {num_colors} colors: {combo_names} | rates={int_ratios} | delta_e={final_de:.2f}")

    print(f"[MIX] Evaluated {evaluated}/{total_combos} combos. Best delta_e={best_priority[0]:.2f}, colors={best_priority[1]}")

    if best_result is None:
        return None

    combo, int_ratios, final_de = best_result
    total_w = sum(int_ratios)

    recipes = []
    for i, c in enumerate(combo):
        recipes.append({
            "id": c["id"],
            "color_name": c["color_name"],
            "hex_value": c["hex_value"],
            "subcollections": {
                "product_img": c["subcollections"].get("product_img", "")
            },
            "rate": int_ratios[i],
        })

    match_results = round(1.0 / (1.0 + final_de / 50.0), 4)

    # Mix in RGB space for accurate resulting color
    mix_rgb = np.zeros(3)
    for i in range(len(combo)):
        comp_rgb = np.array(hex_to_rgb(combo[i]["hex_value"]), dtype=float)
        mix_rgb += (int_ratios[i] / total_w) * comp_rgb
    mix_rgb = tuple(int(round(v)) for v in mix_rgb)
    result_hex = rgb_to_hex(mix_rgb)

    return {
        "recipes": recipes,
        "result_hex": result_hex,
        "match_results": match_results,
    }


@router.post("/recipes")
@limiter.limit("30/minute")
async def get_recipes(request: Request, RequestingRecipes: RequestingRecipes):
    data = list()
    if need_to_cache("recipes", os.environ["CACHE_LIFETIME"]):
        response = supabase.table("colors").select("id, color_name, hex_value, subcollections(id, product_img)").execute()
        cache_data(response.data, "recipes")
        data = response.data
    else:
        data = get_cached_data("recipes")

    # Step 1: Get matching colors (suggestions)
    suggest_colors = get_suggestions(
        RequestingRecipes.hex_value,
        RequestingRecipes.subcollection_ids,
        data
    )

    # Step 2: Exclude exact-match colors from mixing pool, then find best mix recipe (2-4 colors)
    exclude_ids = {item["id"] for item in suggest_colors}
    mix_result = find_best_mix(
        RequestingRecipes.hex_value,
        data,
        exclude_ids,
        RequestingRecipes.subcollection_ids
    )

    response_data = {
        "suggest_colors": suggest_colors,
        "recipes": [],
        "result_hex": "",
        "match_results": 0.0,
    }

    if mix_result is not None:
        response_data["recipes"] = mix_result["recipes"]
        response_data["result_hex"] = mix_result["result_hex"]
        response_data["match_results"] = mix_result["match_results"]

    return response_data


app.include_router(router, prefix="/api/v1")