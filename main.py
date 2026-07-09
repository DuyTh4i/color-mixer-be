import os
import math
from contextlib import asynccontextmanager
from typing import List, Tuple, Optional

from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI
from pydantic import BaseModel, Field
from supabase import Client, create_client

from fastapi import FastAPI, Request
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from fastapi.middleware.cors import CORSMiddleware

from utils import *

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

app.state.limiter = limiter

app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

origins = [ALLOW_ORIGIN]

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


# ============================================================
#  Request Model
# ============================================================

class RequestingRecipes(BaseModel):
    hex_value: str
    subcollection_ids: list[str]
    step_size: int = Field(default=10, ge=1, le=100, description="Step size in percent: 1=fine, 10=coarse")


# ============================================================
#  RGB <-> XYZ <-> CIELAB Conversion (D65, sRGB)
# ============================================================

def _srgb_to_linear(c: float) -> float:
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def _lab_f(t: float) -> float:
    delta = 6 / 29
    if t > delta ** 3:
        return t ** (1 / 3)
    return t / (3 * delta ** 2) + 4 / 29


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip('#')
    if len(h) == 3:
        h = h[0] * 2 + h[1] * 2 + h[2] * 2
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{max(0, min(255, r)):02X}{max(0, min(255, g)):02X}{max(0, min(255, b)):02X}"


def lab_distance_squared(r1: int, g1: int, b1: int,
                         r2: int, g2: int, b2: int) -> float:
    def _rgb_to_lab(r, g, b):
        r_n, g_n, b_n = r / 255.0, g / 255.0, b / 255.0
        r_l = _srgb_to_linear(r_n)
        g_l = _srgb_to_linear(g_n)
        b_l = _srgb_to_linear(b_n)
        x = r_l * 0.4124564 + g_l * 0.3575761 + b_l * 0.1804375
        y = r_l * 0.2126729 + g_l * 0.7151522 + b_l * 0.0721750
        z = r_l * 0.0193339 + g_l * 0.1191920 + b_l * 0.9503041
        xn, yn, zn = 0.95047, 1.00000, 1.08883
        fx = _lab_f(x / xn)
        fy = _lab_f(y / yn)
        fz = _lab_f(z / zn)
        L = 116.0 * fy - 16.0
        a = 500.0 * (fx - fy)
        b_val = 200.0 * (fy - fz)
        return L, a, b_val

    l1, a1, b1_lab = _rgb_to_lab(r1, g1, b1)
    l2, a2, b2_lab = _rgb_to_lab(r2, g2, b2)
    dl = l1 - l2
    da = a1 - a2
    db = b1_lab - b2_lab
    return dl * dl + da * da + db * db


def blend_rgb(r1: int, g1: int, b1: int, ratio1: float,
              r2: int, g2: int, b2: int, ratio2: float) -> Tuple[int, int, int]:
    r = int(round(r1 * ratio1 + r2 * ratio2))
    g = int(round(g1 * ratio1 + g2 * ratio2))
    b = int(round(b1 * ratio1 + b2 * ratio2))
    return max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))


def blend_rgb_3(r1: int, g1: int, b1: int, ratio1: float,
                r2: int, g2: int, b2: int, ratio2: float,
                r3: int, g3: int, b3: int, ratio3: float) -> Tuple[int, int, int]:
    r = int(round(r1 * ratio1 + r2 * ratio2 + r3 * ratio3))
    g = int(round(g1 * ratio1 + g2 * ratio2 + g3 * ratio3))
    b = int(round(b1 * ratio1 + b2 * ratio2 + b3 * ratio3))
    return max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))


# ============================================================
#  Mixing Algorithm (from new_logic.py)
# ============================================================

def get_suggestions(hex_value: str, subcollection_ids: list[str], data: list) -> list:
    """Return colors from data whose hex matches hex_value AND belong to subcollection_ids."""
    return [
        item for item in data
        if item.get("hex_value", "").upper() == hex_value.upper()
        and item.get("subcollections", {}).get("id", "") in subcollection_ids
    ]


def calc_similarity_percent(error: float) -> float:
    """
    Lower error = higher similarity.
    max(0, (200 - sqrt(error))/200 * 100)
    """
    if error == float('inf'):
        return 0.0
    d = math.sqrt(error)
    similarity = max(0.0, (200.0 - d) / 200.0 * 100.0)
    return round(similarity, 1)


def get_blended_color(ingredients: List[Tuple[dict, float]]) -> str:
    """Compute resulting hex from list of (color_dict, percent) tuples."""
    if not ingredients:
        return "#000000"
    r_sum = g_sum = b_sum = 0.0
    for color, pct in ingredients:
        r, g, b = hex_to_rgb(color["hex_value"])
        r_sum += r * pct / 100.0
        g_sum += g * pct / 100.0
        b_sum += b * pct / 100.0
    return rgb_to_hex(int(round(r_sum)), int(round(g_sum)), int(round(b_sum)))


def find_best_formula(target_hex: str, available_colors: list, exclude_ids: set,
                      subcollection_ids: list, step_size: int = 10) -> Optional[dict]:
    """
    Port of findIngredientsForTargetColor.

    Returns dict with 'recipes', 'result_hex', 'match_results', or None.
    """
    tr, tg, tb = hex_to_rgb(target_hex)

    # Build candidate list (filtered)
    candidates = []
    for item in available_colors:
        if item["id"] in exclude_ids:
            continue
        sub_id = item.get("subcollections", {}).get("id", "")
        if sub_id not in subcollection_ids:
            continue
        r, g, b = hex_to_rgb(item["hex_value"])
        error = lab_distance_squared(tr, tg, tb, r, g, b)
        candidates.append({
            "data": item,
            "r": r, "g": g, "b": b,
            "single_error": error,
        })

    if not candidates:
        print("[MIX] No candidates available")
        return None

    # Step 1: Sort by single-color error, take Top 20
    candidates.sort(key=lambda x: x["single_error"])
    top_n = min(20, len(candidates))
    top = candidates[:top_n]
    print(f"[MIX] Target: {target_hex}, Candidates: {len(candidates)}, Top {top_n}, step_size={step_size}")

    best_items = None  # list of (color_dict, percent)
    best_error = float('inf')

    # Phase 2a: 1 ingredient
    for c in top:
        if c["single_error"] < best_error:
            best_error = c["single_error"]
            best_items = [(c["data"], 100.0)]

    # Phase 2b: 2 ingredients
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            c1 = top[i]
            c2 = top[j]
            for step in range(0, 101, step_size):
                ratio = step / 100.0
                complement = 1.0 - ratio
                br, bg_val, bb = blend_rgb(
                    c1["r"], c1["g"], c1["b"], ratio,
                    c2["r"], c2["g"], c2["b"], complement
                )
                error = lab_distance_squared(tr, tg, tb, br, bg_val, bb)
                if error < best_error:
                    best_error = error
                    best_items = [
                        (c1["data"], ratio * 100),
                        (c2["data"], complement * 100),
                    ]

    # Phase 2c: 3 ingredients
    if len(top) >= 3:
        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                for k in range(j + 1, len(top)):
                    c1 = top[i]
                    c2 = top[j]
                    c3 = top[k]
                    for r1 in range(0, 101, step_size):
                        for r2 in range(0, 101 - r1, step_size):
                            r3 = 100 - r1 - r2
                            if r3 < 0:
                                continue
                            ratio1 = r1 / 100.0
                            ratio2 = r2 / 100.0
                            ratio3 = r3 / 100.0
                            br, bg_val, bb = blend_rgb_3(
                                c1["r"], c1["g"], c1["b"], ratio1,
                                c2["r"], c2["g"], c2["b"], ratio2,
                                c3["r"], c3["g"], c3["b"], ratio3
                            )
                            error = lab_distance_squared(tr, tg, tb, br, bg_val, bb)
                            if error < best_error:
                                best_error = error
                                best_items = [
                                    (c1["data"], ratio1 * 100),
                                    (c2["data"], ratio2 * 100),
                                    (c3["data"], ratio3 * 100),
                                ]

    if best_items is None:
        return None

    # Filter out items with 0%
    best_items = [(c, pct) for c, pct in best_items if pct > 0]

    recipes = []
    for color, pct in best_items:
        recipes.append({
            "id": color["id"],
            "color_name": color["color_name"],
            "hex_value": color["hex_value"],
            "subcollections": {
                "product_img": color.get("subcollections", {}).get("product_img", "")
            },
            "rate": round(pct),
        })

    match_results = calc_similarity_percent(best_error)
    result_hex = get_blended_color(best_items)

    print(f"[MIX] Best: {[(c['color_name'], pct) for c, pct in best_items]} | error²={best_error:.1f} | similarity={match_results}% | result={result_hex}")

    return {
        "recipes": recipes,
        "result_hex": result_hex,
        "match_results": match_results,
    }


# ============================================================
#  API Endpoint
# ============================================================

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

    # Step 2: Exclude exact-match colors, find best formula
    exclude_ids = {item["id"] for item in suggest_colors}
    mix_result = find_best_formula(
        RequestingRecipes.hex_value,
        data,
        exclude_ids,
        RequestingRecipes.subcollection_ids,
        step_size=RequestingRecipes.step_size,
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