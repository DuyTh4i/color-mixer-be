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

load_dotenv()

INTERNAL_API_KEY = os.environ["INTERNAL_API_KEY"]

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

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

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
# @limiter.limit("5/minute")
async def get_brands(request: Request):
    response = supabase.table("brands").select("id, name, logo , subcollections(id, name, product_img)").execute()
    return response.data

class RequestingRecipes(BaseModel):
    hex_value: str
    subcollection_ids: list[str]


@router.post("/recipes")
async def get_recipes(RequestingRecipes: RequestingRecipes):
    response = supabase.table("colors").select("id, color_name, hex_value, subcollections(product_img)").in_("collection_id", RequestingRecipes.subcollection_ids).execute()
    return ""

app.include_router(router, prefix="/api/v1")