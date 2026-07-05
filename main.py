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

load_dotenv()


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

@router.get("/brands")
@limiter.limit("5/minute")
def get_brands():
    response = supabase.table("brands").select("id, name, logo , subcollections(id, name, product_img)").execute()
    return response.data

class RequestingRecipes(BaseModel):
    hex_value: str
    subcollection_ids: list[str]


@router.post("/recipes")
def get_recipes(RequestingRecipes: RequestingRecipes):
    response = supabase.table("colors").select("id, color_name, hex_value, subcollections(product_img)").in_("collection_id", RequestingRecipes.subcollection_ids).execute()
    return ""

app.include_router(router, prefix="/api/v1")