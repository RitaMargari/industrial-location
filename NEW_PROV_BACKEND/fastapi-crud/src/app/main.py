from fastapi import FastAPI

from app.api.routers import get_provision

app = FastAPI()

app.include_router(get_provision.router, prefix="/spatial")
