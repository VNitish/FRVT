from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from frt.routes import router as core_router
from frt.commons.config import Config

def create_app():
    frt = FastAPI(
        title="FRT Service",
        version="1.0.0",
        description="Face Recognition Toolkit Service API"
    )

    # Enable CORS
    frt.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Replace with your frontend URL in prod
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    frt.state.config = Config()

    # Register main routes
    frt.include_router(core_router)

    return frt