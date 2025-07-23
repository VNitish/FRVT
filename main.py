from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from frt.routes import router as core_router
from frt.commons.config import Config
from frt.performance_monitor import performance_monitor
import time
import logging

logger = logging.getLogger(__name__)

def create_app():
    frt = FastAPI(
        title="FRT Service - Optimized",
        version="1.0.0",
        description="Face Recognition Toolkit Service API - Performance Optimized"
    )

    # Enable CORS with optimized settings
    frt.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Replace with your frontend URL in prod
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        max_age=3600,  # Cache preflight requests for 1 hour
    )

    # Performance monitoring middleware
    @frt.middleware("http")
    async def performance_middleware(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Record performance metrics
        performance_monitor.record_request(process_time)
        
        # Add performance headers
        response.headers["X-Process-Time"] = str(process_time)
        
        return response

    frt.state.config = Config()

    # Register main routes
    frt.include_router(core_router)
    
    # Add performance monitoring endpoint
    @frt.get("/metrics")
    async def get_metrics():
        """Get performance metrics"""
        return performance_monitor.get_metrics_summary(minutes=10)

    # Start performance monitoring
    @frt.on_event("startup")
    async def startup_event():
        performance_monitor.start_monitoring(interval=10.0)
        logger.info("FRT Service started with performance optimizations")

    @frt.on_event("shutdown")
    async def shutdown_event():
        performance_monitor.stop_monitoring()
        logger.info("FRT Service shutdown")

    return frt