from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from app.api.routes import router
from app.services.session_manager import DatabaseManager
from config import load_secrets
from app.features.complete_template import TripDetails
from app.services.travel_agent import TravelAgent

def create_app():
    app = FastAPI()
    
    # Load configuration
    secrets = load_secrets()
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )
    
    db_manager = DatabaseManager(secrets["DB_CONFIG"])
    db_manager.initialize_database()
    # Store configuration in app state
    app.state.db_manager = db_manager
    app.state.gmaps_api_key = secrets["GOOGLEMAPS_API_KEY"]
    app.state.openai_api_key = secrets["OPENAI_API_KEY"]
    app.state.travel_agent = TravelAgent(
        secrets["OPENAI_API_KEY"],
        secrets["GOOGLEMAPS_API_KEY"],
        db_manager
    )

    # Set up templates directory
    templates = Jinja2Templates(directory="../frontend/public")
    app.state.templates = templates

    # Mount static files for CSS and JS
    app.mount("/src", StaticFiles(directory="../frontend/src"), name="static")

    # Add route for serving index.html
    @app.get("/")
    async def serve_index(request: Request):
        return templates.TemplateResponse(
            "index.html", 
            {
                "request": request,
                "gmaps_api_key": secrets["GOOGLEMAPS_API_KEY"]
            }
        )
    
    # Include routers
    app.include_router(router, prefix="/api")
    
    return app