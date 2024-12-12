from fastapi import APIRouter, HTTPException, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, Dict, Union, List
from app.features.complete_template import TripDetails

templates = Jinja2Templates(directory="../frontend/public")
router = APIRouter()
class ItineraryRequest(BaseModel):
    query: str
    session_id: str
    
class LocationModel(BaseModel):
    name: str
    address: str
    is_start: bool = False
    is_end: bool = False
    lat: Optional[float] = None
    lng: Optional[float] = None

class ItineraryResponse(BaseModel):
    success: bool
    itinerary_text: Optional[str] = None
    locations: Optional[List[LocationModel]] = None
    route_info: Optional[Dict] = None
    route_services: Optional[Dict] = None
    error: Optional[str] = None

class ConversationResponse(BaseModel):
    type: str
    content: Union[Dict, Dict[str, str]]
    trip_details: Optional[TripDetails] = None

@router.get("/")
async def serve_index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "gmaps_api_key": request.app.state.gmaps_api_key
    })

@router.post("/init-session")
async def initialize_session(request: Request):  # Added Request parameter
    try:
        session_id = request.app.state.travel_agent.conversation_manager.initialize_session()
        return {"session_id": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat", response_model=ConversationResponse)
async def chat_endpoint(itinerary_request: ItineraryRequest, request: Request):
    if not itinerary_request.session_id:
        raise HTTPException(status_code=400, detail="Session ID is required")
    
    result = request.app.state.travel_agent.handle_conversation(
        itinerary_request.query, 
        itinerary_request.session_id
    )
    return result