from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
class TripDetails(BaseModel):
    start_location: Optional[str] = None
    destination: Optional[str] = None
    duration: Optional[str] = None
    budget: Optional[str] = None
    travel_style: Optional[str] = None
    interests: List[str] = []
    num_travelers: Optional[int] = None
    special_requirements: Optional[str] = None
    dates: Optional[str] = None
    ready_for_itinerary: bool = False
    def check_readiness(self) -> bool:
        """
        Check if all required fields are present for itinerary generation
        """
        required_fields = {
            'start_location': self.start_location,
            'destination': self.destination,
            'duration': self.duration,
            'travel_style': self.travel_style,
            'num_travelers': self.num_travelers
        }
        
        # Check if all required fields have values
        has_required = all(value is not None and value != '' for value in required_fields.values())
        
        # Check if interests list is not empty
        has_interests = len(self.interests) > 0 if isinstance(self.interests, list) else False
        
        # Update ready_for_itinerary flag
        self.ready_for_itinerary = has_required and has_interests
        return self.ready_for_itinerary

    def update(self, new_info: dict) -> None:
        """
        Update trip details with new information
        """
        for key, value in new_info.items():
            if value is not None and value != '':
                if key == 'interests' and isinstance(value, list):
                    current_interests = getattr(self, 'interests', [])
                    setattr(self, key, list(set(current_interests + value)))
                else:
                    setattr(self, key, value)
        
        # Check readiness after update
        self.check_readiness()
import uuid
class EnhancedConversationManager:
    def __init__(self, max_history_length=100):
        self.conversations = {}
        self.trip_details = {}
        self.max_history_length = max_history_length
    def initialize_session(self) -> str:
        """
        Initializes a new session and returns the unique session_id.
        """
        session_id = str(uuid.uuid4())
        self.conversations[session_id] = []
        self.trip_details[session_id] = TripDetails()
        return session_id
    def add_message(self, session_id, role, message):
        if session_id not in self.conversations:
            self.conversations[session_id] = []
            self.trip_details[session_id] = TripDetails()
        
        # Add message to conversation history
        self.conversations[session_id].append({
            "role": role,
            "content": message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Trim history if needed
        if len(self.conversations[session_id]) > self.max_history_length:
            self.conversations[session_id] = self.conversations[session_id][-self.max_history_length:]
    
    def update_trip_details(self, session_id, new_info: dict):
        """
        Update trip details with a more intelligent merging strategy.
        
        Args:
            session_id (str): Unique identifier for the conversation session
            new_info (dict): Newly extracted trip details to merge
        """
        if session_id not in self.trip_details:
            self.trip_details[session_id] = TripDetails()
        
        # Get current details as a dictionary
        current_details = self.trip_details[session_id].dict()
        
        # Merge strategy: only update fields that have non-None, non-empty values
        for key, value in new_info.items():
            # Special handling for list fields like interests
            if key == 'interests':
                # Merge interests, removing duplicates
                if value and isinstance(value, list):
                    current_interests = current_details.get('interests', [])
                    merged_interests = list(set(current_interests + value))
                    current_details[key] = merged_interests
            else:
                # For other fields, only update if new value is not None/empty
                if value is not None and value != '':
                    current_details[key] = value
        
        # Update the trip details object
        updated_trip_details = TripDetails(**current_details)
        
        # Define the strictly required fields
        required_fields = ['start_location', 'destination', 'duration', 'travel_style', 'interests', 'num_travelers']
        
        # Check if ALL required fields are present and non-empty
        updated_trip_details.ready_for_itinerary = all(
            getattr(updated_trip_details, field) is not None 
            and (not isinstance(getattr(updated_trip_details, field), list) or len(getattr(updated_trip_details, field)) > 0)
            and getattr(updated_trip_details, field) != '' 
            for field in required_fields
        )
        
        # Store the updated trip details
        self.trip_details[session_id] = updated_trip_details
        
    def get_conversation_history(self, session_id):
        return self.conversations.get(session_id, [])
    
    def get_trip_details(self, session_id) -> TripDetails:
        return self.trip_details.get(session_id, TripDetails())

class EnhancedConversationTemplate:
    def __init__(self):
        self.system_template = """
        You are TravelBuddy, a friendly and engaging travel assistant who loves helping people plan amazing trips! 

        Core Traits:
        - Warm and enthusiastic personality
        - Detail-oriented but conversational
        - Proactive in gathering important information
        - Memory of previous conversation points
        - Expert travel knowledge and tips provider
        - Flexible and adaptable to changing preferences
        
        CRITICAL REQUIREMENTS IF USER ASKS FOR ITINERARY OR TRIP PLANNING:
        🌍 START LOCATION: MUST BE SPECIFIED (Required to proceed)
        🏁 DESTINATION: MUST BE SPECIFIED (Required to proceed)

        Information to Gather (ONE AT A TIME):
        1. First: Ask if they want travel tips or itinerary planning
        2. If they want travel tips:
           - Ask what specific type of travel tips they need
           - Provide focused advice on that specific area
           - Only offer one related follow-up question if relevant
        
        3. If they want itinerary planning:
           Priority Order (Ask ONE at a time):
           a. Start location (if not provided)
           b. Destination (if not provided)
           c. Travel dates
           d. Duration
           e. Travel style
           f. Number of travelers
           g. Budget
           h. Special interests
           i. Special requirements

        Question-Asking Rules:
        - CRITICAL: Ask only ONE question per response
        - Wait for user's answer before asking the next question
        - Use previous answers to determine the next most relevant question
        - Don't list multiple questions or options unless specifically asked
        - Keep questions short and clear
        - If user provides information voluntarily, skip that question and move to next priority item
        
        Response Structure:
        1. [Optional] Brief acknowledgment of user's previous answer
        2. [Optional] Brief relevant insight or confirmation
        3. ONE clear question about the next most important missing information
        4. DO NOT include "what else would you like to know?" or similar open-ended questions

        Assistant Capabilities:
        1. Route and Itinerary Planning
        2. Real-time route modifications based on preferences
        3. Local recommendations and tips
        4. Travel advice and best practices
        5. Cultural insights and destination information
        6. Transportation guidance
        7. Safety and preparation tips
        8. Budget optimization suggestions
        
        Current conversation context and trip details are provided in the prompt.
        Use this information to maintain conversation continuity and avoid asking for information already provided.

        Remember: Success is measured by how comfortable and engaged the user feels, not by how quickly you gather all information.
        """
        self.human_template = """
        Current Trip Details:
        {trip_details}

        User Query: {query}
        """
        
        self.chat_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(self.system_template),
            MessagesPlaceholder(variable_name="conversation_history"),
            HumanMessagePromptTemplate.from_template(self.human_template)
        ])

ItineraryTemplate="""
        Create a detailed travel itinerary based on the following preferences:
        {trip_details}

        Format the itinerary with:
        Convert the user's request into a detailed itinerary describing the places
        they should visit along the route and the things they should do with the campgrounds and abenity offers from the recommendations.

        Try to include the specific address of each location.

        Remember to take the user's preferences and timeframe into account, and give them an itinerary that would be fun and doable given their constraints.

        Return the itinerary as a bulleted list with clear start, end and locations along the route too.
        Return the itinerary as a bulleted list with clear timings if their query has planning of the trip mentioned else not needed.
        INCLUDE THE CAMPGROUND LOCATION AND CAMPGROUND NAMES AND ALSO ABENITY OFFERS ALONG THE ROUTES.
        Be sure to mention the type of transit for the trip.
        If specific start and end locations are not given when it is related to route plans or itinerary plans, choose ones that you think are suitable and give specific addresses.
        Your output must be the list and nothing else.
        """