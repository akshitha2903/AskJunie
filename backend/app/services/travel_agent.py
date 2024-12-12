import os
import time
import logging
import traceback
from typing import List, Dict, Optional, Union
from datetime import datetime

import openai
import googlemaps
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)

from app.api.routes import router
from config import load_secrets
from app.features.location_template import Location, EnhancedLocationParser, RouteServices
from app.features.complete_template import TripDetails,EnhancedConversationManager,EnhancedConversationTemplate,ItineraryTemplate
from app.features.detail_extraction_template import detail_extraction
from app.services.session_manager import DatabaseManager
from app.services.data_integration import DataIntegrationService

class TravelAgent:
    def __init__(self, openai_api_key, google_maps_api_key,db_manager,model="gpt-4o-mini", temperature=0.7, debug=True):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.db_manager = db_manager
        # Initialize API clients
        self.chat_model = ChatOpenAI(model=model, temperature=temperature, openai_api_key=openai_api_key)
        self.gmaps = googlemaps.Client(key=google_maps_api_key)
        
        # Initialize components
        self.location_parser = EnhancedLocationParser(self.chat_model)
        self.route_services = RouteServices(google_maps_api_key)
        self.conversation_manager = EnhancedConversationManager()
        self.conversation_template = EnhancedConversationTemplate()
        self.itinerary_template = ItineraryTemplate
        self.data_integration = DataIntegrationService(openai_api_key)
        self.current_itinerary = {}
        self.modification_history = {}
        self.detail_extraction_template = detail_extraction
        # Set up chains
        self.conversation_chain = self._setup_conversation_chain(debug)
        self.extraction_chain = self._setup_extraction_chain(debug)
        self.itinerary_chain = self._setup_itinerary_chain(debug)
    
    def _setup_conversation_chain(self, debug=True):
        return LLMChain(
            llm=self.chat_model,
            prompt=self.conversation_template.chat_prompt,
            verbose=debug,
            output_key="conversation_response"
        )

    def _setup_extraction_chain(self, debug=True):
        return LLMChain(
            llm=self.chat_model,
            prompt=ChatPromptTemplate.from_template(self.detail_extraction_template),
            verbose=debug
        )

    def _setup_itinerary_chain(self, debug=True):
        return LLMChain(
            llm=self.chat_model,
            prompt=ChatPromptTemplate.from_template(self.itinerary_template),
            verbose=debug
        )

    def _extract_trip_details(self, message,session_id):
        try:
            conversation_history = self.db_manager.get_conversation_history(session_id)
            # Get conversation history for context
            conversation_text = " ".join([msg['content'] for msg in conversation_history])
            
            extraction_result = self.extraction_chain.run(
                message=message, 
                conversation_history=conversation_text
            )
            
            extracted_details = {}
            
            for line in extraction_result.split('\n'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if value.lower() == 'none':
                        continue
                    
                    if key == 'start_location':
                        # Remove any quotes
                        value = value.strip('"')
                    elif key == 'destination':
                        # Remove any quotes
                        value = value.strip('"')
                    elif key == 'interests':
                        value = [i.strip() for i in value.split(',')]
                    elif key == 'num_travelers':
                        try:
                            value = int(value)
                        except:
                            continue
                    
                    extracted_details[key] = value
                    
            self.db_manager.update_trip_details(session_id, extracted_details)
            return extracted_details
        
        except Exception as e:
            self.logger.error(f"Error extracting trip details: {e}")
            return {}

    def handle_conversation(self, query: str, session_id: str) -> dict:
        try:
            # Get current context
            self.db_manager.create_or_update_session(session_id)
            conversation_history = self.db_manager.get_conversation_history(session_id)
            trip_details_dict = self.db_manager.get_trip_details(session_id)
            trip_details = TripDetails(**trip_details_dict)
            query_lower = query.lower()
            if 'campground' in query_lower or 'camping' in query_lower:
                return self._get_campground_or_offer_details(query, session_id, 'campgrounds')
            
            if 'offer' in query_lower or 'deals' in query_lower or 'discounts' in query_lower:
                return self._get_campground_or_offer_details(query, session_id, 'offers')
            current_itinerary = self.db_manager.get_current_itinerary(session_id)
            # If we have a current itinerary, check if this is just a conversational response
            if session_id in self.current_itinerary:
                is_conversation = self._analyze_message_type(query, conversation_history)
                if is_conversation == "casual":
                    # Generate conversational response without regenerating itinerary
                    response = self.conversation_chain.run(
                        query=query,
                        conversation_history=conversation_history,
                        trip_details=trip_details.dict()
                    )
                    
                    # Update conversation history
                    self.db_manager.add_message(session_id, "user", query)
                    self.db_manager.add_message(session_id, "assistant", response)
                    return {
                        "type": "conversation",
                        "content": {"response": response},
                        "trip_details": trip_details
                    }

            # Handle modifications
            if (current_itinerary and 
                self._analyze_message_type(query, conversation_history) == "modification"):
                modifications = self._extract_modifications(query)
                
                # Update trip details
                for key, value in modifications.items():
                    if hasattr(trip_details, key):
                        setattr(trip_details, key, value)
                
                # Save updated trip details
                self.db_manager.update_trip_details(session_id, trip_details.dict())
                
                # Generate new itinerary
                new_itinerary = self.generate_itinerary(trip_details)
                
                # Save modification history
                self.db_manager.add_modification_history(
                    session_id,
                    modifications,
                    new_itinerary
                )
                
                # Save new itinerary
                self.db_manager.save_itinerary(session_id, new_itinerary)
                return {
                    "type": "itinerary",
                    "content": new_itinerary,
                    "trip_details": trip_details
                }
            
            # Regular conversation flow
            response = self.conversation_chain.run(
                query=query,
                conversation_history=conversation_history,
                trip_details=trip_details.dict()
            )
            
            # Extract and update trip details only if message type is informational
            message_type = self._analyze_message_type(query, conversation_history)
            if message_type == "informational":
                extracted_info = self._extract_trip_details(query, session_id)
                trip_details.update(extracted_info)
                self.db_manager.update_trip_details(session_id, trip_details.dict())
                
            
            # Update conversation history
            self.db_manager.add_message(session_id, "user", query)
            self.db_manager.add_message(session_id, "assistant", response)
            
            if trip_details.check_readiness() and message_type != "casual":
                itinerary = self.generate_itinerary(trip_details)
                self.db_manager.save_itinerary(session_id, itinerary)
                return {
                    "type": "itinerary",
                    "content": itinerary,
                    "trip_details": trip_details
                }
            
            return {
                "type": "conversation",
                "content": {"response": response},
                "trip_details": trip_details
            }
                    
        except Exception as e:
            self.logger.error(f"Conversation handling error: {e}")
            return {
                "type": "conversation",
                "content": {
                    "response": "I'm having trouble processing your request. Could you rephrase or provide more details?"
                },
                "trip_details": TripDetails()
            }
            
    def _analyze_message_type(self, query: str, conversation_history: list) -> str:
        """
        Use the LLM to analyze the type of message/query from the user.
        Returns: 'casual', 'modification', or 'informational'
        """
        analysis_prompt = f"""
        Analyze the following user message in the context of a travel planning conversation.
        Previous messages: {conversation_history[-3:] if conversation_history else 'None'}
        
        Current message: {query}
        
        Classify the message into ONE of these categories:
        - casual: General conversation, general questions,travel tips, acknowledgments, thanks, or simple replies,compliments
        - modification: Requests to change or modify existing plans
        - informational: Regarding Itinerary plan or trip plans 
        
        Return only one word: casual, modification, or informational
        """
        
        try:
            result = self.chat_model.invoke(analysis_prompt).strip().lower()
            return result if result in ['casual', 'modification', 'informational'] else 'informational'
        except Exception as e:
            self.logger.error(f"Error analyzing message type: {e}")
            return 'informational'
        
    def _get_campground_or_offer_details(self, query: str, session_id: str, query_type: str):
        """
        Handle specific queries for campgrounds or offers with dynamic location extraction
        
        Args:
            query (str): User's query
            session_id (str): Current session ID
            query_type (str): 'campgrounds' or 'offers'
        
        Returns:
            dict: Response containing campgrounds or offers
        """
        try:
            # Extract location from the query
            location_extraction_prompt = f"""
            Extract the location from this query. If a location is mentioned, 
            return just the location name. If no specific location is found, 
            return NONE.

            Query: {query}
            
            Location:
            """
            # Create a separate chain for location extraction
            location_extraction_chain = LLMChain(
                llm=self.chat_model,
                prompt=location_extraction_prompt,
                conversation_history=conversation_history
            )
            # Use the extraction chain to find the location
            location_result = _extract_trip_details.run(message=query,session=session_id)
            locations = [loc.strip() for loc in location_result.split(';') if loc.strip().lower() != 'none']
            if not locations:
            # Check existing trip details if no location found
                trip_details = self.conversation_manager.get_trip_details(session_id)
                if trip_details.destination:
                    locations = [trip_details.destination]
                elif trip_details.start_location:
                    locations = [trip_details.start_location]
        
            if not locations:
                return {
                    "type": "conversation",
                    "content": {
                        "response": "I need a specific location to find nearby campgrounds. Could you please mention a city or location?"
                    },
                    "trip_details": self.conversation_manager.get_trip_details(session_id)
                }
            
            # Process each location
            all_results = []
            for location in locations:
                # Parse the location
                location_obj = self.location_parser.extract_locations(location)[0]
                
                # Find nearby attractions
                attractions = self.data_integration.enhancer.find_nearby_attractions(location_obj)
                
                # Format results for this location
                if query_type == 'campgrounds':
                    location_results = f"\nCampgrounds near {location}:\n"
                    for camp in attractions['campgrounds']:
                        location_results += f"\n{camp['name']}\n"
                        location_results += f"Description: {camp['description']['description']}\n"
                        location_results += f"Location: {camp['description']['location']}\n"
                        location_results += f"Phone: {camp['description']['phone']}\n"
                        location_results += f"Email: {camp['description']['email']}\n"
                        location_results += f"Website: {camp['description']['website']}\n"
                else:  # offers
                    location_results = f"\nOffers near {location}:\n"
                    for offer in attractions['offers']:
                        location_results += f"\n{offer['title']} at {offer['merchant']}\n"
                        location_results += f"Description: {offer['description']}\n"
                        location_results += f"Expiration: {offer.get('expiration', 'No expiration date')}\n"
                
                all_results.append(location_results)
            
            # Combine all results
            combined_response = "\n".join(all_results)
            
            return {
                "type": "conversation",
                "content": {
                    "response": combined_response
                },
                "trip_details": self.conversation_manager.get_trip_details(session_id)
            }
        
        except Exception as e:
            self.logger.error(f"Error fetching {query_type}: {e}")
            return {
                "type": "conversation",
                "content": {
                    "response": f"I'm having trouble finding {query_type}. Could you please try again?"
                },
                "trip_details": TripDetails()
            }
    def _is_modification_request(self, query: str) -> bool:
        """
        Detect if the query is requesting modifications to the existing itinerary
        """
        modification_keywords = [
            'change', 'modify', 'update', 'switch', 'instead', 
            'rather', 'prefer', 'different', 'alternative',
            'replace', 'swap', 'adjust'
        ]
        return any(keyword in query.lower() for keyword in modification_keywords)
    
    def _extract_modifications(self, query: str) -> dict:
        """
        Extract modification requests from the query
        """
        try:
            modification_prompt = f"""
            Extract specific modifications requested in this travel-related query.
            Query: {query}
            
            Return modifications as key-value pairs in the format:
            field = value
            
            Possible fields: start_location, destination, duration, budget, 
            travel_style, interests, num_travelers, special_requirements, dates
            """
            
            result = self.chat_model.predict(modification_prompt)
            
            modifications = {}
            for line in result.split('\n'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    modifications[key.strip()] = value.strip()
            
            return modifications
            
        except Exception as e:
            self.logger.error(f"Error extracting modifications: {e}")
            return {}
    def generate_itinerary(self, trip_details: TripDetails):
        """Generate a complete itinerary based on collected trip details"""
        try:
            
            itinerary_text = self.itinerary_chain.run(trip_details=trip_details.dict())
            
            # Parse locations
            locations = self.location_parser.extract_locations(itinerary_text)
            
            enhanced_itinerary=self.data_integration.enhance_itinerary_with_attractions(itinerary_text,locations)
            
            # Add geocoding data
            locations = self.route_services.geocode_locations(locations)
            
            # Get route information
            route_info = self.route_services.get_route_info(locations)
            
            # Get nearby services
            route_services = {
                "services": self.route_services.find_nearby_services(locations)
            }
            
            return {
                "success": True,
                "itinerary_text": enhanced_itinerary,
                "locations": [vars(loc) for loc in locations],
                "route_info": route_info,
                "route_services": route_services
            }
                    
        except Exception as e:
            self.logger.error(f"Error generating itinerary: {e}")
            return {
                "success": False,
                "error": "Failed to generate itinerary. Please try again."
            }
    
    def get_route_services(self, locations):
        """
        Get nearby services along the route using RouteServices
        """
        return {
            "services": self.route_services.find_nearby_services(locations)
        }
        