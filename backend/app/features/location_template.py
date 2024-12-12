import re
import logging
import requests
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class Location:
    name: str
    address: str
    description: Optional[str] = None
    is_start: bool = False
    is_end: bool = False
    lat: float = None
    lng: float = None

class EnhancedLocationParser:
    def __init__(self, llm):
        self.llm = llm
        self.logger = logging.getLogger(__name__)

    def extract_locations(self, itinerary: str) -> List[Location]:
        """
        Use LLM to intelligently extract locations from itinerary text
        
        Args:
            itinerary (str): Full itinerary text
            
        Returns:
            List[Location]: List of extracted locations with details
        """
        # Prompt for the LLM to extract locations
        location_prompt = f"""
        Extract all locations from this itinerary text. For each location, identify:
        1. The location name
        2. The complete address (if provided)
        3. Any description or relevant details about the location
        4. Whether it's the starting point or end point

        Itinerary text:
        {itinerary}

        Format each location as:
        NAME: [location name]
        ADDRESS: [full address]
        DESCRIPTION: [brief description of location and activities]
        POSITION: [START/END/WAYPOINT]
        ---
        """

        try:
            # Get LLM response
            response = self.llm.predict(location_prompt)
            
            # Parse LLM response into Location objects
            locations = []
            current_location = {}
            
            for line in response.split('\n'):
                line = line.strip()
                if not line or line == '---':
                    if current_location:
                        locations.append(Location(
                            name=current_location.get('name', 'Unknown'),
                            address=current_location.get('address', ''),
                            description=current_location.get('description', ''),
                            is_start=current_location.get('position', '').upper() == 'START',
                            is_end=current_location.get('position', '').upper() == 'END'
                        ))
                        current_location = {}
                    continue

                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    
                    if key == 'name':
                        current_location['name'] = value
                    elif key == 'address':
                        current_location['address'] = value
                    elif key == 'description':
                        current_location['description'] = value
                    elif key == 'position':
                        current_location['position'] = value

            # Add final location if exists
            if current_location:
                locations.append(Location(
                    name=current_location.get('name', 'Unknown'),
                    address=current_location.get('address', ''),
                    description=current_location.get('description', ''),
                    is_start=current_location.get('position', '').upper() == 'START',
                    is_end=current_location.get('position', '').upper() == 'END'
                ))

            # Ensure at least start and end are marked if not already
            if locations:
                if not any(loc.is_start for loc in locations):
                    locations[0].is_start = True
                if not any(loc.is_end for loc in locations):
                    locations[-1].is_end = True

            self.logger.info(f"Extracted {len(locations)} locations from itinerary")
            # validated_locations = self.validate_locations(locations)
            # return validated_locations 
            return locations
        except Exception as e:
            self.logger.error(f"Error extracting locations with LLM: {e}")
            return []

    # def validate_locations(self, locations: List[Location]) -> List[Location]:
    #     """
    #     Validate and clean extracted locations
        
    #     Args:
    #         locations (List[Location]): List of extracted locations
            
    #     Returns:
    #         List[Location]: Validated and cleaned locations
    #     """
    #     validated = []
    #     for loc in locations:
    #         # Ensure required fields are present
    #         if not loc.name or loc.name == 'Unknown':
    #             continue
                
    #         # If no address, use name as address
    #         if not loc.address:
    #             loc.address = loc.name
                
    #         validated.append(loc)
            
    #     return validated

class RouteServices:
    def __init__(self, api_key: str):
        """
        Initialize route services with Google Routes and Places API keys
        
        Args:
            api_key (str): Google Cloud API key
        """
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)
        self.routes_base_url = "https://routes.googleapis.com/directions/v2:computeRoutes"
        self.places_base_url = "https://places.googleapis.com/v1/places:searchNearby"

    def geocode_locations(self, locations: List[Location]) -> List[Location]:
        """
        Geocode locations using Google Geocoding API
        
        Args:
            locations (List[Location]): Locations to geocode
        
        Returns:
            List[Location]: Locations with latitude and longitude
        """
        geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json"
        
        for location in locations:
            try:
                params = {
                    "address": location.address,
                    "key": self.api_key
                }
                response = requests.get(geocode_url, params=params)
                geocode_result = response.json()
                
                if geocode_result['status'] == 'OK':
                    result = geocode_result['results'][0]['geometry']['location']
                    location.lat = result['lat']
                    location.lng = result['lng']
            except Exception as e:
                self.logger.error(f"Error geocoding {location.address}: {e}")
        self.logger.debug(f"Geocoded locations: {[(loc.name, loc.lat, loc.lng) for loc in locations]}")
        return locations

    def get_route_info(self, locations: List[Location]) -> Optional[Dict]:
        """
        Get comprehensive route information using Google Routes API
        
        Args:
            locations (List[Location]): List of locations to route between
        
        Returns:
            Optional[Dict]: Detailed route information
        """
        if len(locations) < 2:
            return None

        try:
            route_request = {
                "origin": {
                    "location": {
                        "latLng": {
                            "latitude": locations[0].lat,
                            "longitude": locations[0].lng
                        }
                    }
                },
                "destination": {
                    "location": {
                        "latLng": {
                            "latitude": locations[-1].lat,
                            "longitude": locations[-1].lng
                        }
                    }
                },
                "intermediates": [
                    {
                        "location": {
                            "latLng": {
                                "latitude": loc.lat,
                                "longitude": loc.lng
                            }
                        }
                    } for loc in locations[1:-1]
                ],
                "travelMode": "DRIVE",
                "routingPreference": "TRAFFIC_AWARE_OPTIMAL",
                "computeAlternativeRoutes": True
            }

            headers = {
                "Content-Type": "application/json",
                "X-Goog-Api-Key": self.api_key,
                "X-Goog-FieldMask": "routes.duration,routes.distanceMeters,routes.polyline,routes.legs"
            }

            response = requests.post(
                self.routes_base_url, 
                json=route_request, 
                headers=headers
            )

            if response.status_code == 200:
                routes_data = response.json().get('routes', [])
                route_details = {
                    "routes": [],
                    "total_routes": len(routes_data)
                }

                for route in routes_data:
                    route_info = {
                        "total_distance": route.get('distanceMeters', 0) / 1000,  # Convert to kilometers
                        "total_duration": int(route.get('duration', '0s')[:-1]),  # Remove 's' and convert to int
                        "polyline": route.get('polyline', {}).get('encodedPolyline', '')
                    }
                    route_details["routes"].append(route_info)
                
                return route_details

        except Exception as e:
            self.logger.error(f"Error getting Routes API directions: {e}")
        
        return None

    def find_nearby_services(self, locations: List[Location], service_types: List[str] = None) -> Dict[str, List[Dict]]:
        """
        Find nearby services along the route using Places API
        
        Args:
            locations (List[Location]): Route locations
            service_types (List[str], optional): Types of services to search
        
        Returns:
            Dict[str, List[Dict]]: Nearby services categorized by type
        """
        if not service_types:
            service_types = [
                'gas_station', 'restaurant', 'cafe', 
                'lodging', 'hospital', 'pharmacy', 
                'parking', 'bank', 'shopping_mall'
            ]
        
        nearby_services = {service: [] for service in service_types}
        
        for location in locations:
            if location.lat and location.lng:
                for service_type in service_types:
                    try:
                        places_request = {
                            "includedTypes": [service_type],
                            "maxResultsPerType": 3,
                            "locationRestriction": {
                                "circle": {
                                    "center": {
                                        "latitude": location.lat,
                                        "longitude": location.lng
                                    },
                                    "radius": 1000.0  # 1 km radius
                                }
                            }
                        }

                        headers = {
                            "Content-Type": "application/json",
                            "X-Goog-Api-Key": self.api_key,
                            "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.location"
                        }

                        response = requests.post(
                            self.places_base_url, 
                            json=places_request, 
                            headers=headers
                        )

                        if response.status_code == 200:
                            places_data = response.json().get('places', [])
                            for place in places_data:
                                service_info = {
                                    "name": place.get('displayName', {}).get('text', 'Unknown'),
                                    "address": place.get('formattedAddress', 'No address'),
                                    "location": {
                                        "lat": place.get('location', {}).get('latitude'),
                                        "lng": place.get('location', {}).get('longitude')
                                    }
                                }
                                nearby_services[service_type].append(service_info)

                    except Exception as e:
                        self.logger.error(f"Error finding {service_type} near {location.name}: {e}")

        return nearby_services