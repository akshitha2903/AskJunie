from typing import List, Dict, Optional, Any
from pydantic import BaseModel
import numpy as np
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from typing import List, Dict
from pydantic import BaseModel
from geopy.distance import geodesic
from config import load_secrets,connect_to_db
# Initialize FastAPI application
import tempfile
tempfile.tempdir = "D:/mysql_temp"
secrets = load_secrets()
openai_api_key=secrets["OPENAI_API_KEY"]
class ModifiedPromptTemplate(BaseModel):
    def __init__(__pydantic_self__, **data: Any) -> None:
        registered, not_registered = __pydantic_self__.filter_data(data)
        super().__init__(**registered)
        for k, v in not_registered.items():
            __pydantic_self__.__dict__[k] = v
    
    @classmethod
    def filter_data(cls, data):
        registered_attr = {}
        not_registered_attr = {}
        annots = cls.__annotations__
        for k, v in data.items():
            if k in annots:
                registered_attr[k] = v
            else:
                not_registered_attr[k] = v
        return registered_attr, not_registered_attr

class RAGTravelDatabase:
    def __init__(self, openai_api_key: str):
        self.conn = connect_to_db()
        self.cursor = self.conn.cursor(dictionary=True)
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.llm = ChatOpenAI(openai_api_key=openai_api_key)
        self.vector_stores = {}
        self._initialize_vector_stores()

    def _initialize_vector_stores(self):
        """Initialize vector stores for different content types"""
        # Campgrounds
        campground_data = self._get_campground_documents()
        self.vector_stores['campgrounds'] = self._create_vector_store(campground_data)

        # Offers
        offer_data = self._get_offer_documents()
        self.vector_stores['offers'] = self._create_vector_store(offer_data)

    def _get_campground_documents(self) -> List[str]:
        """Get campground data as documents with basic information"""
        query = """
        SELECT 
            c.*
        FROM campgrounds c
        """
        self.cursor.execute(query)
        campgrounds = self.cursor.fetchall()
        
        documents = []
        for camp in campgrounds:
            # Construct a simplified document string
            doc = f"""CAMP: {camp['name']}
                    Description: {camp['description']}
                    Location: {camp['address']}, {camp['city']}, {camp['state']}
                    Phone: {camp['phone'] or 'No phone'}
                    Email: {camp['email'] or 'No email'}
                    Website: {camp['website'] or 'No website'}
                    """
            documents.append(doc)
        
        return documents
    def _get_offer_documents(self) -> List[str]:
        """Get Abenity offers as documents"""
        query = """
        SELECT 
            ao.*,
            am.name as merchant_name,
            apc.title as category,
            aol.city,
            aol.state
        FROM abenity_offers ao
        JOIN abenity_merchants am ON ao.abenity_merchant_id = am.id
        JOIN abenity_offer_locations aol ON ao.id = aol.abenity_offer_id
        JOIN abenity_perk_categories apc ON ao.abenity_perk_category_id = apc.id
        """
        self.cursor.execute(query)
        offers = self.cursor.fetchall()
        
        documents = []
        for offer in offers:
            doc = f"""Offer: {offer['title']}
            Merchant: {offer['merchant_name']}
            Category: {offer['category']}
            Location: {offer['city']}, {offer['state']}
            Description: {offer['link']}
            Expiration: {offer['exp_date']}
            """
            documents.append(doc)
        
        return documents

    def _create_vector_store(self, documents: List[str]) -> FAISS:
        """Create FAISS vector store from documents"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.create_documents(documents)
        return FAISS.from_documents(texts, self.embeddings)


class Location(BaseModel):
    name: str
    lat: float
    lng: float

class EnhancedItineraryGenerator:
    def __init__(self, rag_db: RAGTravelDatabase):
        self.rag_db = rag_db
        self.similarity_threshold = 0.8  # Define threshold as class attribute for easy modification

    def find_nearby_attractions(self, location: Location) -> Dict[str, List[str]]:
        attractions = {
            'campgrounds': [],
            'offers': []
        }

        location_query = f"Find locations near {location.name}"

        campground_results = self.rag_db.vector_stores['campgrounds'].similarity_search_with_score(
            location_query, 
            k=5
        )
        
        offer_results = self.rag_db.vector_stores['offers'].similarity_search_with_score(
            location_query,
            k=5
        )

        # Process campgrounds with score threshold
        seen_campgrounds = set()
        for doc, score in campground_results:
            if score < self.similarity_threshold:  # Apply same threshold check
                camp_info = self.extract_campground_info(doc.page_content)
                
                camp_key = f"{camp_info['name']}-{camp_info.get('location', '')}"
                if camp_key not in seen_campgrounds:
                    seen_campgrounds.add(camp_key)
                    
                    coords = self._extract_coordinates(doc.page_content)
                    if coords:
                        distance = geodesic((location.lat, location.lng), coords).miles
                    else:
                        distance = "Unknown"
                    
                    attractions['campgrounds'].append({
                        'name': camp_info['name'],
                        'description': {
                            'description': camp_info['description'],
                            'phone': camp_info['phone'],
                            'email': camp_info['email'],
                            'website': camp_info['website'],
                            'location': camp_info['location'] 
                        },
                        'distance': distance
                    })

        # Process offers (now matching campground logic)
        seen_offers = set()
        for doc, score in offer_results:
            if score < self.similarity_threshold:  # Same threshold
                offer_info = self.extract_offer_info(doc.page_content)
                
                offer_key = f"{offer_info['title']}-{offer_info['merchant']}"
                if offer_key not in seen_offers:
                    seen_offers.add(offer_key)
                    attractions['offers'].append({
                        'title': offer_info['title'],
                        'merchant': offer_info['merchant'],
                        'description': offer_info['description'],
                        'expiration': offer_info['expiration']
                    })

        print(f"Attraction Campgrounds: {attractions['campgrounds']}")
        print(f"Attraction Offers: {attractions['offers']}")
        return attractions

    def enhance_itinerary(self, itinerary_text: str, locations: List[Location]) -> str:
        enhanced_itinerary = itinerary_text + "\n\nNearby Campgrounds:\n"
        
        for location in locations:
            attractions = self.find_nearby_attractions(location)
            enhanced_itinerary += f"\nNear {location.name}:\n"
            
            if attractions['campgrounds']:
                enhanced_itinerary += "\nCampgrounds:\n"
                for camp in attractions['campgrounds']:
                    distance_text = f"({camp['distance']:.1f} miles away)" if isinstance(camp['distance'], (int, float)) else ""
                    enhanced_itinerary += f"- {camp['name']} {distance_text}\n"
                    enhanced_itinerary += f"  Description: {camp['description']['description']}\n"
                    enhanced_itinerary += f"  Location: {camp['description']['location']}\n"
                    enhanced_itinerary += f"  Phone: {camp['description']['phone']}\n"
                    enhanced_itinerary += f"  Email: {camp['description']['email']}\n"
                    enhanced_itinerary += f"  Website: {camp['description']['website']}\n"
            
            if attractions['offers']:
                enhanced_itinerary += "\nLocal Offers:\n"
                for offer in attractions['offers']:
                    enhanced_itinerary += f"- {offer['title']} at {offer['merchant']}\n"
                    enhanced_itinerary += f"  Description: {offer['description']}\n"
                    enhanced_itinerary += f"  Expiration: {offer.get('expiration', 'No expiration date available')}\n"
                    if 'location' in offer:
                        enhanced_itinerary += f"  Location: {offer['location']}\n"
        
        return enhanced_itinerary

    def _extract_coordinates(self, content: str) -> Optional[tuple]:
        try:
            coords_section = content.split('Coordinates:')[1].split('\n')[0].strip()
            lat, lng = map(float, coords_section.split(','))
            return (lat, lng)
        except:
            return None

    def extract_campground_info(self,content: str) -> dict:
        """
        Extract multiple campground details from document content including name,
        description, location, contact information, and website.
        
        Args:
            content (str): Raw document content containing campground information
            
        Returns:
            dict: Dictionary containing extracted campground details with fallback values for missing fields
        """
        fields = {
            'name': ('CAMP:', 'Unknown Campground'),
            'description': ('Description:', 'No description available'),
            'location': ('Location:', 'Unknown Location'),
            'phone': ('Phone:', 'No phone available'),
            'email': ('Email:', 'No email available'),
            'website': ('Website:', 'No website available')
        }
        
        result = {}
        
        for field, (prefix, default) in fields.items():
            try:
                result[field] = content.split(prefix)[1].split('\n')[0].strip()
            except (IndexError, AttributeError):
                result[field] = default
        
        return result
    
    def extract_offer_info(self,content: str) -> dict:
        fields = {
            'title': ('Offer:', 'Unknown Offer'),
            'merchant': ('Merchant:', 'Unknown Merchant'),
            'location': ('Location:', 'Unknown Location'),
            'description': ('Description:', 'No description available'),
            'expiration': ('Expiration:', 'No expiration available')
        }
        
        result = {}
        
        for field, (prefix, default) in fields.items():
            try:
                result[field] = content.split(prefix)[1].split('\n')[0].strip()
            except (IndexError, AttributeError):
                result[field] = default
                
        return result
    
class DataIntegrationService:
    def __init__(self, openai_api_key: str):
        self.rag_db = RAGTravelDatabase(openai_api_key)
        self.enhancer = EnhancedItineraryGenerator(self.rag_db)
    
    def enhance_itinerary_with_attractions(self, itinerary_text: str, locations: List[Location]) -> str:
        """
        Wrapper method to enhance itinerary with nearby attractions
        """
        return self.enhancer.enhance_itinerary(itinerary_text, locations)

    