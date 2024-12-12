# detail_extraction_template.py

detail_extraction = """
You are an expert at extracting travel-related location information from user messages.

Extract travel-related details from the following conversation message:

EXTRACTION RULES:
- CONTEXT: The previous location mentioned as the start location
- Current message will be assessed for:
1. Confirming start location
2. Identifying destination
3. Other trip details
4. If the user asks for Tips and other queries then extract them too

LOCATION EXTRACTION STRATEGY:
- If no previous start location exists:
* Treat the first location as start location
- If start location already exists:
* Treat the new location as destination

Conversation History:
{conversation_history}

CURRENT Message: {message}

EXTRACTION GUIDELINES:
- Prioritize extracting the destination when start location is already set
- Use context from previous messages
- Look for explicit location indicators
- If the user asks for travel tips regarding anything extract that too.

Output Format:
start_location = [start location based on context]
destination = [destination location]
duration = [extracted duration or None]
budget = [extracted budget or None]
travel_style = [extracted style or None]
interests = [extracted interests as comma-separated list or None]
num_travelers = [extracted number or None]
special_requirements = [extracted requirements or None]
dates = [extracted dates or None]
"""