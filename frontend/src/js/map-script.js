let map;
let markers = [];
let directionsService;
let directionsRenderer;
// Initialize session ID as null
let currentSessionId = null;

// Function to fetch session ID from server
async function initializeSession() {
    try {
        const response = await fetch('/api/init-session', { method: 'POST' });
        const data = await response.json();
        currentSessionId = data.session_id;
        return currentSessionId;
    } catch (error) {
        console.error('Error initializing session:', error);
        return 'session_' + crypto.randomUUID();
    }
}
async function initMap() {
    await initializeSession();
    // Initialize map with a default center (will be updated)
    map = new google.maps.Map(document.getElementById('map'), {
        zoom: 13,
        center: { lat: 0, lng: 0 } // temporary center
    });
    
    directionsService = new google.maps.DirectionsService();
    directionsRenderer = new google.maps.DirectionsRenderer();
    directionsRenderer.setMap(map);

    // Try to get user's location
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(
            // Success callback
            (position) => {
                const userLocation = {
                    lat: position.coords.latitude,
                    lng: position.coords.longitude
                };
                
                // Center map on user's location
                map.setCenter(userLocation);
                
                // Add a marker for user's location
                new google.maps.Marker({
                    position: userLocation,
                    map: map,
                    title: 'Your Location',
                    icon: 'http://maps.google.com/mapfiles/ms/icons/purple-dot.png'
                });
            },
            // Error callback
            (error) => {
                const errorDiv = document.getElementById('location-error');
                errorDiv.style.display = 'block';
                
                switch(error.code) {
                    case error.PERMISSION_DENIED:
                        errorDiv.textContent = "Location access was denied. Using default map view.";
                        break;
                    case error.POSITION_UNAVAILABLE:
                        errorDiv.textContent = "Location information unavailable. Using default map view.";
                        break;
                    case error.TIMEOUT:
                        errorDiv.textContent = "Location request timed out. Using default map view.";
                        break;
                    default:
                        errorDiv.textContent = "An unknown error occurred getting location. Using default map view.";
                }
                
                // Fall back to a reasonable default view
                map.setCenter({ lat: 40.7128, lng: -74.0060 }); // New York City
                map.setZoom(10);
            }
        );
    } else {
        // Browser doesn't support geolocation
        document.getElementById('location-error').textContent = 
            "Your browser doesn't support geolocation. Using default map view.";
        document.getElementById('location-error').style.display = 'block';
        
        // Fall back to default view
        map.setCenter({ lat: 40.7128, lng: -74.0060 }); // New York City
        map.setZoom(10);
    }
}

function clearMap() {
    markers.forEach(marker => marker.setMap(null));
    markers = [];
    directionsRenderer.setDirections({routes: []});
}

function generateItinerary() {
    const queryInput = document.getElementById('query');
    const query = queryInput.value.trim();
    
    if (!query) return;

    // Add user message to chat display
    addMessageToChat('user', query);
    
    // Clear input
    queryInput.value = '';

    // Add session_id as a query parameter
    fetch('/api/chat', {  
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
            query: query,
            session_id: currentSessionId
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.type === 'itinerary' && data.content.success) {
            // If it's an itinerary, display route and markers
            displayResults(data.content);
            addMessageToChat('assistant', data.content.itinerary_text);
        } else if (data.type === 'conversation') {
            // If it's a conversation, display the text response
            addMessageToChat('assistant', data.content.response);
        }

        // Update trip details display if provided
        if (data.trip_details) {
            updateTripDetailsDisplay(data.trip_details);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        addMessageToChat('assistant', 'An error occurred while processing your request. Please try again.');
    });
}

function addMessageToChat(role, message) {
    const chatContainer = document.getElementById('chat-history');
    const messageElement = document.createElement('div');
    messageElement.classList.add('chat-message', role);
    
    // Create message content
    const messageContent = document.createElement('p');
    messageContent.textContent = message;
    
    // Create role indicator
    const roleIndicator = document.createElement('span');
    roleIndicator.classList.add('role-indicator');
    roleIndicator.textContent = role === 'user' ? 'You' : 'AskJunie';
    
    messageElement.appendChild(roleIndicator);
    messageElement.appendChild(messageContent);
    
    chatContainer.appendChild(messageElement);
    
    // Scroll to bottom of chat
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function updateTripDetailsDisplay(tripDetails) {
    const detailsContainer = document.getElementById('trip-details');
    if (!detailsContainer) return;

    let detailsHtml = '<h3>Trip Details</h3>';
    const details = {
        'Destination': tripDetails.destination,
        'Duration': tripDetails.duration,
        'Budget': tripDetails.budget,
        'Travel Style': tripDetails.travel_style,
        'Interests': tripDetails.interests?.join(', '),
        'Number of Travelers': tripDetails.num_travelers,
        'Special Requirements': tripDetails.special_requirements,
        'Dates': tripDetails.dates
    };

    for (const [key, value] of Object.entries(details)) {
        if (value) {
            detailsHtml += `<p><strong>${key}:</strong> ${value}</p>`;
        }
    }

    detailsContainer.innerHTML = detailsHtml;
}

function displayResults(data) {
    clearMap();
    
    let validLocations = [];
    
    // Add markers for each location
    data.locations.forEach(location => {
        if (location.lat && location.lng) {
            const position = { lat: location.lat, lng: location.lng };
            validLocations.push(position);
            
            const marker = new google.maps.Marker({
                position: position,
                map: map,
                title: location.name,
                icon: location.is_start ? 'http://maps.google.com/mapfiles/ms/icons/green-dot.png' :
                      location.is_end ? 'http://maps.google.com/mapfiles/ms/icons/red-dot.png' :
                      'http://maps.google.com/mapfiles/ms/icons/blue-dot.png'
            });
            
            const infowindow = new google.maps.InfoWindow({
                content: `
                    <div style="padding: 10px;">
                        <h3 style="margin: 0 0 10px 0;">${location.name}</h3>
                        <p style="margin: 0;">${location.description || ''}</p>
                        <p style="margin: 5px 0 0 0; color: #666;">${location.address}</p>
                    </div>
                `
            });
            
            marker.addListener('click', () => {
                infowindow.open(map, marker);
            });
            
            markers.push(marker);
        }
    });
    
    // Display route if we have valid locations
    if (validLocations.length >= 2) {
        const waypoints = validLocations.slice(1, -1).map(location => ({
            location: location,
            stopover: true
        }));
        
        const request = {
            origin: validLocations[0],
            destination: validLocations[validLocations.length - 1],
            waypoints: waypoints,
            travelMode: 'DRIVING',
            optimizeWaypoints: true
        };
        
        directionsService.route(request, function(result, status) {
            if (status === 'OK') {
                directionsRenderer.setDirections(result);
            } else {
                console.error('Directions request failed due to ' + status);
            }
        });
    }
    
    // Fit map to markers
    if (markers.length > 0) {
        const bounds = new google.maps.LatLngBounds();
        markers.forEach(marker => bounds.extend(marker.getPosition()));
        map.fitBounds(bounds);
    }
}

// Add event listener for Enter key
document.getElementById('query').addEventListener('keypress', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        generateItinerary();
    }
});

// Initialize map when page loads
window.onload = initMap;