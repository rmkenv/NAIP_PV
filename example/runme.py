
import os
import sys
import time
import threading
from typing import Optional

import nest_asyncio
import requests
import uvicorn
from pyngrok import ngrok

# Apply nest_asyncio for Jupyter compatibility
nest_asyncio.apply()

# Import the FastAPI instance directly to avoid module path issues
try:
    from app import app as fastapi_app
except ImportError:
    print("Error: Could not import FastAPI app. Make sure you're in the NAIP_PV directory.")
    sys.exit(1)

def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the FastAPI server with uvicorn."""
    try:
        uvicorn.run(fastapi_app, host=host, port=port, log_level="info")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)


def wait_for_server(host: str = "127.0.0.1", port: int = 8000, timeout: int = 40) -> bool:
    """Wait for the server to start up and respond to health checks."""
    print("Waiting for server to start...")
    
    for attempt in range(timeout):
        try:
            response = requests.get(f"http://{host}:{port}", timeout=1.5)
            print(f"Server health check: {response.status_code}")
            return True
        except requests.exceptions.RequestException:
            time.sleep(0.5)
    
    return False


def start_server_thread(host: str = "0.0.0.0", port: int = 8000) -> threading.Thread:
    """Start the server in a daemon thread."""
    thread = threading.Thread(target=run_server, args=(host, port), daemon=True)
    thread.start()
    return thread

def setup_ngrok_tunnel(port: int = 8000) -> Optional[str]:
    """Set up ngrok tunnel and return the public URL."""
    # Get auth token from environment variable
    auth_token = os.getenv("NGROK_AUTH_TOKEN")
    if not auth_token:
        print("Error: NGROK_AUTH_TOKEN environment variable not set.")
        print("Please set it with: export NGROK_AUTH_TOKEN='your_token_here'")
        return None
    
    try:
        # Authenticate with ngrok
        ngrok.set_auth_token(auth_token)
        
        # Create the tunnel to the running server
        public_url = ngrok.connect(addr=port, proto="http")
        print(f"Public URL: {public_url}")
        return str(public_url)
        
    except Exception as e:
        print(f"Error setting up ngrok tunnel: {e}")
        return None


def main():
    """Main function to start the server and set up the tunnel."""
    HOST = "0.0.0.0"
    PORT = 8000
    
    # Start the server in a background thread
    server_thread = start_server_thread(HOST, PORT)
    
    # Wait for server to be ready
    if not wait_for_server("127.0.0.1", PORT):
        print("Error: Server failed to start within timeout period")
        sys.exit(1)
    
    # Set up ngrok tunnel
    public_url = setup_ngrok_tunnel(PORT)
    if not public_url:
        print("Warning: Failed to set up ngrok tunnel. Server is running locally only.")
    
    print(f"Server is running on http://localhost:{PORT}")
    if public_url:
        print(f"Public access available at: {public_url}")
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        ngrok.disconnect(public_url)
        ngrok.kill()


if __name__ == "__main__":
    main()
