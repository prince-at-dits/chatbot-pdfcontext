#!/usr/bin/env python3
"""
Script to expose your local server using ngrok
Install ngrok first: https://ngrok.com/download
"""
import subprocess
import sys
import time

def start_ngrok():
    try:
        # Start ngrok tunnel
        print("Starting ngrok tunnel...")
        process = subprocess.Popen(['ngrok', 'http', '8081'], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE)
        
        time.sleep(3)  # Wait for ngrok to start
        
        # Get the public URL
        result = subprocess.run(['curl', '-s', 'http://localhost:4040/api/tunnels'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            if data.get('tunnels'):
                public_url = data['tunnels'][0]['public_url']
                print(f"\n✅ Your backend is now accessible at: {public_url}")
                print(f"Use this URL as BACKEND_URL in your frontend deployment")
                return public_url
        
        print("❌ Could not get ngrok URL")
        return None
        
    except FileNotFoundError:
        print("❌ ngrok not found. Install it from https://ngrok.com/download")
        return None

if __name__ == "__main__":
    start_ngrok()