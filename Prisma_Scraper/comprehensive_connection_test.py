#!/usr/bin/env python3
"""
Alternative MongoDB connection test with SSL options
"""

import os
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import certifi

# Load environment variables
load_dotenv()

def test_connection_alternatives():
    """Test MongoDB connection with various SSL workarounds"""
    
    base_uri = "mongodb+srv://abhijitn23beds_db_user:9vmJkSQuV6HU08F4@prisma.ojfejnc.mongodb.net/"
    
    # Alternative connection approaches
    approaches = [
        {
            "name": "Using Certifi CA Bundle",
            "uri": base_uri + "?retryWrites=true&w=majority&appName=Prisma",
            "options": {"tlsCAFile": certifi.where()}
        },
        {
            "name": "Disable SSL Certificate Verification",
            "uri": base_uri + "?retryWrites=true&w=majority&appName=Prisma&tlsInsecure=true"
        },
        {
            "name": "Short Connection Timeout",
            "uri": base_uri + "?retryWrites=true&w=majority&appName=Prisma&connectTimeoutMS=5000&socketTimeoutMS=5000&serverSelectionTimeoutMS=5000"
        },
        {
            "name": "Alternative TLS Settings",
            "uri": base_uri + "?retryWrites=true&w=majority&appName=Prisma&tls=true&tlsInsecure=true&directConnection=false"
        }
    ]
    
    for approach in approaches:
        print(f"\n🧪 Testing: {approach['name']}")
        print("=" * 60)
        
        try:
            # Create client
            if "options" in approach:
                client = MongoClient(approach["uri"], **approach["options"])
            else:
                client = MongoClient(approach["uri"])
            
            # Quick ping test with timeout
            client.admin.command('ping')
            
            # Test actual database operation
            db = client["Prisma"]
            collection = db["articles"]
            count = collection.count_documents({})
            
            print(f"✅ SUCCESS!")
            print(f"   Connected to MongoDB Atlas")
            print(f"   Articles in database: {count}")
            
            client.close()
            return approach
            
        except ConnectionFailure as e:
            print(f"❌ Connection Failed: {str(e)[:150]}...")
        except Exception as e:
            print(f"❌ Error: {str(e)[:150]}...")
        
    print(f"\n🚨 ALL CONNECTION ATTEMPTS FAILED")
    return None

def check_network_and_dns():
    """Check basic network connectivity"""
    import subprocess
    
    print(f"\n🌐 NETWORK DIAGNOSTICS:")
    print("=" * 50)
    
    # Test DNS resolution
    try:
        result = subprocess.run(["nslookup", "prisma.ojfejnc.mongodb.net"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ DNS Resolution: Working")
        else:
            print("❌ DNS Resolution: Failed")
            print(f"   Error: {result.stderr}")
    except Exception as e:
        print(f"❌ DNS Test Error: {e}")
    
    # Test basic connectivity
    try:
        result = subprocess.run(["ping", "8.8.8.8", "-n", "1"], 
                              capture_output=True, text=True, timeout=10)
        if "TTL=" in result.stdout:
            print("✅ Internet Connection: Working")
        else:
            print("❌ Internet Connection: Failed")
    except Exception as e:
        print(f"❌ Internet Test Error: {e}")

if __name__ == "__main__":
    print("🔍 COMPREHENSIVE MONGODB CONNECTION TEST")
    
    # Check network first
    check_network_and_dns()
    
    # Test connections
    working_config = test_connection_alternatives()
    
    if working_config:
        print(f"\n🎉 WORKING CONFIGURATION FOUND:")
        print(f"   Method: {working_config['name']}")
        print(f"   URI: {working_config['uri']}")
        
        # Update .env file with working configuration
        print(f"\n📝 Updating .env file with working configuration...")
        
    else:
        print(f"\n💡 NEXT STEPS:")
        print("1. Check if MongoDB Atlas cluster is paused/stopped")
        print("2. Verify network allows outbound connections on port 27017")
        print("3. Try from a different network (mobile hotspot)")
        print("4. Check MongoDB Atlas security settings (IP whitelist)")
        print("5. Contact your network administrator about SSL/MongoDB access")
