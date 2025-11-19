#!/usr/bin/env python3
"""
Test MongoDB connection with different SSL configurations
"""

import os
from dotenv import load_dotenv
from pymongo import MongoClient
import ssl

# Load environment variables
load_dotenv()

def test_mongodb_connections():
    """Test different MongoDB connection configurations"""
    
    base_uri = "mongodb+srv://abhijitn23beds_db_user:9vmJkSQuV6HU08F4@prisma.ojfejnc.mongodb.net/"
    
    # Test configurations
    test_configs = [
        {
            "name": "Standard SSL",
            "uri": base_uri + "?retryWrites=true&w=majority&appName=Prisma"
        },
        {
            "name": "SSL with TLS",
            "uri": base_uri + "?retryWrites=true&w=majority&appName=Prisma&ssl=true&tls=true"
        },
        {
            "name": "SSL with TLS Allow Invalid Certs",
            "uri": base_uri + "?retryWrites=true&w=majority&appName=Prisma&ssl=true&tls=true&tlsAllowInvalidCertificates=true"
        },
        {
            "name": "Direct Client Options",
            "uri": base_uri + "?retryWrites=true&w=majority&appName=Prisma",
            "client_options": {
                "ssl": True,
                "ssl_cert_reqs": ssl.CERT_NONE,
                "ssl_match_hostname": False
            }
        }
    ]
    
    for config in test_configs:
        print(f"\n🧪 Testing: {config['name']}")
        print("=" * 50)
        
        try:
            # Create client with optional SSL options
            if "client_options" in config:
                client = MongoClient(config["uri"], **config["client_options"])
            else:
                client = MongoClient(config["uri"])
            
            # Test connection with shorter timeout
            client.admin.command('ping')
            
            # Test database access
            db = client["Prisma"]
            collection = db["articles"]
            
            # Count documents (should work if connection is good)
            count = collection.count_documents({})
            
            print(f"✅ SUCCESS: Connected to MongoDB")
            print(f"   Database: Prisma")
            print(f"   Collection: articles")
            print(f"   Document count: {count}")
            
            client.close()
            return config  # Return first working configuration
            
        except Exception as e:
            print(f"❌ FAILED: {str(e)[:200]}...")
            continue
    
    print(f"\n🚨 All configurations failed!")
    return None

if __name__ == "__main__":
    print("🔍 Testing MongoDB Atlas Connection...")
    working_config = test_mongodb_connections()
    
    if working_config:
        print(f"\n🎉 RECOMMENDED CONFIGURATION:")
        print(f"Use: {working_config['name']}")
        print(f"URI: {working_config['uri']}")
    else:
        print(f"\n💡 TROUBLESHOOTING TIPS:")
        print("1. Check internet connection")
        print("2. Verify MongoDB Atlas cluster is running")
        print("3. Check firewall/antivirus settings")
        print("4. Try connecting from a different network")
