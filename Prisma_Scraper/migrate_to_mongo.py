import os
import csv
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

def migrate_csv_to_mongo():
    """
    Migrates data from articles.csv to a MongoDB collection.
    """
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri or mongo_uri == "YOUR_MONGO_CONNECTION_STRING_HERE":
        print("Error: MONGO_URI not found or not set in .env file.")
        return

    csv_file = 'articles.csv'
    if not os.path.exists(csv_file):
        print(f"'{csv_file}' not found. No data to migrate.")
        return

    try:
        client = MongoClient(mongo_uri)
        # The default database is usually the one specified in the connection string.
        db = client.get_database("Prisma")
        collection = db.articles

        print("Starting migration from CSV to MongoDB...")

        with open(csv_file, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            articles_to_migrate = list(reader)
            
            if not articles_to_migrate:
                print("CSV file is empty. Nothing to migrate.")
                client.close()
                return

            migrated_count = 0
            for article in articles_to_migrate:
                # Use URL as a unique key to prevent duplicates
                result = collection.update_one(
                    {'url': article['url']},
                    {'$setOnInsert': {'text': article['text']}},
                    upsert=True
                )
                if result.upserted_id:
                    migrated_count += 1
        
        print(f"Migration complete. {migrated_count} new articles were added to the database.")
        client.close()

    except Exception as e:
        print(f"An error occurred during migration: {e}")

if __name__ == "__main__":
    migrate_csv_to_mongo()
