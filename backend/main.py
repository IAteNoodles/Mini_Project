from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset
import io
import mysql.connector
from mysql.connector import Error
import requests

# --- Database Credentials ---
hostname = "76b2td.h.filess.io"
database = "Prisma_buildeardo"
port = 3305  # Use integer for port
username = "Prisma_buildeardo"
password = "eb7f884156929ec86b5244b63ab9a35d1bcbd2f9"

# --- Pydantic Models ---
class ArticleCreate(BaseModel):
    url: str
    news_article: str
    summary: str
    bias_political: bool = False
    bias_gender: bool = False
    bias_cultural: bool = False
    bias_ideology: bool = False
    generated_by_ai: bool = False

class ArticleResponse(ArticleCreate):
    id: int

# --- FastAPI App Initialization ---
app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Database Connection and Initialization ---
def get_db_connection():
    """Establishes a new database connection using the user-provided logic."""
    try:
        connection = mysql.connector.connect(
            host=hostname,
            database=database,
            user=username,
            password=password,
            port=port,
            collation='utf8mb4_unicode_ci'
        )
        if connection.is_connected():
            return connection
    except Error as e:
        print(f"Error while connecting to MariaDB: {e}")
        # This will send a 500 error to the client if the DB is down
        raise HTTPException(status_code=500, detail=f"Database connection error: {e}")

def get_db():
    """FastAPI dependency to get a DB connection and close it after the request."""
    connection = None
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        yield connection, cursor
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

# --- API Endpoints ---
@app.get("/health")
def health_check(db = Depends(get_db)):
    conn, cursor = db
    try:
        cursor.execute("SHOW TABLES LIKE 'articles'")
        table_exists = cursor.fetchone() is not None
        return {"status": "ok", "db_initialized": table_exists}
    except Error as e:
        # This can happen if the database itself doesn't exist
        return {"status": "error", "db_initialized": False, "detail": str(e)}

@app.post("/initialize-database")
def initialize_database(db = Depends(get_db)):
    conn, cursor = db
    try:
        # Check if table exists
        cursor.execute("SHOW TABLES LIKE 'articles'")
        if cursor.fetchone():
            return {"message": "Table 'articles' already exists."}

        # If not, create it
        create_table_query = """
        CREATE TABLE articles (
            id INT AUTO_INCREMENT PRIMARY KEY,
            url VARCHAR(255),
            news_article TEXT,
            summary TEXT,
            bias_political BOOLEAN DEFAULT FALSE,
            bias_gender BOOLEAN DEFAULT FALSE,
            bias_cultural BOOLEAN DEFAULT FALSE,
            bias_ideology BOOLEAN DEFAULT FALSE,
            generated_by_ai BOOLEAN DEFAULT FALSE,
            INDEX(url)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """
        cursor.execute(create_table_query)
        conn.commit()
        return {"message": "Table 'articles' created successfully."}
    except Error as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database operation failed: {e}")


@app.post("/articles/", response_model=ArticleResponse)
def create_article(article: ArticleCreate, db = Depends(get_db)):
    conn, cursor = db
    query = """
    INSERT INTO articles (url, news_article, summary, bias_political, bias_gender, bias_cultural, bias_ideology, generated_by_ai)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    values = (article.url, article.news_article, article.summary, article.bias_political, article.bias_gender, article.bias_cultural, article.bias_ideology, article.generated_by_ai)
    
    try:
        cursor.execute(query, values)
        conn.commit()
        new_id = cursor.lastrowid
        return ArticleResponse(id=new_id, **article.model_dump())
    except Error as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {e}")

@app.get("/articles/", response_model=list[ArticleResponse])
def read_articles(skip: int = 0, limit: int = 100, db = Depends(get_db)):
    _, cursor = db
    query = "SELECT * FROM articles LIMIT %s OFFSET %s"
    cursor.execute(query, (limit, skip))
    articles = cursor.fetchall()
    return articles

def get_articles_as_df(db):
    conn, cursor = db
    query = "SELECT * FROM articles"
    cursor.execute(query)
    articles = cursor.fetchall()
    return pd.DataFrame(articles)

@app.get("/articles/csv")
def export_articles_csv(db = Depends(get_db)):
    df = get_articles_as_df(db)
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=articles.csv"
    return response

@app.get("/articles/parquet")
def export_articles_parquet(db = Depends(get_db)):
    df = get_articles_as_df(db)
    stream = io.BytesIO()
    df.to_parquet(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]), media_type="application/octet-stream")
    response.headers["Content-Disposition"] = "attachment; filename=articles.parquet"
    return response

@app.get("/articles/dataset")
def export_articles_dataset(db = Depends(get_db)):
    df = get_articles_as_df(db)
    if df.empty:
        return {"data": []}
    # Convert boolean columns from 0/1 to True/False if needed
    for col in df.columns:
        if df[col].dtype == 'int64' and set(df[col].unique()) <= {0, 1}:
             df[col] = df[col].astype(bool)
    dataset = Dataset.from_pandas(df)
    return dataset.to_dict()

@app.get("/news")
def get_news(q: str):
    news_api_key = "8f51adebb6f54e3da3861e73e3efa150"
    url = f"https://newsapi.org/v2/everything?q={q}&apiKey={news_api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Error connecting to NewsAPI: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
