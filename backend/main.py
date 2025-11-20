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
from bias_detector import get_detector
import ollama
import numpy as np

import logging
import time
import string

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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

class AnalyzeRequest(BaseModel):
    text: str
    explanation_mode: str = "SHAP"

class ExplainRequest(BaseModel):
    text: str
    label: str
    explanation_mode: str
    tokens: list[str]
    values: list[float]

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
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Error connecting to NewsAPI: {e}"
                            )

# --- Bias Analysis Logic ---
LLM_MODEL_NAME = "granite4:micro-h"
TOP_FACTOR_LIMIT = 5

def _format_contributions_for_prompt(df: pd.DataFrame, value_column: str, limit: int = 15) -> str:
    working = df.copy()
    working["abs"] = working[value_column].abs()
    trimmed = working.sort_values("abs", ascending=False).head(limit)
    return "\n".join(
        f"- {row['Token']}: {row[value_column]:.4f}" for _, row in trimmed.drop(columns=["abs"]).iterrows()
    )

def _summarize_with_granite(
    *,
    text: str,
    friendly_label: str,
    method: str,
    contributions: pd.DataFrame,
    contribution_column: str,
) -> str:
    try:
        import ollama
    except ImportError:
        return "Install the `ollama` package and ensure Granite4 is pulled to enable narrative summaries."

    contribution_block = _format_contributions_for_prompt(contributions, contribution_column)
    
    if method == "SHAP":
        prompt = f"""
You are a media literacy analyst. The model has classified this article as **{friendly_label}**.

ARTICLE TEXT:
\"\"\"{text}\"\"\"

METHOD: SHAP (SHapley Additive exPlanations)
CONTEXT: SHAP values represent the marginal contribution of each word to the final prediction. High positive values mean the word strongly pushes the model towards the '{friendly_label}' label.

KEY INFLUENCERS (Token: SHAP Value):
{contribution_block}

INSTRUCTIONS:
1. **Explain the Decision (SHAP)**: Explain how the specific words listed above mathematically contributed to the **{friendly_label}** classification. Discuss the "push and pull" of these words on the score.
2. **Analyze Tone**: Discuss how these high-impact words shape the article's tone.
3. **Strictly No Advice**: Do not mention rewriting, improving, or fixing the text. Do not state that the text is fine as is. Stop after the tone analysis.

Output Markdown.
"""
    elif method == "LIME":
        prompt = f"""
You are a media literacy analyst. The model has classified this article as **{friendly_label}**.

ARTICLE TEXT:
\"\"\"{text}\"\"\"

METHOD: LIME (Local Interpretable Model-agnostic Explanations)
CONTEXT: LIME identifies the words that are most influential in the local context. These are the words that, if removed or changed, would most significantly alter the prediction.

KEY INFLUENCERS (Token: LIME Weight):
{contribution_block}

INSTRUCTIONS:
1. **Explain the Decision (LIME)**: Explain why the presence of the words listed above makes the article **{friendly_label}**. Discuss them as the "triggers" for this classification.
2. **Analyze Tone**: Discuss how these trigger words affect the overall tone.
3. **Strictly No Advice**: Do not mention rewriting, improving, or fixing the text. Do not state that the text is fine as is. Stop after the tone analysis.

Output Markdown.
"""
    elif method == "Attention":
        prompt = f"""
You are a media literacy analyst. The model has classified this article as **{friendly_label}**.

ARTICLE TEXT:
\"\"\"{text}\"\"\"

METHOD: Self-Attention Mechanism
CONTEXT: Attention weights show where the model "looked" while processing the text. High attention means the model focused heavily on these words to derive its meaning and classification.

KEY INFLUENCERS (Token: Attention Weight):
{contribution_block}

INSTRUCTIONS:
1. **Explain the Decision (Attention)**: Explain why the model focused its "gaze" on the words listed above to conclude the article is **{friendly_label}**. Why are these words the most salient?
2. **Analyze Tone**: Discuss why these focal points are critical to the article's tone.
3. **Strictly No Advice**: Do not mention rewriting, improving, or fixing the text. Do not state that the text is fine as is. Stop after the tone analysis.

Output Markdown.
"""
    else:
        # Fallback prompt
        prompt = f"""
You are a media literacy analyst. The model has classified this article as **{friendly_label}**.

ARTICLE TEXT:
\"\"\"{text}\"\"\"

KEY INFLUENCERS (Token: Contribution Value):
{contribution_block}

INSTRUCTIONS:
1. **Explain the Bias**: Using the key influencers and their values, explain *why* the article was classified as {friendly_label}. Discuss how specific words (and their high contribution values) signal this bias.
2. **Analyze Tone**: Discuss how the identified words affect the overall tone.
3. **Strictly No Advice**: Do not mention rewriting, improving, or fixing the text. Do not state that the text is fine as is. Stop after the tone analysis.

Output Markdown.
"""

    try:
        response = ollama.chat(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful AI that explains bias detection results using SHAP/LIME/Attention values."},
                {"role": "user", "content": prompt},
            ],
        )
        message = response.get("message", {})
        return message.get("content", "Granite4 did not return any content.").strip()
    except Exception as exc:
        return f"Unable to query Granite4: {exc}"

@app.post("/explain")
def explain_analysis(request: ExplainRequest):
    # Reconstruct DataFrame from request
    contributions = pd.DataFrame({
        "Token": request.tokens,
        "Contribution": request.values
    })
    
    narrative = _summarize_with_granite(
        text=request.text,
        friendly_label=request.label,
        method=request.explanation_mode,
        contributions=contributions,
        contribution_column="Contribution",
    )
    return {"narrative": narrative}

@app.post("/analyze")
def analyze_text(request: AnalyzeRequest):
    logger.info(f"Received analysis request. Mode: {request.explanation_mode}")
    start_time = time.time()

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    detector = get_detector()
    # Truncate text to avoid model errors
    text_to_analyze = detector.truncate(request.text)
    logger.info(f"Text truncated to {len(text_to_analyze)} chars.")

    logger.info("Running prediction...")
    prediction = detector.predict(text_to_analyze)
    probabilities = detector.predict_proba([text_to_analyze])[0]
    friendly_label = detector.display_label(prediction.label)
    logger.info(f"Prediction done. Label: {friendly_label}, Score: {prediction.score}")
    
    contribution_column = "Contribution"
    explanation_data = {"tokens": [], "values": []}
    explanation_table = None
    heatmap_html = None

    if request.explanation_mode == "LIME":
        logger.info("Generating LIME explanation (1000 samples)...")
        explanation = detector.explain_lime(text_to_analyze, num_features=20, num_samples=1000)
        contributions = explanation.as_list()
        explanation_table = pd.DataFrame(contributions, columns=["Token", contribution_column])
        explanation_data = {
            "tokens": [c[0] for c in contributions],
            "values": [c[1] for c in contributions]
        }
    elif request.explanation_mode == "Attention":
        logger.info("Extracting Attention weights...")
        raw_tokens, values, heads_data = detector.explain_attention(text_to_analyze)
        
        # Generate Heatmap HTML
        heatmap_html = detector.generate_attention_html(raw_tokens, values)
        
        # Clean tokens for the chart
        cleaned_tokens = [t.replace('Ġ', '') for t in raw_tokens]
        
        # Filter out punctuation and special tokens for the chart/heatmap
        indices_to_keep = [
            i for i, t in enumerate(cleaned_tokens) 
            if t.strip() not in string.punctuation and t.strip() not in ["<s>", "</s>", "<pad>", ""]
        ]
        
        filtered_tokens = [cleaned_tokens[i] for i in indices_to_keep]
        filtered_values = values[indices_to_keep]
        
        # Create DataFrame with original indices to map back to heads
        explanation_table = pd.DataFrame({
            "Token": filtered_tokens, 
            contribution_column: filtered_values,
            "OriginalIndex": indices_to_keep
        })
        
        # Sort by absolute value for the chart
        explanation_table["abs"] = explanation_table[contribution_column].abs()
        sorted_df = explanation_table.sort_values("abs", ascending=False).head(20)
        
        # Filter head data to only include top 20 tokens
        # heads_data is [num_heads, seq_len]
        top_indices = sorted_df["OriginalIndex"].tolist()
        top_heads_data = heads_data[:, top_indices]

        explanation_data = {
            "tokens": sorted_df["Token"].tolist(),
            "values": sorted_df[contribution_column].tolist(),
            "heads": top_heads_data.tolist(), # [num_heads, 20]
        }
        # No heatmap HTML for attention yet, or we could generate a custom one
        # heatmap_html = None 
    else:
        logger.info("Generating SHAP explanation...")
        # Limit max_evals to prevent timeouts
        shap_explanation = detector.shap_explain(text_to_analyze, max_evals=300)
        target_label = prediction.label
        shap_df = detector.shap_dataframe(shap_explanation, target_label)
        explanation_table = shap_df.rename(columns={"SHAP Value": contribution_column})
        
        # Generate Heatmap HTML
        heatmap_html = detector.shap_text_html(shap_explanation, target_label)
        
        # Sort by absolute value for the chart
        explanation_table["abs"] = explanation_table[contribution_column].abs()
        sorted_df = explanation_table.sort_values("abs", ascending=False).head(20)
        explanation_data = {
            "tokens": sorted_df["Token"].tolist(),
            "values": sorted_df[contribution_column].tolist()
        }
    
    logger.info("Explanation generated.")

    # Narrative is now generated on demand via /explain endpoint
    narrative = None

    probs_formatted = [
        {"label": detector.display_label(label), "value": float(prob)}
        for label, prob in zip(detector.class_names, probabilities)
    ]
    probs_formatted.sort(key=lambda x: x["value"], reverse=True)

    elapsed = time.time() - start_time
    logger.info(f"Analysis complete in {elapsed:.2f}s")

    return {
        "label": friendly_label,
        "score": float(prediction.score),
        "probabilities": probs_formatted,
        "explanation": explanation_data,
        "heatmap_html": heatmap_html,
        "narrative": narrative
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
