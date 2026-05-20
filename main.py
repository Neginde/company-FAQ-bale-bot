import sys
import os
import time
import logging
import threading
from datetime import datetime
from typing import List, Dict, Optional
import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from search_engine import FAQSearchEngine
# 1. Environment & Logging Configuration
BOT_TOKEN = os.getenv("BALE_BOT_TOKEN")
if not BOT_TOKEN:
    raise ValueError("Environment variable 'BALE_BOT_TOKEN' not found.")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from logger import FAQLogger
    LOGGER_AVAILABLE = True
    logger.info("Custom FAQLogger successfully activated.")
except ImportError:
    LOGGER_AVAILABLE = False
    logger.warning("Custom logger not found - falling back to standard Python logging.")

# 2. Search Engine Instance Initialization
try:
    search_engine = FAQSearchEngine()
except Exception as e:
    logger.error(f"Failed to initialize search engine dependency: {e}")
    search_engine = None

# 3. Messenger Integration Layer (BaleFAQBot)
class BaleFAQBot:
    def __init__(self, token: str, engine: FAQSearchEngine):
        self.token = token
        self.base_url = f"https://tapi.bale.ai/bot{self.token}/"
        self.offset = 0
        self.engine = engine
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "EnterpriseFAQBot/1.0",
            "Accept": "application/json",
            "Content-Type": "application/json"
        })
        logger.info("Bale Messenger Polling Client initialized with stable session layer.")

    def get_updates(self) -> list:
        url = f"{self.base_url}getUpdates"
        params = {"offset": self.offset, "timeout": 20}
        try:
            response = self.session.get(url, params=params, timeout=30)
            if response.status_code == 200:
                res_json = response.json()
                if res_json.get("ok") is True:
                    return res_json.get("result", [])
                else:
                    logger.error(f"Bale API Internal Error: {res_json.get('description')}")
            elif response.status_code in [500, 502, 503, 504]:
                logger.warning(f"Bale server temporarily unavailable (Status Code: {response.status_code}).")
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            logger.warning("Network fluctuation or connection timeout detected. Retrying automatically...")
        except Exception as e:
            logger.error(f"Unexpected exception in get_updates: {e}")
        return []

    def send_message(self, chat_id: int, text: str, reply_id: int = None):
        url = f"{self.base_url}sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": text
        }
        if reply_id:
            payload["reply_to_message_id"] = reply_id

        for attempt in range(3):
            try:
                response = self.session.post(url, json=payload, timeout=15)
                if response.status_code == 200 and response.json().get("ok") is True:
                    return
                logger.warning(f"Failed to send message. Retrying execution (Attempt {attempt+1}/3)...")
                time.sleep(1)
            except Exception as e:
                logger.warning(f"Exception occurred during dispatch (Attempt {attempt+1}/3): {e}")
                time.sleep(2)

    def process_update(self, update: dict):
        if "message" not in update or "text" not in update["message"]:
            return

        message = update["message"]
        chat_id = message["chat"]["id"]
        user_query = message["text"].strip()
        message_id = message["message_id"]

        if user_query == "/start":
            welcome = (
                " سلام! من یک بات مبتنی برپردازش زبان طبیعی به عنوان دستیار هوشمند پاسخگویی به سوالات شرکت .. هستم. \n\n"
                " می‌توانم به سوالات شما پاسخ دهم.\n\n"
                "لطفاً سوال خود را مطرح کنید."
            )
            self.send_message(chat_id, welcome)
            return

        if not self.engine:
            self.send_message(chat_id, "⚙️ سیستم پاسخگویی در حال حاضر در دسترس نیست.")
            return

        # Setting threshold to 0.60 to align with BGE-M3 optimal config
        results = self.engine.search(query=user_query, top_k=3, threshold=0.50, user_ip="Bale_Bot")

        if results:
            best_match = results[0]
            reply_text = best_match["short_answer"] if best_match["short_answer"] else best_match["full_answer"]
            self.send_message(chat_id, reply_text, reply_id=message_id)
        else:
            fallback = (
                "متاسفانه پاسخ دقیقی برای این سوال پیدا نکردم. \n\n"
                "سوال شما ثبت شد تا توسط کارشناسان بررسی شود."
            )
            self.send_message(chat_id, fallback, reply_id=message_id)

    def start_polling(self):
        logger.info("Stable long-polling loop activated for Bale integration layer.")
        error_sleep = 1
        while True:
            updates = self.get_updates()
            if updates:
                error_sleep = 1
                for update in updates:
                    self.offset = update["update_id"] + 1
                    self.process_update(update)
            else:
                if error_sleep < 8:
                    error_sleep *= 1.5
            time.sleep(error_sleep)


# ==========================================
# 4. REST API Engine Layer (FastAPI)
# ==========================================
app = FastAPI(
    title="Intelligent FAQ System API",
    description="Central core web service handling semantic vector searches and knowledge retrieval.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3
    threshold: Optional[float] = 0.60  # Updated to match BGE-M3 recommendations

class FAQResult(BaseModel):
    rank: int
    score: float
    question: str
    short_answer: str
    full_answer: Optional[str] = ""
    topic: Optional[str] = ""
    id: str

class SearchResponse(BaseModel):
    query: str
    results: List[FAQResult]
    total_results: int
    response_time_ms: Optional[float] = None

@app.get("/", tags=["Root"])
async def root():
    return {
        "service": "FAQ Intelligent API",
        "status": "active" if search_engine else "degraded"
    }

@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search_faq(request: SearchRequest, http_request: Request):
    if not search_engine:
        raise HTTPException(status_code=503, detail="Vector search engine is currently unavailable.")
    
    start_time = time.time()
    try:
        client_ip = http_request.client.host if http_request.client else "unknown"
        results = search_engine.search(
            query=request.query,
            top_k=request.top_k,
            threshold=request.threshold,
            user_ip=client_ip
        )
        response_time = (time.time() - start_time) * 1000
        return SearchResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            response_time_ms=response_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/health", tags=["Monitoring"])
async def health_check():
    if not search_engine:
        return {"status": "unhealthy", "timestamp": datetime.utcnow().isoformat()}
    stats = search_engine.get_stats()
    return {
        "status": "healthy",
        "total_faqs": stats.get("total_faqs", 0),
        "model_in_use": stats.get("model"),
        "timestamp": datetime.utcnow().isoformat()
    }


# ==========================================
# 5. Service Application Lifespan (Execution)
# ==========================================
if __name__ == "__main__":
    def launch_bale_thread():
        try:
            if search_engine:
                bot = BaleFAQBot(token=BOT_TOKEN, engine=search_engine)
                bot.start_polling()
            else:
                logger.error("Bale Bot thread failed to launch because search_engine is uninitialized.")
        except Exception as bot_err:
            logger.critical(f"Critical failure on dedicated Bale runtime thread: {bot_err}")

    bot_thread = threading.Thread(target=launch_bale_thread, daemon=True)
    bot_thread.start()
    logger.info("Dedicated background worker thread for messenger polling deployed.")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False 
    )