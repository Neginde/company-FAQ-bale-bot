# logger.py
import json
import time
from datetime import datetime
from pathlib import Path
import logging
from typing import Dict, List, Optional, Any
import threading

# تنظیمات لاگ
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# فایل‌های مختلف برای اهداف مختلف
LOG_FILES = {
    "queries": LOG_DIR / "faq_queries.jsonl",      # سوالات کاربران
    "errors": LOG_DIR / "faq_errors.jsonl",        # خطاها
    "performance": LOG_DIR / "faq_performance.jsonl",  # عملکرد
    "analytics": LOG_DIR / "faq_analytics.jsonl"   # آمار
}

# ایجاد فایل‌ها اگر وجود ندارند
for log_file in LOG_FILES.values():
    if not log_file.exists():
        log_file.touch()

# قفل برای thread-safe نوشتن
write_lock = threading.Lock()

class FAQLogger:
    """لاگر پیشرفته برای سیستم FAQ"""
    
    @staticmethod
    def log_query(
        query: str,
        normalized_query: str,
        results: List[Dict],
        user_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
        session_id: Optional[str] = None,
        timings: Optional[Dict] = None
    ):
        """
        لاگ کردن سوال کاربر و نتایج
        
        Args:
            query: سوال اصلی کاربر
            normalized_query: سوال نرمالایز شده
            results: لیست نتایج
            user_ip: IP کاربر
            user_agent: User-Agent مرورگر
            session_id: شناسه نشست
            timings: زمان‌بندی‌های مختلف
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "query",
            "query": {
                "original": query,
                "normalized": normalized_query,
                "length": len(query),
                "word_count": len(query.split())
            },
            "results": {
                "count": len(results),
                "top_score": results[0]["score"] if results else None,
                "questions": [r["question"] for r in results[:3]]  # فقط ۳ تا اول
            },
            "user_info": {
                "ip": user_ip,
                "user_agent": user_agent,
                "session_id": session_id
            },
            "timings": timings or {},
            "success": len(results) > 0
        }
        
        # حذف فیلدهای None
        log_entry["user_info"] = {k: v for k, v in log_entry["user_info"].items() if v is not None}
        
        with write_lock:
            with open(LOG_FILES["queries"], "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        
        # همچنین لاگ کنسول
        logging.info(f"📝 Query logged: '{query[:50]}...' → {len(results)} results")
    
    @staticmethod
    def log_error(
        error_type: str,
        error_message: str,
        query: Optional[str] = None,
        user_ip: Optional[str] = None,
        traceback: Optional[str] = None
    ):
        """لاگ کردن خطاها"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "error",
            "error": {
                "type": error_type,
                "message": error_message,
                "traceback": traceback
            },
            "query": query,
            "user_ip": user_ip
        }
        
        with write_lock:
            with open(LOG_FILES["errors"], "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        
        logging.error(f"❌ Error logged: {error_type} - {error_message}")
    
    @staticmethod
    def log_performance(
        operation: str,
        duration_ms: float,
        details: Optional[Dict] = None
    ):
        """لاگ کردن عملکرد سیستم"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "performance",
            "operation": operation,
            "duration_ms": duration_ms,
            "details": details or {}
        }
        
        with write_lock:
            with open(LOG_FILES["performance"], "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        
        logging.debug(f"⚡ Performance: {operation} took {duration_ms:.2f}ms")
    
    @staticmethod
    def log_feedback(
        query: str,
        result_id: str,
        helpful: bool,
        user_ip: Optional[str] = None,
        comment: Optional[str] = None
    ):
        """لاگ کردن بازخورد کاربران"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "feedback",
            "query": query,
            "result_id": result_id,
            "helpful": helpful,
            "comment": comment,
            "user_ip": user_ip
        }
        
        with write_lock:
            with open(LOG_FILES["analytics"], "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        
        logging.info(f"📊 Feedback: {'👍' if helpful else '👎'} for result {result_id}")
    
    @staticmethod
    def get_stats(days: int = 7) -> Dict:
        """گرفتن آمار از لاگ‌ها"""
        stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "avg_response_time": 0,
            "top_queries": [],
            "error_types": {}
        }
        
        try:
            # خواندن لاگ‌های اخیر
            cutoff_date = datetime.utcnow().timestamp() - (days * 86400)
            
            with open(LOG_FILES["queries"], "r", encoding="utf-8") as f:
                queries = []
                response_times = []
                
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        entry_time = datetime.fromisoformat(entry["timestamp"]).timestamp()
                        
                        if entry_time > cutoff_date:
                            stats["total_queries"] += 1
                            if entry["success"]:
                                stats["successful_queries"] += 1
                            else:
                                stats["failed_queries"] += 1
                            
                            # جمع‌آوری زمان‌ها
                            if "timings" in entry and "total" in entry["timings"]:
                                response_times.append(entry["timings"]["total"])
                            
                            # جمع‌آوری سوالات پرتکرار
                            queries.append(entry["query"]["original"])
                    except:
                        continue
                
                # محاسبه میانگین
                if response_times:
                    stats["avg_response_time"] = sum(response_times) / len(response_times)
                
                # سوالات پرتکرار
                from collections import Counter
                query_counter = Counter(queries)
                stats["top_queries"] = query_counter.most_common(10)
        
        except FileNotFoundError:
            pass
        
        return stats
    
    @staticmethod
    def rotate_logs(max_size_mb: int = 100):
        """چرخش فایل‌های لاگ وقتی بزرگ می‌شوند"""
        for log_name, log_file in LOG_FILES.items():
            if log_file.exists():
                size_mb = log_file.stat().st_size / (1024 * 1024)
                
                if size_mb > max_size_mb:
                    # ایجاد backup
                    backup_file = log_file.with_suffix(f".{datetime.now().strftime('%Y%m%d_%H%M%S')}.backup")
                    
                    with write_lock:
                        # خواندن ۵۰۰۰ خط آخر
                        with open(log_file, "r", encoding="utf-8") as f:
                            lines = f.readlines()
                        
                        if len(lines) > 10000:
                            # ذخیره قدیمی
                            with open(backup_file, "w", encoding="utf-8") as f:
                                f.writelines(lines[:-5000])
                            
                            # نوشتن جدید
                            with open(log_file, "w", encoding="utf-8") as f:
                                f.writelines(lines[-5000:])
                            
                            logging.info(f"🔄 Rotated log file: {log_name} (kept last 5000 lines)")