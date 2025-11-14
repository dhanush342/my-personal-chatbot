import requests
import os
from dotenv import load_dotenv
from typing import Optional, List, Tuple, Any
import re
from datetime import datetime
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
import glob

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TOMORROW_API_KEY = os.getenv("TOMORROW_API_KEY")
DEFAULT_LOCATION = os.getenv("DEFAULT_LOCATION", "42.3478,-71.0466")
GEOCODE_COUNTRY_CODE = os.getenv("GEOCODE_COUNTRY_CODE")  # e.g., IN, US, GB


def _build_http_session() -> requests.Session:
    retry = Retry(
        total=5,
        connect=3,
        read=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "POST"]),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s = requests.Session()
    s.headers.update({
        "Connection": "keep-alive",
        "User-Agent": "chatbot-weather/1.0"
    })
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


SESSION = _build_http_session()

# --- Simple TF-IDF Retrieval (RAG) ---
try:
    import importlib
    _sklearn_text = importlib.import_module("sklearn.feature_extraction.text")
    _sklearn_metrics = importlib.import_module("sklearn.metrics.pairwise")
    TfidfVectorizer = getattr(_sklearn_text, "TfidfVectorizer")
    cosine_similarity = getattr(_sklearn_metrics, "cosine_similarity")
    _SKLEARN_OK = True
except Exception:
    TfidfVectorizer = None
    cosine_similarity = None
    _SKLEARN_OK = False


class SimpleRAG:
    def __init__(self):
        self.vectorizer: Optional[Any] = None
        self.doc_texts: List[str] = []
        self.doc_labels: List[str] = []
        self.matrix = None

    def load_sources(self, kb_dir: str = "kb", memory_file: str = "memory.json"):
        texts: List[str] = []
        labels: List[str] = []

        # Load KB files (*.txt, *.md)
        for pattern in ("*.txt", "*.md"):
            for path in glob.glob(os.path.join(kb_dir, pattern)):
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    if content.strip():
                        texts.append(content)
                        labels.append(f"file:{os.path.basename(path)}")
                except Exception:
                    pass

        # Load memory entries
        try:
            if os.path.exists(memory_file):
                with open(memory_file, "r", encoding="utf-8") as f:
                    mem = json.load(f)
                if isinstance(mem, list):
                    for i, item in enumerate(mem):
                        if isinstance(item, str) and item.strip():
                            texts.append(item)
                            labels.append(f"memory:{i}")
        except Exception:
            pass

        self.doc_texts = texts
        self.doc_labels = labels

        if _SKLEARN_OK and texts:
            self.vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
            self.matrix = self.vectorizer.fit_transform(texts)
        else:
            self.vectorizer = None
            self.matrix = None

    def retrieve(self, query: str, top_k: int = 3, min_sim: float = 0.12) -> List[Tuple[str, float]]:
        if not _SKLEARN_OK or not self.vectorizer or self.matrix is None or not self.doc_texts:
            return []
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.matrix)[0]
        ranked = sorted(list(enumerate(sims)), key=lambda x: x[1], reverse=True)[:top_k]
        results: List[Tuple[str, float]] = []
        for idx, score in ranked:
            if score >= min_sim:
                results.append((self.doc_texts[idx], float(score)))
        return results


RAG = SimpleRAG()
RAG.load_sources()


def _is_coordinates(text: str) -> bool:
    try:
        lat_str, lon_str = text.split(",", 1)
        float(lat_str.strip())
        float(lon_str.strip())
        return True
    except Exception:
        return False


def _geocode_location(name: str) -> Optional[str]:
    """Geocode a place name to "lat,lon" using OpenStreetMap Nominatim."""
    try:
        params = {"q": name, "format": "json", "limit": 1}
        # Prefer a specific country when provided (ISO 3166-1 alpha-2)
        if GEOCODE_COUNTRY_CODE:
            params["countrycodes"] = GEOCODE_COUNTRY_CODE.lower()

        resp = SESSION.get(
            "https://nominatim.openstreetmap.org/search",
            params=params,
            headers={"User-Agent": "chatbot-weather/1.0"},
            timeout=10,
        )
        if not resp.ok:
            return None
        data = resp.json()
        if not data:
            return None
        lat = data[0].get("lat")
        lon = data[0].get("lon")
        if not lat or not lon:
            return None
        return f"{lat},{lon}"
    except Exception:
        return None

def get_weather(location: Optional[str] = None):
    # Use default location if none provided
    original_label = (location or "").strip() or DEFAULT_LOCATION

    # Turn names into coordinates when needed
    if _is_coordinates(original_label):
        query_location = original_label
        label = original_label
    else:
        geocoded = _geocode_location(original_label)
        query_location = geocoded or original_label
        label = original_label

    url = "https://api.tomorrow.io/v4/weather/realtime"
    params = {
        "location": query_location,
        "apikey": TOMORROW_API_KEY,
        "units": "metric",
    }

    try:
        response = SESSION.get(url, params=params, timeout=10)
        if not response.ok:
            return f"Weather API error: {response.status_code} - {response.text}"

        data = response.json()
        data_block = data.get("data", {})
        values = data_block.get("values", {})
        temp = values.get("temperature")
        feels = values.get("temperatureApparent")
        if temp is None and feels is None:
            return "Weather API error: Temperature not available in response."

        # Observation time and resolved coords for transparency
        obs_time = data_block.get("time") or data.get("asOf")
        loc_obj = data.get("location", {})
        loc_lat = loc_obj.get("lat") if isinstance(loc_obj, dict) else None
        loc_lon = loc_obj.get("lon") if isinstance(loc_obj, dict) else None
        if (loc_lat is None or loc_lon is None) and _is_coordinates(query_location):
            try:
                lat_str, lon_str = query_location.split(",", 1)
                loc_lat, loc_lon = float(lat_str.strip()), float(lon_str.strip())
            except Exception:
                pass

        suffix = ""
        if obs_time:
            suffix += f" as of {obs_time}"
        if loc_lat is not None and loc_lon is not None:
            suffix += f" (at {loc_lat:.4f},{loc_lon:.4f})"

        if temp is not None and feels is not None and abs(feels - temp) >= 0.5:
            return f"The current temperature in {label} is {temp}°C (feels like {feels}°C){suffix}."
        elif temp is not None:
            return f"The current temperature in {label} is {temp}°C{suffix}."
        else:
            return f"The current temperature in {label} feels like {feels}°C{suffix}."
    except Exception as e:
        return f"Weather API failed: {str(e)}"


def chat_with_groq(message):
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    # Build messages with optional retrieved context
    messages = [
        {"role": "system", "content": "You are a personal chatbot. Answer using provided context when it is relevant. If context is not relevant, ignore it."},
    ]

    retrieved = RAG.retrieve(message)
    if retrieved:
        snippet = "\n\n--- Context ---\n" + "\n\n".join(
            [ctx[:1200] for ctx, _ in retrieved]
        )
        messages.append({"role": "system", "content": f"Use this context if relevant: {snippet}"})

    messages.append({"role": "user", "content": message})

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": messages,
    }

    try:
        response = SESSION.post(url, json=payload, headers=headers, timeout=45)
        result = response.json()
        if response.status_code == 200:
            return result["choices"][0]["message"]["content"]
        # Handle rate limiting with optional retry-after
        if response.status_code == 429:
            wait_s = None
            ra = response.headers.get("retry-after")
            if ra:
                try:
                    wait_s = float(ra)
                except Exception:
                    pass
            if wait_s is None:
                m = re.search(r"try again in\s+([0-9]+(?:\.[0-9]+)?)s", result.get("error", {}).get("message", ""), re.I)
                if m:
                    try:
                        wait_s = float(m.group(1))
                    except Exception:
                        pass
            if wait_s is not None and wait_s <= 60:
                time.sleep(wait_s)
                response = SESSION.post(url, json=payload, headers=headers, timeout=45)
                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    try:
                        result = response.json()
                    except Exception:
                        result = {"error": {"message": response.text}}
                    return f"Groq API error: {response.status_code} - {result.get('error', {}).get('message', 'Unknown error')}"

        return f"Groq API error: {response.status_code} - {result.get('error', {}).get('message', 'Unknown error')}"
    except requests.exceptions.RequestException as e:
        return f"Groq API error: {str(e)}"


def main():
    print(
        "Your Personal Chatbot is running.\n"
        "- Type 'weather' or 'weather <city>'\n"
        "- Ask e.g. 'temperature in Chennai' or 'temp in Delhi'\n"
        "- Basic: 'time', 'date', 'day'\n"
        "- Set DEFAULT_LOCATION in .env to change the default city\n"
        "- Optional: set GEOCODE_COUNTRY_CODE (e.g., IN) to prefer a country\n"
        "- Type 'exit' to quit.\n"
    )

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("Bot: Goodbye!")
            break

        lower = user_input.lower().strip()

        # Basic local commands: time, date, day
        def basic_command_response(text: str) -> Optional[str]:
            t = text.lower()
            # Time
            time_patterns = [
                r"^time$",
                r"\bwhat time is it\b",
                r"\bwhat(?:'s| is) the (current )?time\b",
                r"\btell me the time\b",
                r"\bcurrent time\b",
            ]
            for p in time_patterns:
                if re.search(p, t):
                    return f"The current local time is {datetime.now().strftime('%H:%M:%S')}."

            # Date
            date_patterns = [
                r"^date$",
                r"\bwhat(?:'s| is) the date\b",
                r"\btoday'?s date\b",
            ]
            for p in date_patterns:
                if re.search(p, t):
                    return f"Today's date is {datetime.now().strftime('%Y-%m-%d')}."

            # Day of week
            day_patterns = [
                r"^day$",
                r"\bwhat day is it\b",
                r"\bday of the week\b",
            ]
            for p in day_patterns:
                if re.search(p, t):
                    return f"Today is {datetime.now().strftime('%A')}."

            return None

        basic = basic_command_response(user_input)
        if basic is not None:
            print("Bot:", basic)
            continue

        # Explicit weather command: "weather" or "weather <location>"
        if lower.startswith("weather"):
            parts = user_input.split(maxsplit=1)
            if len(parts) == 2 and parts[1].strip():
                location = parts[1].strip()
            else:
                location = DEFAULT_LOCATION
            print("Bot:", get_weather(location))
            continue

        # Natural language detection for weather queries
        patterns = [
            r"\b(?:temp|temperature)\s+(?:in|at|for)\s+(.+)$",
            r"\bweather\s+(?:in|at|for)\s+(.+)$",
            r"^(?:what(?:'s| is) the )?(?:current )?(?:temp|temperature)\s+(?:in|at|for)\s+(.+)$",
            r"^(.+?)\s+weather$",  # e.g., "Chennai weather"
            r"^(?:weather|temp(?:erature)?)\s+(.+)$",  # e.g., "weather delhi", "temp delhi"
            r"^(.+?)\s+(?:temp|temperature)$",  # e.g., "Chennai temp"
            # Common misspellings
            r"\b(?:wether|wheather)\s+(?:in|at|for)\s+(.+)$",
            r"^(?:wether|wheather)\s+(.+)$",
            r"\btemprature\s+(?:in|at|for)\s+(.+)$",
        ]

        matched = False
        for pat in patterns:
            m = re.search(pat, user_input.strip(), flags=re.IGNORECASE)
            if m:
                loc = m.group(1).strip().strip(".?!")
                if loc:
                    print("Bot:", get_weather(loc))
                    matched = True
                    break

        if matched:
            continue

        # Learn/forget commands for simple memory management
        if lower.startswith("learn ") or lower.startswith("learn:"):
            taught = user_input.split(":", 1)
            taught_text = (taught[1] if len(taught) > 1 else user_input[len("learn "):]).strip()
            if taught_text:
                try:
                    mem = []
                    if os.path.exists("memory.json"):
                        with open("memory.json", "r", encoding="utf-8") as f:
                            mem_loaded = json.load(f)
                            if isinstance(mem_loaded, list):
                                mem = mem_loaded
                    mem.append(taught_text)
                    with open("memory.json", "w", encoding="utf-8") as f:
                        json.dump(mem, f, ensure_ascii=False, indent=2)
                    RAG.load_sources()
                    print("Bot:", "Learned successfully.")
                except Exception as e:
                    print("Bot:", f"Couldn't learn: {e}")
            else:
                print("Bot:", "Please provide text to learn, e.g., learn: ACME product returns in 30 days.")
            continue

        if lower in ("forget memory", "clear memory"):
            try:
                if os.path.exists("memory.json"):
                    os.remove("memory.json")
                RAG.load_sources()
                print("Bot:", "Memory cleared.")
            except Exception as e:
                print("Bot:", f"Couldn't clear memory: {e}")
            continue

        # Fallback to chat model
        print("Bot:", chat_with_groq(user_input))


if __name__ == "__main__":
    main()
