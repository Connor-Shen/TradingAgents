from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Create a custom config
config = DEFAULT_CONFIG.copy()
config["llm_provider"] = "google"
config["deep_think_llm"] = "gemini-2.5-flash"
config["quick_think_llm"] = "gemini-2.5-flash"
config["google_use_vertexai"] = (
    os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "true").lower() == "true"
)
config["google_cloud_project"] = os.getenv("GOOGLE_CLOUD_PROJECT")
config["google_cloud_location"] = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
config["google_network_max_retries"] = int(
    os.getenv("TRADINGAGENTS_GOOGLE_NETWORK_MAX_RETRIES", "4")
)
config["google_network_retry_base_delay"] = float(
    os.getenv("TRADINGAGENTS_GOOGLE_NETWORK_RETRY_BASE_DELAY", "1.0")
)
config["google_network_retry_max_delay"] = float(
    os.getenv("TRADINGAGENTS_GOOGLE_NETWORK_RETRY_MAX_DELAY", "12.0")
)
config["google_network_retry_jitter"] = float(
    os.getenv("TRADINGAGENTS_GOOGLE_NETWORK_RETRY_JITTER", "0.5")
)
config["max_debate_rounds"] = 0  # Increase debate rounds

if config["google_use_vertexai"] and not config["google_cloud_project"]:
    raise ValueError(
        "GOOGLE_CLOUD_PROJECT is required when GOOGLE_GENAI_USE_VERTEXAI=true."
    )

# Configure data vendors (default uses yfinance, no extra API keys needed)
config["data_vendors"] = {
    "core_stock_apis": "yfinance",           # Options: alpha_vantage, yfinance
    "technical_indicators": "yfinance",      # Options: alpha_vantage, yfinance
    "fundamental_data": "yfinance",          # Options: alpha_vantage, yfinance
    "news_data": "yfinance",                 # Options: alpha_vantage, yfinance
}

# Initialize with custom config
ta = TradingAgentsGraph(debug=True, config=config)

# forward propagate
_, decision = ta.propagate("NVDA", "2024-05-10")
print(decision)

# Memorize mistakes and reflect
# ta.reflect_and_remember(1000) # parameter is the position returns
