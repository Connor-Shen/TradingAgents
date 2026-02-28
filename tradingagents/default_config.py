import os

DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", "./results"),
    "data_cache_dir": os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
        "dataflows/data_cache",
    ),
    # LLM settings
    "llm_provider": "openai",
    "deep_think_llm": "gpt-5.2",
    "quick_think_llm": "gpt-5-mini",
    "backend_url": "https://api.openai.com/v1",
    # Provider-specific thinking configuration
    "google_thinking_level": None,      # "high", "minimal", etc.
    # Vertex AI settings (None means read from environment variables)
    "google_use_vertexai": None,        # True/False/None
    "google_cloud_project": None,       # e.g. "my-gcp-project"
    "google_cloud_location": None,      # e.g. "us-central1"
    # Extra network robustness on top of SDK retries
    "google_network_max_retries": None,         # int, default from env or 4
    "google_network_retry_base_delay": None,    # float seconds, default 1.0
    "google_network_retry_max_delay": None,     # float seconds, default 12.0
    "google_network_retry_jitter": None,        # float seconds, default 0.5
    "openai_reasoning_effort": None,    # "medium", "high", "low"
    # Debate and discussion settings
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    # Optional fixed team execution switches (for ablation)
    "enabled_fixed_teams": ["research", "trading", "risk", "portfolio"],
    # Data vendor configuration
    # Category-level configuration (default for all tools in category)
    "data_vendors": {
        "core_stock_apis": "yfinance",       # Options: alpha_vantage, yfinance
        "technical_indicators": "yfinance",  # Options: alpha_vantage, yfinance
        "fundamental_data": "yfinance",      # Options: alpha_vantage, yfinance
        "news_data": "yfinance",             # Options: alpha_vantage, yfinance
    },
    # Tool-level configuration (takes precedence over category-level)
    "tool_vendors": {
        # Example: "get_stock_data": "alpha_vantage",  # Override category default
    },
}
