from openai import api_key
from zai import ZaiClient
import os

api_key = os.getenv("OPENAI_API_KEY")

client = ZaiClient(api_key=api_key)  # Fill in your own APIKey

response = client.web_search.web_search(
  search_engine="search-prime",
  search_query="chatgpt 最新新闻",
  count=15, # The number of results to return, ranging from 1-50, default 10
  search_domain_filter="www.sohu.com", # Only access content from specified domain names.
  search_recency_filter="noLimit", # Search for content within specified date ranges
)
print(response)