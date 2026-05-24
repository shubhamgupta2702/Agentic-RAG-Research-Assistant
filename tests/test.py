from dotenv import load_dotenv
import os

load_dotenv()

print(os.getenv("ASTRA_DB_APPLICATION_TOKEN"))
print(os.getenv("ASTRA_DB_API_ENDPOINT"))
print(os.getenv("ASTRA_DB_ASTRA_DB_KEYSPACEAPPLICATION_TOKEN"))
print(os.getenv("HUGGINGFACEHUB_API_TOKEN"))
print(os.getenv("TAVILY_API_KEY"))
print(os.getenv("ASTRA_DB_APPLICATION_TOKEN"))