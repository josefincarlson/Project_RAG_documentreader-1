import time
import httpx
import certifi
import ssl

start = time.perf_counter()
print(f"import httpx + certifi + ssl klart: {time.perf_counter() - start:.2f} sek")

start = time.perf_counter()
ctx = ssl.create_default_context(cafile=certifi.where())
print(f"ssl.create_default_context: {time.perf_counter() - start:.2f} sek")