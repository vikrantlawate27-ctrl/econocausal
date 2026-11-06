import os
import sys
import time

sys.path.insert(0, os.getcwd())
print("Python version:", sys.version)

modules = [
    "sqlite3",
    "pandas",
    "flask",
    "sklearn",
    "database.database",
    "services.email_service",
    "models.demand_model",
    "pricing.pricing_engine",
    "models.dml_model",
    "app",
]

for mod in modules:
    t0 = time.time()
    try:
        __import__(mod)
        elapsed = time.time() - t0
        print(f"SUCCESS: {mod} imported in {elapsed:.3f}s")
    except Exception as e:
        print(f"ERROR: {mod} failed: {e}")
