import sys
print("Python version:", sys.version)

packages = ["fastapi", "uvicorn", "flask", "openavmkit", "geopandas", "shapely"]
for pkg in packages:
    try:
        __import__(pkg)
        print(f"  {pkg}: AVAILABLE")
    except ImportError as e:
        print(f"  {pkg}: NOT AVAILABLE ({e})")
