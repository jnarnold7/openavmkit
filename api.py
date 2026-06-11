import os
import sys
import uuid
import json
import shutil
import statistics
import traceback
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="OpenAVMKit API Service")

# Enable CORS for the GeoLibre plugin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Insert ok_avmkit path into sys.path to access its core libraries
sys.path.insert(0, '/workspace/ok_avmkit/ok_avmkit')
sys.path.insert(0, '/workspace/openavmkit')

# Auto-generate sample data if it's missing on startup
try:
    sys.path.insert(0, '/workspace/ok_avmkit/sample_data')
    import generate_sample
    cama_src = "/workspace/ok_avmkit/sample_data/beckham_sample_cama.csv"
    geo_src = "/workspace/ok_avmkit/sample_data/beckham_sample_parcels.geojson"
    if not os.path.exists(cama_src) or not os.path.exists(geo_src):
        print("Generating Beckham county sample data...")
        generate_sample.main()
        print("Beckham county sample data generated.")
except Exception as e:
    print("Could not auto-generate sample data:", e)


@app.get("/")
def read_root():
    return {
        "status": "healthy",
        "service": "OpenAVMKit API Service",
        "version": "1.0.0"
    }


@app.post("/run-sample")
async def run_sample():
    # Create a unique work directory under /workspace/openavmkit to avoid path conflicts
    work_dir = os.path.join("/workspace/openavmkit", f"_work_api_{uuid.uuid4().hex}")
    os.makedirs(os.path.join(work_dir, "in"), exist_ok=True)
    os.makedirs(os.path.join(work_dir, "out"), exist_ok=True)

    try:
        cama_src = "/workspace/ok_avmkit/sample_data/beckham_sample_cama.csv"
        geo_src = "/workspace/ok_avmkit/sample_data/beckham_sample_parcels.geojson"

        if not os.path.exists(cama_src) or not os.path.exists(geo_src):
            raise HTTPException(status_code=404, detail="Sample data not found. Please generate sample data first.")

        # Import modules dynamically to ensure paths are clean
        from core.ok_column_mapper import auto_detect_mapping
        from core.ok_data_ingest import read_csv, normalize_dates, write_to_locality
        from core.ok_compliance import generate_ok_settings, generate_ok_metadata
        from core.ok_geo_stage import stage_parcel_geometry
        from core.ok_openavmkit_runner import run_pipeline
        from core.analysis.ok_result_adapter import build_ratio_study_results, build_model_summary

        # 1. Read CAMA csv
        cols, rows = read_csv(cama_src)
        mapping = auto_detect_mapping(cols)

        # Normalize dates
        ds = mapping.get("sale_date")
        if ds:
            rows = normalize_dates(rows, [ds])
        write_to_locality(rows, work_dir, "parcel_data.csv")

        # 2. Stage geometry (GeoJSON -> Parquet)
        keys = [str(r.get(mapping["parcel_id"], "")).strip() for r in rows]
        stage_parcel_geometry(
            geo_src,
            keys,
            os.path.join(work_dir, "in", "geo", "parcels.parquet")
        )

        # 3. Calculate center coordinates
        center = None
        lat_s, lon_s = mapping.get("latitude"), mapping.get("longitude")
        if lat_s and lon_s:
            lats = [float(r[lat_s]) for r in rows if r.get(lat_s)]
            lons = [float(r[lon_s]) for r in rows if r.get(lon_s)]
            if lats and lons:
                center = (statistics.mean(lats), statistics.mean(lons))

        # 4. Generate settings & metadata
        settings = generate_ok_settings(
            locality_name="us-ok-beckham",
            county_name="Beckham",
            assessment_year=2026,
            fips_code="40009",
            mapping=mapping,
            assessment_ratio=0.12,
            analyst_name="Demo Analyst",
            models=["mra", "lightgbm", "xgboost"],
            center=center,
            n_trials=5,
            output_path=os.path.join(work_dir, "in", "settings.json")
        )

        meta = generate_ok_metadata(
            county_name="Beckham",
            assessment_year=2026,
            fips_code="40009",
            assessment_ratio=0.12,
            analyst_name="Demo Analyst",
            output_path=os.path.join(work_dir, "in", "ok_metadata.json")
        )

        # 5. Run OpenAVMKit pipeline
        oavmk = run_pipeline(
            work_dir,
            settings,
            run_main=True,
            run_vacant=False,
            run_ensemble=True,
            run_scrutiny=True
        )

        # 6. Build results and summary adapters
        results = build_ratio_study_results(oavmk, meta)
        summary = build_model_summary(oavmk, meta)

        # 7. Hydrate original GeoJSON with predicted valuations & details
        with open(geo_src, "r") as f:
            geojson = json.load(f)

        # Create CAMA details lookup
        cama_lookup = {}
        for r in rows:
            acc = str(r.get(mapping["parcel_id"], "")).strip()
            cama_lookup[acc] = {
                "sale_price": float(r[mapping["sale_price"]]) if r.get(mapping["sale_price"]) else None,
                "assr_market_value": float(r[mapping["assr_market_value"]]) if r.get(mapping["assr_market_value"]) else None,
                "property_class": r.get("PropertyClass"),
                "nbhd": r.get("Nbhd"),
                "nbhd_desc": r.get("NbhdDesc")
            }

        # Create predictions lookup
        pred_lookup = {}
        if oavmk.pred_universe is not None:
            for _, row_p in oavmk.pred_universe.iterrows():
                pred_lookup[str(row_p["key"])] = float(row_p["prediction"])

        # Update geojson feature properties
        for feat in geojson.get("features", []):
            props = feat.get("properties", {})
            acc = str(props.get("Account", "")).strip()
            c_info = cama_lookup.get(acc, {})
            props.update(c_info)
            props["prediction"] = pred_lookup.get(acc, None)
            
            # Calculate predicted/actual ratios
            if props.get("sale_price") and props.get("prediction"):
                try:
                    sp = float(props["sale_price"])
                    pred = float(props["prediction"])
                    if sp > 0:
                        props["ratio"] = round(pred / sp, 4)
                except:
                    pass
            feat["properties"] = props

        return {
            "success": True,
            "results": results,
            "summary": summary,
            "geojson": geojson
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary work directory
        shutil.rmtree(work_dir, ignore_errors=True)


@app.post("/run")
async def run_pipeline_endpoint(
    cama_file: UploadFile = File(...),
    geo_file: UploadFile = File(...),
    county_name: str = Form("CustomCounty"),
    analyst_name: str = Form("Jane Doe"),
    assessment_year: int = Form(2026),
    assessment_ratio: float = Form(0.11),
    fips_code: str = Form("40009"),
    models: str = Form("mra,lightgbm,xgboost"),
):
    work_dir = os.path.join("/workspace/openavmkit", f"_work_api_{uuid.uuid4().hex}")
    os.makedirs(os.path.join(work_dir, "in"), exist_ok=True)
    os.makedirs(os.path.join(work_dir, "out"), exist_ok=True)

    try:
        # Save uploaded files temporarily
        uploaded_cama_path = os.path.join(work_dir, cama_file.filename)
        uploaded_geo_path = os.path.join(work_dir, geo_file.filename)

        with open(uploaded_cama_path, "wb") as buffer:
            shutil.copyfileobj(cama_file.file, buffer)

        with open(uploaded_geo_path, "wb") as buffer:
            shutil.copyfileobj(geo_file.file, buffer)

        from core.ok_column_mapper import auto_detect_mapping
        from core.ok_data_ingest import read_csv, normalize_dates, write_to_locality
        from core.ok_compliance import generate_ok_settings, generate_ok_metadata
        from core.ok_geo_stage import stage_parcel_geometry
        from core.ok_openavmkit_runner import run_pipeline
        from core.analysis.ok_result_adapter import build_ratio_study_results, build_model_summary

        # 1. Read CAMA csv
        cols, rows = read_csv(uploaded_cama_path)
        mapping = auto_detect_mapping(cols)

        # Normalize dates
        ds = mapping.get("sale_date")
        if ds:
            rows = normalize_dates(rows, [ds])
        write_to_locality(rows, work_dir, "parcel_data.csv")

        # 2. Stage geometry
        keys = [str(r.get(mapping["parcel_id"], "")).strip() for r in rows]
        stage_parcel_geometry(
            uploaded_geo_path,
            keys,
            os.path.join(work_dir, "in", "geo", "parcels.parquet")
        )

        # 3. Calculate center coordinates
        center = None
        lat_s, lon_s = mapping.get("latitude"), mapping.get("longitude")
        if lat_s and lon_s:
            lats = [float(r[lat_s]) for r in rows if r.get(lat_s)]
            lons = [float(r[lon_s]) for r in rows if r.get(lon_s)]
            if lats and lons:
                center = (statistics.mean(lats), statistics.mean(lons))

        # 4. Generate settings & metadata
        model_list = [m.strip() for m in models.split(",") if m.strip()]
        settings = generate_ok_settings(
            locality_name=f"us-ok-{county_name.lower()}",
            county_name=county_name,
            assessment_year=assessment_year,
            fips_code=fips_code,
            mapping=mapping,
            assessment_ratio=assessment_ratio,
            analyst_name=analyst_name,
            models=model_list,
            center=center,
            n_trials=5,
            output_path=os.path.join(work_dir, "in", "settings.json")
        )

        meta = generate_ok_metadata(
            county_name=county_name,
            assessment_year=assessment_year,
            fips_code=fips_code,
            assessment_ratio=assessment_ratio,
            analyst_name=analyst_name,
            output_path=os.path.join(work_dir, "in", "ok_metadata.json")
        )

        # 5. Run OpenAVMKit pipeline
        oavmk = run_pipeline(
            work_dir,
            settings,
            run_main=True,
            run_vacant=False,
            run_ensemble=True,
            run_scrutiny=True
        )

        # 6. Build results
        results = build_ratio_study_results(oavmk, meta)
        summary = build_model_summary(oavmk, meta)

        # 7. Hydrate uploaded GeoJSON
        with open(uploaded_geo_path, "r") as f:
            geojson = json.load(f)

        # Create CAMA details lookup
        cama_lookup = {}
        for r in rows:
            acc = str(r.get(mapping["parcel_id"], "")).strip()
            cama_lookup[acc] = {
                "sale_price": float(r[mapping["sale_price"]]) if r.get(mapping["sale_price"]) else None,
                "assr_market_value": float(r[mapping["assr_market_value"]]) if r.get(mapping["assr_market_value"]) else None,
                "property_class": r.get("PropertyClass"),
                "nbhd": r.get("Nbhd"),
                "nbhd_desc": r.get("NbhdDesc")
            }

        # Create predictions lookup
        pred_lookup = {}
        if oavmk.pred_universe is not None:
            for _, row_p in oavmk.pred_universe.iterrows():
                pred_lookup[str(row_p["key"])] = float(row_p["prediction"])

        # Update geojson
        for feat in geojson.get("features", []):
            props = feat.get("properties", {})
            
            # Look for common ID columns if Account isn't present
            acc = None
            for key in ["Account", "account", "parcel_id", "PARCEL_ID", "key"]:
                if key in props:
                    acc = str(props[key]).strip()
                    break
            if not acc:
                # Fallback to the first property value
                acc = str(next(iter(props.values()))).strip()
                
            c_info = cama_lookup.get(acc, {})
            props.update(c_info)
            props["prediction"] = pred_lookup.get(acc, None)
            
            if props.get("sale_price") and props.get("prediction"):
                try:
                    sp = float(props["sale_price"])
                    pred = float(props["prediction"])
                    if sp > 0:
                        props["ratio"] = round(pred / sp, 4)
                except:
                    pass
            feat["properties"] = props

        return {
            "success": True,
            "results": results,
            "summary": summary,
            "geojson": geojson
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
