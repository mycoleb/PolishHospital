"""
Poland: Hospitals vs Population Density (NUTS2)

Map:
- Choropleth: population density (people per km^2)
- Proportional circles: hospital beds per 100k people (proxy for hospital capacity)

Data sources (Eurostat):
- Boundaries: GISCO NUTS 2021 shapefile (NUTS2 level, EPSG:4326)
- Population: tgs00096 (Population on 1 January by NUTS 2 region)
- Hospital beds: hlth_rs_bdsrg2 (Available beds in hospitals by NUTS 2 region)

Outputs:
- outputs/poland_hospitals_vs_density_<year>.png
- outputs/poland_hospitals_vs_density_<year>.csv
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import requests
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from eurostat import get_data_df


GISCO_NUTS2_URL = (
    "https://gisco-services.ec.europa.eu/distribution/v2/nuts/shp/"
    "NUTS_RG_01M_2021_4326_LEVL_2.shp.zip"
)

DEFAULT_YEAR = None  # if None, auto-pick latest common year across datasets


def download_and_read_nuts2_poland(cache_dir: Path) -> gpd.GeoDataFrame:
    """
    Download NUTS2 shapefile zip (cached) and return Poland only as GeoDataFrame.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "NUTS_RG_01M_2021_4326_LEVL_2.shp.zip"

    if not zip_path.exists():
        print(f"Downloading NUTS2 boundaries -> {zip_path}")
        r = requests.get(GISCO_NUTS2_URL, timeout=60)
        r.raise_for_status()
        zip_path.write_bytes(r.content)

    # Read shapefile from zip without extracting everything permanently
    with zipfile.ZipFile(zip_path, "r") as zf:
        # find .shp
        shp_files = [n for n in zf.namelist() if n.lower().endswith(".shp")]
        if not shp_files:
            raise RuntimeError("Could not find .shp inside the GISCO zip.")
        shp_name = shp_files[0]

        # geopandas can read from a "virtual" extracted folder only if files exist on disk,
        # so we extract to a cache folder once.
        extract_dir = cache_dir / "nuts2_2021_4326_levl2"
        if not extract_dir.exists():
            print(f"Extracting shapefile -> {extract_dir}")
            zf.extractall(extract_dir)

    # Now read from extracted directory
    shp_on_disk = next((p for p in extract_dir.rglob("*.shp")), None)
    if shp_on_disk is None:
        raise RuntimeError("Extraction completed but .shp not found on disk.")

    gdf = gpd.read_file(shp_on_disk)

    # Typical GISCO fields include CNTR_CODE (country code) and NUTS_ID.
    if "CNTR_CODE" not in gdf.columns or "NUTS_ID" not in gdf.columns:
        raise RuntimeError(f"Unexpected GISCO schema. Columns: {list(gdf.columns)}")

    pl = gdf[gdf["CNTR_CODE"] == "PL"].copy()
    pl = pl[["NUTS_ID", "NAME_LATN", "geometry"]].rename(columns={"NAME_LATN": "region_name"})
    pl = pl.to_crs(4326)
    return pl


def pick_year(pop_df: pd.DataFrame, beds_df: pd.DataFrame, year: int | None) -> int:
    """
    Choose a year:
    - If year provided, use it (and error if missing).
    - Else pick the latest year present in BOTH datasets for Poland.
    """
    # Eurostat package returns "time" as int or str depending on dataset.
    pop_years = set(pd.to_numeric(pop_df["time"], errors="coerce").dropna().astype(int).unique())
    beds_years = set(pd.to_numeric(beds_df["time"], errors="coerce").dropna().astype(int).unique())
    common = sorted(pop_years.intersection(beds_years))

    if not common:
        raise RuntimeError("No common years found between population and beds datasets.")

    if year is None:
        return common[-1]

    if year not in common:
        raise RuntimeError(f"Requested year={year} not available in both datasets. Common years: {common[:5]}...{common[-5:]}")
    return year


def load_population_nuts2() -> pd.DataFrame:
    """
    Load population at NUTS2 level (tgs00096).
    Returns columns: geo, time, population
    """
    df = get_data_df("tgs00096", flags=False)
    # Expect columns: geo, time, values (sometimes 'values' or 'value')
    value_col = "values" if "values" in df.columns else ("value" if "value" in df.columns else None)
    if value_col is None:
        raise RuntimeError(f"Unexpected tgs00096 schema: {list(df.columns)}")

    df = df.rename(columns={value_col: "population"})
    df = df[df["geo"].astype(str).str.startswith("PL")].copy()
    df["population"] = pd.to_numeric(df["population"], errors="coerce")
    df = df.dropna(subset=["population"])
    return df[["geo", "time", "population"]]


def load_hospital_beds_nuts2() -> pd.DataFrame:
    """
    Load available beds in hospitals at NUTS2 (hlth_rs_bdsrg2).
    Returns columns: geo, time, beds
    """
    df = get_data_df("hlth_rs_bdsrg2", flags=False)

    value_col = "values" if "values" in df.columns else ("value" if "value" in df.columns else None)
    if value_col is None:
        raise RuntimeError(f"Unexpected hlth_rs_bdsrg2 schema: {list(df.columns)}")

    df = df.rename(columns={value_col: "beds"})

    # Filter to Poland regions
    df = df[df["geo"].astype(str).str.startswith("PL")].copy()
    df["beds"] = pd.to_numeric(df["beds"], errors="coerce")
    df = df.dropna(subset=["beds"])

    # hlth_rs_bdsrg2 can contain multiple dimensions (e.g., care type / unit).
    # Keep "Total" / "All" where possible by preferring rows with minimal extra dimensions.
    # A simple robust approach: if multiple rows exist per (geo,time), take the max (often Total).
    grp = df.groupby(["geo", "time"], as_index=False)["beds"].max()

    return grp


def build_map(year: int | None, out_dir: Path, cache_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading boundaries...")
    pl_gdf = download_and_read_nuts2_poland(cache_dir)

    print("Loading population (tgs00096)...")
    pop = load_population_nuts2()

    print("Loading hospital beds (hlth_rs_bdsrg2)...")
    beds = load_hospital_beds_nuts2()

    chosen_year = pick_year(pop, beds, year)
    print(f"Using year: {chosen_year}")

    pop_y = pop[pd.to_numeric(pop["time"], errors="coerce").astype(int) == chosen_year].copy()
    beds_y = beds[pd.to_numeric(beds["time"], errors="coerce").astype(int) == chosen_year].copy()

    # Merge tables onto shapes (NUTS_ID == geo)
    df = pl_gdf.merge(pop_y, left_on="NUTS_ID", right_on="geo", how="left").merge(
        beds_y, left_on="NUTS_ID", right_on="geo", how="left", suffixes=("", "_beds")
    )

    # Compute area (km^2) in a projected CRS
    df_m = df.to_crs(3035)  # ETRS89 / LAEA Europe (great for area)
    df["area_km2"] = (df_m.geometry.area / 1_000_000.0).astype(float)

    df["pop_density"] = df["population"] / df["area_km2"]
    df["beds_per_100k"] = (df["beds"] / df["population"]) * 100_000

    # Some regions may be missing values depending on Eurostat coverage; drop for plotting circles
    circles = df.dropna(subset=["beds_per_100k"]).copy()
    centroids = circles.to_crs(3035).centroid.to_crs(4326)
    circles["cx"] = centroids.x
    circles["cy"] = centroids.y

    # Scale circle sizes nicely
    # (matplotlib scatter uses area in points^2; keep stable with sqrt scaling)
    b = circles["beds_per_100k"].clip(lower=0)
    size = (b ** 0.5) * 25  # tweak factor for aesthetics
    circles["size"] = size

    # Plot
    fig, ax = plt.subplots(figsize=(10, 12))

    # Choropleth of population density
    df.plot(
        ax=ax,
        column="pop_density",
        legend=True,
        legend_kwds={"label": "Population density (people / km²)", "shrink": 0.6},
        linewidth=0.7,
        edgecolor="white",
        missing_kwds={"color": "lightgrey", "label": "Missing data"},
    )

    # Proportional circles for beds per 100k
    ax.scatter(
        circles["cx"],
        circles["cy"],
        s=circles["size"],
        alpha=0.55,
        linewidths=0.8,
        edgecolors="black",
    )

    ax.set_title(f"Poland (NUTS2): Population Density vs Hospital Capacity ({chosen_year})", pad=14)
    ax.set_axis_off()

    # Small note
    note = "Circles = hospital beds per 100k (proxy for hospital capacity). Fill = population density."
    ax.text(0.01, 0.02, note, transform=ax.transAxes, fontsize=9)

    out_png = out_dir / f"poland_hospitals_vs_density_{chosen_year}.png"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    out_csv = out_dir / f"poland_hospitals_vs_density_{chosen_year}.csv"
    df.drop(columns="geometry").to_csv(out_csv, index=False)

    print(f"Wrote:\n - {out_png}\n - {out_csv}")
    return out_png


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR, help="Year to plot (default: auto latest common year)")
    parser.add_argument("--out", type=str, default="outputs", help="Output directory")
    parser.add_argument("--cache", type=str, default="cache", help="Cache directory (downloads/extracts)")
    args = parser.parse_args()

    build_map(year=args.year, out_dir=Path(args.out), cache_dir=Path(args.cache))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
