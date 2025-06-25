import pandas as pd
import geopandas as gpd
import os
from rapidfuzz import process, fuzz
from shapely.geometry import Point
import itertools
from datetime import datetime

def Create_FC_Guide(csv_path, city_shapefile_path, frontline_shapefile_path, distance_threshold=15000):
    city_csv_col = 'FC'
    city_shp_col = 'ADM4_EN'
    fuzzy_min_score = 40

    # Read input data
    df = pd.read_csv(csv_path)
    gdf_cities = gpd.read_file(city_shapefile_path)
    gdf_poly = gpd.read_file(frontline_shapefile_path)
    df[city_csv_col] = df[city_csv_col].str.lower().str.strip()
    gdf_cities[city_shp_col] = gdf_cities[city_shp_col].str.lower().str.strip()

    # Generate all possible i/y swaps for fuzzy matching
    def generate_alternatives(city):
        city = city.lower()
        indices = [i for i, c in enumerate(city) if c in {'i', 'y'}]
        alternatives = set()
        for n in range(1, len(indices)+1):
            for idxs in itertools.combinations(indices, n):
                chars = list(city)
                for idx in idxs:
                    chars[idx] = 'y' if chars[idx] == 'i' else 'i'
                alternatives.add(''.join(chars))
        return list(alternatives)

    # Fuzzy match with alternatives
    def fuzzy_match_with_alternatives(city, choices, scorer=fuzz.ratio, min_score=0):
        match, score, idx = process.extractOne(city, choices, scorer=scorer)
        if score >= min_score:
            return match
        for alt in generate_alternatives(city):
            match, score, idx = process.extractOne(alt, choices, scorer=scorer)
            if score >= min_score:
                return match
        return None

    # Project to metric CRS for distance calculation
    metric_crs = gdf_cities.estimate_utm_crs()
    gdf_cities_proj = gdf_cities.to_crs(metric_crs)
    gdf_poly_proj = gdf_poly.to_crs(metric_crs)

    # Merge on city name (inner join)
    merged = pd.merge(df, gdf_cities_proj, left_on=city_csv_col, right_on=city_shp_col, how='inner')
    unmatched = df[~df[city_csv_col].isin(merged[city_csv_col])]

    # Fuzzy match for unmatched cities
    fuzzy_rows = []
    for idx, row in unmatched.iterrows():
        city = row[city_csv_col]
        matches = process.extract(city, gdf_cities_proj[city_shp_col].tolist(), scorer=fuzz.ratio, limit=3)
        for match_name, score, _ in matches:
            if score >= fuzzy_min_score:
                candidates = gdf_cities_proj[gdf_cities_proj[city_shp_col] == match_name]
                for _, cand_row in candidates.iterrows():
                    combined = pd.concat([row, cand_row], axis=0)
                    fuzzy_rows.append(combined)
    if fuzzy_rows:
        fuzzy_df = pd.DataFrame(fuzzy_rows)
        merged = pd.concat([merged, fuzzy_df], ignore_index=True)

    # Handle any remaining NaN geometries with alternative fuzzy matching
    nan_mask = merged['geometry'].isna()
    if nan_mask.any():
        shp_names = gdf_cities_proj[city_shp_col].tolist()
        for idx in merged[nan_mask].index:
            city = merged.at[idx, city_csv_col]
            match = fuzzy_match_with_alternatives(city, shp_names)
            if match:
                candidates = gdf_cities_proj[gdf_cities_proj[city_shp_col] == match]
                candidates['distance_to_border'] = candidates.geometry.apply(lambda geom: gdf_poly_proj.geometry.unary_union.distance(geom))
                row = candidates.loc[candidates['distance_to_border'].idxmin()]
                merged.at[idx, 'geometry'] = row.geometry
                merged.at[idx, city_shp_col] = match

    # Calculate distance to frontline for each city
    line = gdf_poly_proj.geometry.unary_union
    merged['distance_to_border'] = merged.geometry.apply(lambda geom: line.distance(geom) if geom is not None else float('inf'))

    results = []
    unplaced = []

    # For each city, keep only the closest match within threshold
    for city, group in merged.groupby(city_csv_col):
        group_sorted = group.sort_values('distance_to_border')
        found = False
        matches_within_thresh = group_sorted[group_sorted['distance_to_border'] <= distance_threshold]
        if not matches_within_thresh.empty:
            results.append(matches_within_thresh.iloc[0])
            found = True
        # Try fuzzy match with alternatives if no match within threshold
        if not found and len(group_sorted) > 1:
            candidate_names = group_sorted[city_shp_col].tolist()
            fuzzy_name = fuzzy_match_with_alternatives(city, candidate_names, min_score=fuzzy_min_score)
            if fuzzy_name:
                fuzzy_row = group_sorted[(group_sorted[city_shp_col] == fuzzy_name) & (group_sorted['distance_to_border'] <= distance_threshold)]
                if not fuzzy_row.empty:
                    results.append(fuzzy_row.iloc[0])
                    found = True
        if not found:
            unplaced.append(city)

    # Prepare output GeoDataFrame and CSV
    closest_per_city = gpd.GeoDataFrame(results, geometry='geometry', crs=metric_crs).to_crs(gdf_cities.crs)
    closest_per_city['latitude'] = closest_per_city.geometry.apply(lambda geom: geom.y if geom is not None else None)
    closest_per_city['longitude'] = closest_per_city.geometry.apply(lambda geom: geom.x if geom is not None else None)
    out_df = closest_per_city[[city_csv_col, 'latitude', 'longitude']]
    gdf_out = gpd.GeoDataFrame(
        out_df,
        geometry=[Point(xy) if pd.notnull(xy[0]) and pd.notnull(xy[1]) else None for xy in zip(out_df['longitude'], out_df['latitude'])],
        crs=gdf_cities.crs
    )

    # Export with date in filename
    today = datetime.now().strftime("%m%d%Y")
    output_csv = f"FC_{today}.csv"
    output_shp = f"FC_{today}.shp"
    gdf_out.to_file(output_shp)
    out_df.to_csv(output_csv, index=False)
    print(out_df)
    if unplaced:
        print("\nCities with no match within 15km:")
        for city in unplaced:
            print(city)
    else:
        print("\nAll cities matched within 15km.")

Create_FC_Guide("cities0625.csv", "citypoints.shp", "frontline.shp")