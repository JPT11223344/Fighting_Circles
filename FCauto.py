from docx import Document
import pandas as pd
import geopandas as gpd
import os
from rapidfuzz import process, fuzz
from shapely.geometry import Point
import itertools
from datetime import datetime
import re

def is_run_reddish(run, r_min=200, g_max=80, b_max=80):
    color = run.font.color
    if color is None or color.rgb is None:
        return False
    rgb = color.rgb
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    return (r >= r_min) and (g <= g_max) and (b <= b_max)

def extract_all_reddish_text(docx_path):
    doc = Document(docx_path)
    red_text = []
    for para in doc.paragraphs:
        for run in para.runs:
            if is_run_reddish(run):
                red_text.append(run.text)
    return ''.join(red_text)

def split_phrases(text):
    return [s.strip() for s in re.split(r',|\band\b| {2,}|\t|\.', text) if s.strip()]

def Create_FC_Guide(
    docx_path,
    city_shapefile_path,
    frontline_shapefile_path,
    distance_threshold=15000
):
    # Extract red phrases from Word doc
    red_string = extract_all_reddish_text(docx_path)
    phrases = split_phrases(red_string)
    special_cases = {"hryhorivka", "myrne", "serhiivka", "stepanivka"}
    filtered_phrases = [
        p for p in phrases
        if not re.search(r'\d', p) and " RUAF" not in p and " UAF" not in p
    ]
    # Build DataFrame as if it were the CSV
    df = pd.DataFrame({'FC': filtered_phrases})

    city_csv_col = 'FC'
    city_shp_col = 'ADM4_EN'
    fuzzy_min_score = 40

    gdf_cities = gpd.read_file(city_shapefile_path)
    gdf_poly = gpd.read_file(frontline_shapefile_path)
    df[city_csv_col] = df[city_csv_col].str.lower().str.strip()
    gdf_cities[city_shp_col] = gdf_cities[city_shp_col].str.lower().str.strip()

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

    def fuzzy_match_with_alternatives(city, choices, scorer=fuzz.ratio, min_score=0):
        match, score, idx = process.extractOne(city, choices, scorer=scorer)
        if score >= min_score:
            return match
        for alt in generate_alternatives(city):
            match, score, idx = process.extractOne(alt, choices, scorer=scorer)
            if score >= min_score:
                return match
        return None

    metric_crs = gdf_cities.estimate_utm_crs()
    gdf_cities_proj = gdf_cities.to_crs(metric_crs)
    gdf_poly_proj = gdf_poly.to_crs(metric_crs)
    merged = pd.merge(df, gdf_cities_proj, left_on=city_csv_col, right_on=city_shp_col, how='inner')
    unmatched = df[~df[city_csv_col].isin(merged[city_csv_col])]
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

    line = gdf_poly_proj.geometry.unary_union
    merged['distance_to_border'] = merged.geometry.apply(lambda geom: line.distance(geom) if geom is not None else float('inf'))

    results = []
    unplaced = []

    for city, group in merged.groupby(city_csv_col):
        group_sorted = group.sort_values('distance_to_border')
        found = False
        city_clean = city.strip().lower()
        if city_clean in special_cases:
            matches_within_thresh = group_sorted[group_sorted['distance_to_border'] <= distance_threshold]
            if len(matches_within_thresh) > 1:
                # Select the second closest
                results.append(matches_within_thresh.iloc[1])
                found = True
            elif len(matches_within_thresh) == 1:
                # Only one match within threshold, use it
                results.append(matches_within_thresh.iloc[0])
                found = True
            else:
                # Fallback to best fuzzy match (regardless of distance)
                candidate_names = group_sorted[city_shp_col].tolist()
                fuzzy_name = fuzzy_match_with_alternatives(city, candidate_names, min_score=fuzzy_min_score)
                if fuzzy_name:
                    fuzzy_row = group_sorted[group_sorted[city_shp_col] == fuzzy_name]
                    if not fuzzy_row.empty:
                        results.append(fuzzy_row.iloc[0])
                        found = True
        else:
            matches_within_thresh = group_sorted[group_sorted['distance_to_border'] <= distance_threshold]
            if not matches_within_thresh.empty:
                results.append(matches_within_thresh.iloc[0])
                found = True
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

    closest_per_city = gpd.GeoDataFrame(results, geometry='geometry', crs=metric_crs).to_crs(gdf_cities.crs)
    closest_per_city['latitude'] = closest_per_city.geometry.apply(lambda geom: geom.y if geom is not None else None)
    closest_per_city['longitude'] = closest_per_city.geometry.apply(lambda geom: geom.x if geom is not None else None)
    out_df = closest_per_city[[city_csv_col, 'latitude', 'longitude']]

    # Remove specified coordinates from the final DataFrame
    coords_to_remove = {
        
    }
    out_df = out_df[~out_df.apply(lambda row: (round(row['latitude'], 6), round(row['longitude'], 6)) in coords_to_remove, axis=1)]

    gdf_out = gpd.GeoDataFrame(
        out_df,
        geometry=[Point(xy) if pd.notnull(xy[0]) and pd.notnull(xy[1]) else None for xy in zip(out_df['longitude'], out_df['latitude'])],
        crs=gdf_cities.crs
    )

    # Manually relocation
    manual_locations = {
        "serhiivka": (49.362507, 37.954753),      
        "hryhorivkasouth of hryhorivka": (48.638785, 37.853184),   
        "myrne": (49.068183, 37.928193),
        "toward pishchane": (48.235077,37.106922),
        "pishchane": (48.235077,37.106922)  
    }

    for city, (lat, lon) in manual_locations.items():
        mask = out_df[city_csv_col] == city
        out_df.loc[mask, 'latitude'] = lat
        out_df.loc[mask, 'longitude'] = lon

    # Rebuild geometry after manual override
    gdf_out = gpd.GeoDataFrame(
        out_df,
        geometry=[Point(xy) if pd.notnull(xy[0]) and pd.notnull(xy[1]) else None for xy in zip(out_df['longitude'], out_df['latitude'])],
        crs=gdf_cities.crs
    )

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

Create_FC_Guide(
    docx_path="20250625 Military Collect.docx",
    city_shapefile_path="citypoints.shp",
    frontline_shapefile_path="frontline.shp"
)