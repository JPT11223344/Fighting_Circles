from docx import Document
import pandas as pd
import geopandas as gpd
import os
from rapidfuzz import process, fuzz
from shapely.geometry import Point
from datetime import datetime
import re
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

#find red text
def is_run_reddish(run, r_min=200, g_max=80, b_max=80):
    color = run.font.color
    if color is None or color.rgb is None:
        return False
    rgb = color.rgb
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    return (r >= r_min) and (g <= g_max) and (b <= b_max)

#extracting the last time a phrase _____ oblast was mentioned and associating the city with that oblast 
def extract_red_phrases_with_oblast(docx_path):
    doc = Document(docx_path)
    oblast = None
    data = []
    for para in doc.paragraphs:
        text = para.text.strip()
        # Detect oblast
        if text.lower().endswith('oblast'):
            oblast = text.replace(' Oblast', '').replace(' oblast', '').strip()
            continue
        # Associating oblast with city
        if oblast:
            for run in para.runs:
                if run.font.color and run.font.color.rgb is not None:
                    rgb = run.font.color.rgb
                    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
                    if (r >= 200) and (g <= 80) and (b <= 80):
                        phrases = [s.strip() for s in re.split(r',|\band\b| {2,}|\t|\.', run.text) if s.strip()]
                        for phrase in phrases:
                            if not re.search(r'\d', phrase):
                                data.append({'Oblast': oblast, 'phrase': phrase})
    return data

def fuzzy_match_oblasts(df, gdf_cities, text_col='Oblast', adm1_col='ADM1_EN', min_score=70):
    shapefile_oblasts = gdf_cities[adm1_col].dropna().unique()
    shapefile_oblasts = [str(o).lower().strip() for o in shapefile_oblasts]

    def match_func(oblast):
        if pd.isnull(oblast):
            return None
        match, score, _ = process.extractOne(oblast.lower().strip(), shapefile_oblasts, scorer=fuzz.ratio)
        if score >= min_score:
            return match
        return None

    df['Oblast_matched'] = df[text_col].apply(match_func)
    return df

# ===========MAIN WORKFLOW==============
docx_path = r"C:\Users\jthompson\Desktop\FCauto\20250708 Military Collect.docx" #replace with your path
city_shapefile_path = "citypoints.shp"
frontline_shapefile_path = "frontline.shp"

def split_phrases(text):
    return [s.strip() for s in re.split(r',|\band\b| {2,}|\t|\.', text) if s.strip()]

def Create_FC_Guide(
    docx_path,
    city_shapefile_path,
    frontline_shapefile_path
):
    # Extract from Word doc
    data = extract_red_phrases_with_oblast(docx_path)
    df = pd.DataFrame(data)

    # Hardcoded oblasts 
    oblast_hardcode = {
        "donetsk": "donetskyi",
        "western zaporizhia": "zaporizkyi",
        "kharkiv": "kharkivskyi"
    }
    df['Oblast'] = df['Oblast'].replace(oblast_hardcode)

    #cleaning 
    remove_phrases = ["ruaf","army", "brigade", "the", "spetsnaz", "uaf", "ivo", "brig", "towards", "dir"]
    pattern = '|'.join([fr"\b{re.escape(word)}\b" for word in remove_phrases])
    df = df[~df['phrase'].str.lower().str.contains(pattern)]
    df['FC'] = df['phrase'].str.lower().str.strip()


    gdf_cities = gpd.read_file(city_shapefile_path)  
    df = fuzzy_match_oblasts(df, gdf_cities, text_col='Oblast', adm1_col='ADM1_EN', min_score=50)

    #  further cleaning
    df = df[df['FC'].str.len() > 2]
    df = df[df['FC'].str.match(r'^[a-zA-Z\s\-]+$')] #TAG THIS FOR LATER 

    # matching city names
    city_names = set(gdf_cities['ADM4_EN'].str.lower().str.strip())
    def get_best_city_match(phrase, city_names):
        match, score, _ = process.extractOne(phrase, city_names, scorer=fuzz.ratio)
        return pd.Series({'city_fuzzy_match': match, 'fuzzy_score': score})
    df[['city_fuzzy_match', 'fuzzy_score']] = df['FC'].apply(lambda x: get_best_city_match(x, city_names))

    city_csv_col = 'FC'
    city_shp_col = 'ADM4_EN'

    gdf_cities = gpd.read_file(city_shapefile_path)
    gdf_poly = gpd.read_file(frontline_shapefile_path)
    df = fuzzy_match_oblasts(df, gdf_cities, text_col='Oblast', adm1_col='ADM1_EN')

    df[city_csv_col] = df[city_csv_col].str.lower().str.strip()
    gdf_cities[city_shp_col] = gdf_cities[city_shp_col].str.lower().str.strip()

    #setting projection
    metric_crs = gdf_cities.estimate_utm_crs()
    gdf_cities_proj = gdf_cities.to_crs(metric_crs)
    gdf_poly_proj = gdf_poly.to_crs(metric_crs)

    results = []
    unplaced = []

    #best city match based on distance and oblast
    for idx, row in df.iterrows():
        city = row['city_fuzzy_match']
        oblast = row['Oblast_matched']
        candidates = gdf_cities_proj[gdf_cities_proj[city_shp_col] == city].copy()
        if candidates.empty:
            oblast_cands = gdf_cities_proj[gdf_cities_proj['ADM1_EN'].str.lower().str.strip() == (oblast if pd.notnull(oblast) else "")]
            if not oblast_cands.empty:
                match_name, score, _ = process.extractOne(city, oblast_cands[city_shp_col].tolist(), scorer=fuzz.ratio)
                candidates = oblast_cands[oblast_cands[city_shp_col] == match_name].copy()
        if candidates.empty:
            unplaced.append(city)
            continue

        # Distance calculation
        candidates['distance_to_border'] = candidates.geometry.apply(
            lambda geom: gdf_poly_proj.geometry.unary_union.distance(geom)
        )

        # MCE scoring
        candidates['oblast_score'] = (candidates['ADM1_EN'].str.lower().str.strip() == (oblast if pd.notnull(oblast) else "")).astype(float)
        candidates['distance_score'] = 1 - (candidates['distance_to_border'] / 15000)
        candidates['distance_score'] = candidates['distance_score'].clip(lower=0, upper=1)
        candidates['mce_score'] = 0.2 * candidates['oblast_score'] + 0.8 * candidates['distance_score']
        best = candidates.sort_values('mce_score', ascending=False).iloc[0]
        results.append(pd.concat([row, best]))

    # Output
    closest_per_city = gpd.GeoDataFrame(results, geometry='geometry', crs=metric_crs).to_crs(gdf_cities.crs)
    closest_per_city['latitude'] = closest_per_city.geometry.apply(lambda geom: geom.y if geom is not None else None)
    closest_per_city['longitude'] = closest_per_city.geometry.apply(lambda geom: geom.x if geom is not None else None)
    out_df = closest_per_city[[city_csv_col, 'phrase', 'city_fuzzy_match', 'fuzzy_score', 'latitude', 'longitude', 'Oblast', 'Oblast_matched', 'ADM1_EN', 'distance_to_border', 'mce_score']]
 
    far_df= out_df[out_df['distance_to_border'] > 15000]
    out_df = out_df[out_df['mce_score'] > 0.2]

    out_df = out_df.drop_duplicates(subset=['latitude', 'longitude'])

    gdf_out = gpd.GeoDataFrame(
        out_df,
        geometry=[Point(xy) if pd.notnull(xy[0]) and pd.notnull(xy[1]) else None for xy in zip(out_df['longitude'], out_df['latitude'])],
        crs=gdf_cities.crs
    )

    # Hard relocations 
    manual_locations = {
        "serhiivka": (49.362507, 37.954753),      
        "hryhorivkasouth of hryhorivka": (48.638785, 37.853184),   
        "myrne": (49.068183, 37.928193),
        "toward pishchane": (48.235077,37.106922),
        "pishchane": (48.235077,37.106922),
        "katerynivka": (48.4161569, 37.7552697)  
    }

    for city, (lat, lon) in manual_locations.items():
        mask = out_df[city_csv_col] == city
        out_df.loc[mask, 'latitude'] = lat
        out_df.loc[mask, 'longitude'] = lon

    # Rebuild geometry after hardcoding 
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
    far_df=far_df.drop_duplicates(subset=['latitude', 'longitude'])
    if len(far_df) > 0:
        print("\n Cities with failed matches listed below with best match, fuzzy score, and distance to frontline shown:\n")
        unique_pairs = set()
        for index, row in far_df.iterrows():
            if row['fuzzy_score'] >= 90 and row['distance_to_border'] <= 15000:
                unique_pairs.add((row[city_csv_col], row['city_fuzzy_match'], row['fuzzy_score'], row['distan`ce_to_border']))
        for orig, match, score, dist in unique_pairs:
            print(f"{orig} → {match} | Fuzzy: {score} | Distance: {dist:.1f}m")
    else:
        print("\nAll cities matched within 15km.")

    remove_phrases = ["army", "brigade", "the", "spetsnaz", "uaf", "ivo"]
    pattern = '|'.join([fr"\b{re.escape(word)}\b" for word in remove_phrases])
    df = df[~df['phrase'].str.lower().str.contains(pattern)]
    

Create_FC_Guide(
    docx_path="20250801 Russian Military Collect .docx",
    city_shapefile_path="citypoints.shp",
    frontline_shapefile_path="frontline.shp"
)