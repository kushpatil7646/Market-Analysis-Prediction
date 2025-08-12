# generate_competitor_data.py
import os, random
import numpy as np
import pandas as pd

RNG_SEED = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

OUT_DIR = "data_competitor_analysis"
OUT_CSV = os.path.join(OUT_DIR, "competitor_data.csv")
os.makedirs(OUT_DIR, exist_ok=True)

brands = [
    "SpiceMaster", "SpiceCraft", "PureSpice", "MasalaWorks",
    "KeralaGold", "DesiKitchen", "AromaLeaf", "RoyalHarvest"
]

categories = {
    "Turmeric": {"base": 7.5, "sub": ["Powder"], "origins": ["Kerala", "Maharashtra", "Odisha"]},
    "Red Chilli": {"base": 6.0, "sub": ["Powder", "Flakes"], "origins": ["Andhra Pradesh","Telangana","Karnataka"]},
    "Cumin": {"base": 5.8, "sub": ["Seeds","Powder"], "origins": ["Rajasthan","Gujarat"]},
    "Coriander": {"base": 5.2, "sub": ["Seeds","Powder"], "origins": ["Madhya Pradesh","Rajasthan"]},
    "Black Pepper": {"base": 12.5, "sub": ["Whole","Cracked"], "origins": ["Kerala","Tamil Nadu"]},
    "Cardamom": {"base": 18.0, "sub": ["Green Pods"], "origins": ["Kerala","Karnataka"]},
    "Mustard": {"base": 4.0, "sub": ["Yellow Seeds","Black Seeds"], "origins": ["Rajasthan","Haryana"]},
    "Fenugreek": {"base": 4.8, "sub": ["Leaves","Seeds"], "origins": ["Rajasthan","Madhya Pradesh"]},
    "Garam Masala": {"base": 9.0, "sub": ["Blend"], "origins": ["Maharashtra","Delhi","Punjab"]},
    "Clove": {"base": 13.5, "sub": ["Whole"], "origins": ["Tamil Nadu","Kerala"]},
    "Cinnamon": {"base": 10.0, "sub": ["Quills","Powder"], "origins": ["Kerala","Sri Lanka"]},
    "Bay Leaf": {"base": 3.9, "sub": ["Whole"], "origins": ["Uttarakhand","West Bengal"]},
    "Star Anise": {"base": 11.0, "sub": ["Whole"], "origins": ["Mizoram","Assam"]},
}

# Brand pricing tendencies (multipliers)
brand_bias = {
    "SpiceMaster": 1.06, "SpiceCraft": 1.00, "PureSpice": 1.03, "MasalaWorks": 0.98,
    "KeralaGold": 1.10, "DesiKitchen": 0.96, "AromaLeaf": 1.04, "RoyalHarvest": 1.08
}

# Size options (g) with price scaling (vs 100g baseline)
sizes_g = [50, 75, 100, 200, 250, 500]
size_price_mult = {50: 0.6, 75: 0.8, 100: 1.0, 200: 1.85, 250: 2.2, 500: 4.0}

# Description bits for variety
selling_points_pos = [
    "rich color", "earthy aroma", "vibrant hue", "bold heat", "warm, nutty notes",
    "subtle citrus finish", "pungent and sharp", "aromatic sweetness", "ideal for marinades",
    "slow-roasted profile", "stone-ground texture", "handpicked quality", "sun-dried",
    "single-origin", "small-batch crafted"
]
selling_points_neg = [
    "mild heat", "slightly uneven grind", "limited aroma retention", "color fades on cooking",
    "batch variability", "less pungent than expected"
]
use_cases = [
    "curries", "tadka", "marinades", "biryani", "chai", "desserts", "pickles",
    "lentils", "grills", "soups"
]
quality_tags = ["organic", "non-GMO", "pesticide-free", "stone-ground", "steam-sterilized"]

def make_desc(cat, sub, origin, is_org, pos_pts, maybe_neg):
    org_txt = "Organic " if is_org else ""
    lead = f"{org_txt}{cat} {sub}"
    pos = ", ".join(np.random.choice(selling_points_pos, size=np.random.randint(2,4), replace=False))
    uses = ", ".join(np.random.choice(use_cases, size=np.random.randint(1,3), replace=False))
    extras = ", ".join(np.random.choice(quality_tags, size=np.random.randint(1,3), replace=False))
    neg = ""
    if maybe_neg and np.random.rand() < 0.15:
        neg = f" Note: {np.random.choice(selling_points_neg)}."
    return (f"{lead} — {pos}. Sourced from {origin}. Great for {uses}. "
            f"Attributes: {extras}.{neg}")

def price_for_item(cat_base, brand, size, is_org, rating):
    # base → brand bias → size multiplier → organic premium → random noise → rating tilt
    p = cat_base
    p *= brand_bias[brand]
    p *= size_price_mult[size]
    if is_org:
        p *= 1.12  # organic premium ~12%
    # slight rating influence (higher rating nudges price up a touch)
    p *= (1 + (rating - 4.0) * 0.03)
    # noise
    p *= np.random.normal(1.0, 0.06)
    return round(max(1.99, p), 2)

def build_dataset(n_rows=200):
    rows = []
    for i in range(n_rows):
        brand = np.random.choice(brands, p=[0.14,0.14,0.14,0.12,0.10,0.12,0.12,0.12])
        cat = random.choice(list(categories.keys()))
        sub = random.choice(categories[cat]["sub"])
        origin = random.choice(categories[cat]["origins"])
        size = np.random.choice(sizes_g, p=[0.15,0.15,0.35,0.18,0.10,0.07])  # 100g most common
        is_org = np.random.rand() < (0.28 if cat in ["Turmeric","Black Pepper","Cardamom"] else 0.18)

        # rating & reviews (skewed realistic)
        rating = np.clip(np.random.normal(4.2, 0.35), 3.2, 5.0)
        reviews = int(np.clip(np.random.lognormal(mean=3.2, sigma=0.7), 5, 5000))

        # compute price
        cat_base = categories[cat]["base"]
        price = price_for_item(cat_base, brand, size, is_org, rating)

        # assemble text
        desc = make_desc(cat, sub, origin, is_org, selling_points_pos, maybe_neg=True)

        product_name = f"{brand} {('Organic ' if is_org else '')}{cat} {sub} {size}g"
        rows.append({
            "product_id": f"P{i+1:04d}",
            "brand": brand,
            "category": cat,
            "subcategory": sub,
            "product_name": product_name,
            "size_g": size,
            "is_organic": int(is_org),
            "origin_state": origin,
            "rating": round(rating, 2),
            "reviews_count": reviews,
            "price": price,
            "product_text": desc,
        })
    return pd.DataFrame(rows)

df = build_dataset(200)

# Light deduplication by (brand, product_name)
df = df.drop_duplicates(subset=["brand","product_name"]).reset_index(drop=True)

df.to_csv(OUT_CSV, index=False, encoding="utf-8")
print(f"✅ Wrote {len(df)} rows to {OUT_CSV}")
print(df.head(10).to_string(index=False))
