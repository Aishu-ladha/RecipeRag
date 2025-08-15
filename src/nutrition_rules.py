from typing import Dict, Any

# Simple rule thresholds (can be expanded/tuned)
HEALTH_RULES = {
    "diabetes": {
        "sugar_g_max": 12,      # prefer lower sugar per serving
        "carbs_g_max": 60,      # moderate carbs
        "fiber_g_min": 5,       # encourage fiber
    },
    "hypertension": {
        "sodium_mg_max": 500,   # limit sodium
    },
    "celiac": {
        "must_have_tags": ["gluten_free"],
    },
    "high_cholesterol": {
        "fat_g_max": 20,        # simplify: limit total fat
    },
}

def passes_health_filters(row: Dict[str, Any], conditions: list[str]) -> bool:
    for cond in conditions:
        rule = HEALTH_RULES.get(cond)
        if not rule:
            continue
        # Check numeric max thresholds
        for key, val in rule.items():
            if key.endswith("_max"):
                field = key.replace("_max", "")
                if row.get(field) is not None and float(row[field]) > float(val):
                    return False
            if key.endswith("_min"):
                field = key.replace("_min", "")
                if row.get(field) is not None and float(row[field]) < float(val):
                    return False
        # Tag requirements
        must_tags = rule.get("must_have_tags", [])
        if must_tags:
            tags = set((row.get("tags") or "").split(";"))
            if not set(must_tags).issubset(tags):
                return False
    return True

def violates_diet_restrictions(row: Dict[str, Any], diets: list[str], allergens_to_avoid: list[str]) -> bool:
    tags = set((row.get("tags") or "").split(";"))
    allergens = set((row.get("allergens") or "").split(";"))
    ingredients = set((row.get("ingredients") or "").split(";"))

    # Simple checks for diets via tags
    for d in diets:
        if d in ["vegan", "vegetarian", "gluten_free", "low_carb", "high_protein"]:
            if d not in tags:
                return True

    # Allergen avoidance
    for a in allergens_to_avoid:
        a = a.strip().lower()
        if a and (a in allergens or a in (ing.lower() for ing in ingredients)):
            return True

    return False
