"""Configuration constants for PSC Incident Analysis."""

# Risk calculation weights
BASELINE_RISK_WEIGHT = 20
DEVIATION_RISK_WEIGHT = 40
SEVERITY_RISK_WEIGHT = 40

# Time-based parameters
ISSUE_HALF_LIFE = 365  # Days for issue weight to reduce by half
HALF_LIFE_CHANGES = 180  # 6 months for change impact to reduce by half
COUNT_WEIGHT = 0.6  # Weight for count vs severity in vessel risk

# Buffer zone for maximum issue counts
BUFFER_ZONE = 2

# Action code severity weights
WACTION = 0.7  # 70% weight on action codes vs PSC codes

# Maximum severity score
MAX_SEVERITY = 100

# Weight mapping for action codes
WEIGHT_MAP = {
    "10": 10, "14": 25, "15": 35, "16": 45, "17": 60, "18": 40, "19": 80,
    "21": 50, "26": 45, "30": 100, "35": 85, "36": 85, "40": 55, "45": 75,
    "46": 65, "47": 50, "48": 50, "49": 45, "50": 40, "55": 30, "60": 45,
    "65": 80, "70": 35, "80": 45, "81": 45, "85": 70, "95": 20, "96": 10,
    "99": 70
}

# Change weight mapping for dynamic factors
CHANGE_WEIGHT_MAPPING = {
    "Changes In Vessel Name And Call Sign": 5,
    "Changes In Vessel Flag": 10,
    "Changes In Vessel Class": 15,
    "Change In Vessel Ownership": 20,
    "Change In Ship Management": 15,
    "Changes In Vessel Critical Systems": 25
}

# Entity to issue type mapping
ENTITY_ISSUETYPE_MAPPING = {
    "R_OWNERS": ["Process", "Equipment"],
    "YARD": ["Equipment"],
    "YARD_COUNTRY": ["Equipment"],
    "FLAG_STATE": ["Process"],
    "MANAGER_GROUP": ["Process"],
    "ME_MAKE": ["Equipment"],
    "ME_MODEL": ["Equipment"],
    "VESSEL_CLASS": ["Process"],
    "NATIONALITY_OF_THE_CREW": ["Process"],
    "INSPECTOR": ["Process"],
    "MARINE_MANAGER": ["Process"],
    "MARINE_SUPERINTENDENT": ["Process"],
    "TECHNICAL_MANAGER": ["Process"]
}
