"""
Political lexicon for classifying Twitter trending topics.
Sources:
  - Alex Gamez planning notes (planning.docx) — primary
  - Gentzkow & Shapiro (2010) congressional speech phrases
  - Common political hashtag knowledge

Labels: "right", "left", "neutral"
Usage:
    from lexicon import classify_topic
    label = classify_topic("MAGA")   # -> "right"
"""

import re

# ---------------------------------------------------------------
# RIGHT-LEANING keywords (conservative / MAGA / alt-right)
# ---------------------------------------------------------------
RIGHT_KEYWORDS = [
    # MAGA / Trump
    "maga", "trump", "makeamericagreatagain", "trumptrain", "trump2024",
    "trump2020", "trump2022", "kag", "keepamericagreat", "djt",
    "americafirst", "stopthesteal", "fraudwasreal",
    # GOP / Republican
    "republican", "gop", "rnc", "conservative", "tpusa",
    "turning point", "charlie kirk", "ben shapiro", "ted cruz",
    "desantis", "ron desantis", "marjorie taylor greene", "mtg",
    "matt gaetz", "jim jordan",
    # Alt-right / incel
    "redpill", "red pill", "bluepill", "blue pill", "blackpill",
    "black pill", "chad", "stacy", "looksmaxing", "looksmax",
    "8020 rule", "80 20 rule", "alpha male", "beta male",
    "soyboy", "soy boy", "cuck", "npc", "simp", "mewing",
    "andrew tate", "fresh and fit", "sneako", "myron gaines",
    "pearl", "adin ross", "jordan peterson", "incel",
    "femoid", "niceguy", "nice guy", "sigma",
    "high value man", "sexual market value", "smv",
    # Anti-left rhetoric
    "antifa bad", "defund fails", "anti-woke", "antiwoke",
    "woke mob", "cancel culture bad", "dei bad", "dei hire",
    "groomer", "let's go brandon", "fjb", "deep state",
    "qanon", "cabal", "great replacement",
    # Immigration (right framing)
    "build the wall", "illegal alien", "open borders bad",
    "border invasion", "securetheborder",
    # Pro-Israel (right framing)
    "hamas terrorist", "islamic terrorism", "standwithisrael",
    "idf", "never again", "proIsrael",
    # Energy / economy
    "clean coal", "drill baby drill", "energy independence",
    "no green new deal", "anti esg",
    # Social
    "all lives matter", "blue lives matter", "backtheblue",
    "thinblueline", "family values", "traditional values",
    "pro life", "prolife", "anti abortion",
    "gun rights", "2a", "second amendment", "nra",
    "rfk", "robert f kennedy",
]

# ---------------------------------------------------------------
# LEFT-LEANING keywords (progressive / liberal / left)
# ---------------------------------------------------------------
LEFT_KEYWORDS = [
    # Progressive politics
    "progressive", "democrat", "dnc", "liberal", "leftwing",
    "left wing", "bernie", "bernie sanders", "aoc",
    "alexandria ocasio cortez", "squad", "the squad",
    "elizabeth warren", "ilhan omar", "rashida tlaib",
    "hasan piker", "emma vigeland", "sam seder",
    "majority report", "the young turks", "tyt",
    # Policy
    "medicare for all", "m4a", "green new deal", "gnd",
    "taxtherich", "tax the rich", "wealth tax",
    "student debt", "student loan forgiveness",
    "living wage", "livable wage", "minimum wage hike",
    "universal healthcare", "ubi",
    "climate justice", "climate action", "renewableenergy",
    "renewable energy", "solar energy", "wind energy",
    # Social justice
    "blacklivesmatter", "blm", "black lives matter",
    "stopaapihate", "stop aapi hate",
    "nodapl", "landback", "land back",
    "abolish ice", "defund the police", "acab",
    "prison abolition", "restorative justice",
    # Reproductive rights
    "mybodymychoice", "prochoice", "pro choice",
    "abortion rights", "repeal the ban", "roevwade",
    "roe v wade",
    # Gender / feminist
    "metoo", "me too", "believewomen", "believe women",
    "equal pay", "gender equality", "feminist", "feminism",
    "womens rights", "wnba", "nwsl", "womens sports",
    "sexism", "misogyny",
    # LGBTQ+
    "pride", "lgbtq", "lgbt", "trans rights",
    "transrightsarehumanrights", "lovewins",
    "anti discrimination", "protecttranskids",
    # Palestine / anti-war
    "freepalestine", "free palestine", "ceasefire",
    "gazagenocide", "gaza", "occupation ends",
    "endtheoccupation", "palestinewillbefree",
    "antiwar", "anti war", "peacenow",
    # Immigration (left framing)
    "familiesbelongtogether", "nobanbonwall",
    "dreamers", "daca", "immigrantsarehumans",
    # Misinformation / science
    "followthescience", "climatechange is real",
    "vaccineswork", "vaccines work",
    # Anti-authoritarian
    "resist", "notmypresident", "impeach",
    "fascism warning", "antifascist",
]

# ---------------------------------------------------------------
# NEUTRAL / control topics (sports, entertainment, weather, etc.)
# ---------------------------------------------------------------
NEUTRAL_KEYWORDS = [
    "nfl", "nba", "mlb", "nhl", "mls", "ncaa",
    "superbowl", "super bowl", "world series", "nba finals",
    "oscars", "grammys", "emmys", "golden globes",
    "netflix", "hulu", "disney plus", "hbo",
    "taylor swift", "beyonce", "drake", "bad bunny",
    "hurricane", "earthquake", "tornado", "wildfire",
    "storm", "blizzard", "heat wave",
    "iphone", "apple", "google", "microsoft",
    "worldcup", "world cup", "olympics", "fifa",
    "thanksgiving", "christmas", "halloween", "new year",
    "movie", "film", "album", "concert", "tour",
]

# ---------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------
def _tokenize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower().strip())


def classify_topic(topic: str) -> str:
    """
    Returns 'right', 'left', or 'neutral'.
    Checks for keyword matches; 'neutral' is the default fallback.
    """
    t = _tokenize(topic)

    right_hits = sum(1 for kw in RIGHT_KEYWORDS if kw in t)
    left_hits  = sum(1 for kw in LEFT_KEYWORDS  if kw in t)
    neutral_hits = sum(1 for kw in NEUTRAL_KEYWORDS if kw in t)

    if right_hits == 0 and left_hits == 0 and neutral_hits == 0:
        return "neutral"   # unclassified defaults to neutral

    scores = {"right": right_hits, "left": left_hits, "neutral": neutral_hits}
    return max(scores, key=scores.get)


def classify_batch(topics: list) -> list:
    """Classify a list of topic strings. Returns list of labels."""
    return [classify_topic(t) for t in topics]


if __name__ == "__main__":
    # Quick smoke test
    tests = [
        ("MAGA", "right"),
        ("Black Lives Matter", "left"),
        ("Super Bowl", "neutral"),
        ("Andrew Tate", "right"),
        ("Free Palestine", "left"),
        ("Taylor Swift", "neutral"),
        ("gun rights", "right"),
        ("Medicare for All", "left"),
    ]
    print("Lexicon smoke test:")
    for topic, expected in tests:
        result = classify_topic(topic)
        status = "OK" if result == expected else f"FAIL (expected {expected})"
        print(f"  {topic:30s} -> {result:8s}  {status}")
