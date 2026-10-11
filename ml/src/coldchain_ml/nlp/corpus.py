"""SYNTHETIC incident-report generator for a cold-store warehouse.

Four classes: damage, temperature, delay, safety. Every report is built from a *family*: a set of
sentence structures plus its own pool of event phrases. Families are partitioned whole into
train / validation / test, so the held-out sets contain sentence structures and phrasings the model
has never seen. Within a family the surface varies (location, ids, product, actor, time) and noisy
real-world habits are applied: typos, shorthand, dropped punctuation, odd casing, filler, and
(25%) a secondary issue from ANOTHER class mentioned after the main one. The label is always the
class of the main issue.

This is generated text. A model that scores well here has learned this generator, not real
incident reports; see docs/model-cards/incident-nlp.md.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

CLASSES = ("damage", "temperature", "delay", "safety")

LOCS = [
    "dock 3",
    "dock 7",
    "dock 11",
    "zone A",
    "zone B",
    "zone D",
    "aisle 12",
    "aisle 4",
    "freezer 2",
    "chill room 1",
    "staging lane 4",
    "goods-in",
    "outbound bay 5",
    "rack C14",
    "rack F02",
    "crane aisle 3",
    "the blast freezer",
    "pick face 18",
    "mezzanine",
    "the plant room",
]
ACTORS = [
    "the night picker",
    "the driver",
    "an FLT operator",
    "the shift supervisor",
    "a temp worker",
    "the goods-in clerk",
    "the cleaner",
    "a QA tech",
    "the loader",
    "one of the pickers",
]
PRODUCTS = [
    "frozen peas",
    "ice cream tubs",
    "chicken breast",
    "fish fingers",
    "frozen pizza",
    "mixed berries",
    "oven chips",
    "ready meals",
    "dumplings",
    "prawns",
    "burger patties",
    "garlic bread",
]
CUSTOMERS = [
    "Tesco",
    "Albert Heijn",
    "Lidl",
    "Carrefour",
    "Jumbo",
    "Aldi",
    "the Rotterdam store",
    "retail DC 4",
]
TIMES = [
    "at 02:15",
    "around 6am",
    "this morning",
    "during shift change",
    "yesterday evening",
    "at 14:40",
    "just before break",
    "on nights",
    "at about 22:30",
    "first thing today",
    "late afternoon",
]
FILLER = [
    "Supervisor informed.",
    "Photo attached.",
    "Please advise.",
    "Will update later.",
    "Logged in WMS.",
    "Raised with the team lead.",
    "No further action yet.",
    "Ref to follow.",
    "Checked on site.",
    "Reported by the shift lead.",
]
# Context appended to reports of ANY class (decouples ids/products/places from the label).
CONTEXT = [
    "Ref {id}.",
    "Load was {product}.",
    "({product}, {id})",
    "Seen by {actor}.",
    "Area: {loc}.",
    "Customer {customer}.",
    "Trailer {trailer}.",
    "{time}.",
    "Pallet {id}, {product}.",
    "Raised from {loc}.",
]
# Secondary issue clauses (one pool per class) used as distractors after the main issue.
SECONDARY = {
    "damage": [
        "A couple of cartons also looked dented.",
        "Some wrap was torn as well.",
        "One box was crushed too.",
    ],
    "temperature": [
        "The room also felt warmer than usual.",
        "Temp was a little high as well.",
        "The alarm had beeped earlier too.",
    ],
    "delay": [
        "This also put us behind schedule.",
        "The next load will be late as well.",
        "We are running behind anyway.",
    ],
    "safety": [
        "Someone nearly got hurt as well.",
        "The floor was slippery too.",
        "A pedestrian was close by.",
    ],
}


@dataclass(frozen=True)
class Family:
    name: str
    label: str
    templates: tuple[str, ...]
    events: tuple[str, ...]


def F(name: str, label: str, templates: list[str], events: list[str]) -> Family:
    return Family(name, label, tuple(templates), tuple(events))


FAMILIES: list[Family] = [
    # ------------------------------------------------------------------ damage
    F(
        "damage/forklift_strike",
        "damage",
        [
            "{actor} reversed the forklift into {id} at {loc}, {event}.",
            "{id} {event} after a forklift strike {time}.",
            "Forklift hit {id} in {loc} {time}: {event}.",
        ],
        [
            "bottom cartons crushed",
            "stretch wrap torn and two layers shifted",
            "pallet board split",
            "corner cartons caved in",
            "product spilling from ripped boxes",
        ],
    ),
    F(
        "damage/dropped",
        "damage",
        [
            "{id} {event} while being put away in {loc}.",
            "{event} - {id}, {loc}, {time}.",
            "During putaway {id} {event}.",
        ],
        [
            "collapsed into the aisle",
            "toppled off the rack",
            "slid off the forks and hit the floor",
            "was leaning badly and the top tier fell",
            "came down off the beam",
        ],
    ),
    F(
        "damage/packaging",
        "damage",
        [
            "Found {n} cartons {event} on {id} ({product}) {time}.",
            "{n} outers {event}, {id}, {product}.",
            "{product} on {id} has {n} cartons {event}.",
        ],
        ["soaked through and soft", "with torn corners", "leaking brine", "crushed flat", "with burst seams"],
    ),
    F(
        "damage/inbound",
        "damage",
        [
            "Goods-in rejected {id} from {trailer}: {event}.",
            "{trailer} unloaded, {id} {event}.",
            "Receipt note for {id} off {trailer}: {event}.",
        ],
        [
            "pallet received broken",
            "outers visibly damaged",
            "wrap missing and boxes dented",
            "damaged in transit per the driver",
            "load shifted and boxes split",
        ],
    ),
    F(
        "damage/pallet_base",
        "damage",
        [
            "{id} unfit to store, {event}.",
            "Cannot rack {id} at {loc}, {event}.",
            "{id} {event}, pulled from stock.",
        ],
        [
            "cracked bottom boards",
            "protruding nails",
            "pallet deck broken",
            "base boards missing",
            "runner snapped",
        ],
    ),
    F(
        "damage/qa_product",
        "damage",
        [
            "QA hold on {product} {id}: {event}.",
            "Quality flagged {id} ({product}), {event}.",
            "{product} {id} on hold - {event}.",
        ],
        [
            "bags punctured",
            "retail packs flattened",
            "cartons wet and misshapen",
            "lids cracked",
            "tubs deformed",
        ],
    ),
    F(
        "damage/crane",
        "damage",
        [
            "The crane dropped {id} in {loc}: {event}.",
            "ASRS fault, {id} {event}.",
            "{loc} shuttle mishandled {id}, {event}.",
        ],
        [
            "load fell about 2m",
            "pallet wedged and deformed",
            "carton stack crushed against the upright",
            "boxes burst on impact",
            "wrap ripped open",
        ],
    ),
    F(
        "damage/telegraphic",
        "damage",
        ["dmg {id} {loc} {event}", "{loc} {id} dmg {event}", "damaged {id} {event} {time}"],
        ["crushed", "torn wrap", "broken plt", "wet boxes", "split cartons"],
    ),
    # ------------------------------------------------------------- temperature
    F(
        "temperature/alarm",
        "temperature",
        [
            "Temp alarm in {loc} {time}: {event}.",
            "{loc} high temperature alert, {event}.",
            "Alarm {time} - {loc} - {event}.",
        ],
        [
            "probe reading -12C against a -25C setpoint",
            "air temp above limit for 40 minutes",
            "high temp alert not cleared",
            "reading -15 and still rising",
            "setpoint exceeded since the early hours",
        ],
    ),
    F(
        "temperature/door",
        "temperature",
        [
            "{loc} door {event}, temperature climbing.",
            "Door at {loc} {event} {time}, room warming up.",
            "{actor} noticed the {loc} door {event}; temps rising.",
        ],
        [
            "left open after loading",
            "would not close fully",
            "stuck ajar overnight",
            "propped open during putaway",
            "left wide open for a long time",
        ],
    ),
    F(
        "temperature/product",
        "temperature",
        ["{id} {product} {event}.", "{product} on {id} {event} {time}.", "Checked {id}: {product} {event}."],
        [
            "arrived soft to touch",
            "is showing signs of thaw",
            "had a core temp of -9C on the probe check",
            "has ice crystals that melted and refroze",
            "feels warm to the touch",
        ],
    ),
    F(
        "temperature/equipment",
        "temperature",
        [
            "Compressor in {loc} {event}.",
            "{loc} refrigeration {event} {time}.",
            "Cooling plant {event}, {loc} affected.",
        ],
        [
            "tripped and the zone warmed to -17",
            "cut out for about an hour",
            "is short cycling and temps are drifting up",
            "threw a fault code and the freezer is warming",
            "lost suction pressure",
        ],
    ),
    F(
        "temperature/defrost",
        "temperature",
        [
            "Defrost cycle {event} in {loc}.",
            "{loc} defrost {event} {time}.",
            "Evaporator defrost {event}, {loc}.",
        ],
        [
            "ran long and the zone went above -20",
            "did not terminate",
            "started during putaway and temps spiked",
            "overran its window",
            "left the coil iced and the room warm",
        ],
    ),
    F(
        "temperature/reefer",
        "temperature",
        [
            "Reefer {trailer} {event}.",
            "{trailer} refrigeration unit {event}.",
            "On arrival {trailer} {event}.",
        ],
        [
            "set point -22 but returned air -10",
            "unit failed en route and product was warm on arrival",
            "logger shows 3h above -18",
            "was running warm the whole trip",
            "had the fridge unit switched off",
        ],
    ),
    F(
        "temperature/logger",
        "temperature",
        [
            "Temperature logger on {id} {event}.",
            "Data logger for {id} {event}, {product}.",
            "Logger {id} {event} {time}.",
        ],
        [
            "shows an excursion to -8C",
            "flagged out of range between 02:00 and 04:00",
            "reports a breach of the cold chain",
            "recorded 5 hours above limit",
            "triggered a deviation",
        ],
    ),
    F(
        "temperature/telegraphic",
        "temperature",
        ["temp excursion {loc} {event}", "{loc} temp {event} {time}", "excursion {id} {event}"],
        ["-14C", "warm", "alarm", "over limit", "too high"],
    ),
    # ------------------------------------------------------------------- delay
    F(
        "delay/inbound_late",
        "delay",
        ["{trailer} {event}.", "Inbound {trailer} {event} {time}.", "{trailer} for {loc} {event}."],
        [
            "arrived 3 hours after its slot",
            "is still not at the gate",
            "was a no show for the 06:00 booking",
            "is held at the border",
            "turned up a day late",
        ],
    ),
    F(
        "delay/outbound_late",
        "delay",
        [
            "Outbound to {customer} {event}.",
            "{customer} load {event} {time}.",
            "Dispatch for {customer} {event}.",
        ],
        [
            "missed the cutoff by 40 minutes",
            "left late because pickers were behind",
            "is delayed, loads not ready",
            "slot rebooked to tomorrow",
            "will not make the delivery window",
        ],
    ),
    F(
        "delay/dock_queue",
        "delay",
        ["{loc} {event}.", "Queue at {loc} {time}: {event}.", "Dock congestion - {event}."],
        [
            "backed up with 5 trucks waiting",
            "no door free and drivers queuing",
            "slot overrun by 90 minutes",
            "trucks stacking up in the yard",
            "turnaround is well over target",
        ],
    ),
    F(
        "delay/capacity",
        "delay",
        [
            "{event}, so putaway is behind.",
            "{event} {time}, backlog building in {loc}.",
            "Because {event} we are behind plan.",
        ],
        [
            "we are short of operators on nights",
            "the WMS was down for 25 minutes",
            "RF scanners are not syncing",
            "no forklift was available",
            "the crane was in maintenance",
        ],
    ),
    F(
        "delay/paperwork",
        "delay",
        ["{id} {event}.", "{trailer} / {id}: {event}.", "Release of {id} {event}."],
        [
            "held, CMR not signed",
            "is waiting on customs paperwork",
            "cannot be released without documentation",
            "has a PO mismatch under investigation",
            "is stuck pending the quality certificate",
        ],
    ),
    F(
        "delay/picking",
        "delay",
        [
            "Picking for {customer} {event}.",
            "{loc} pick wave {event} {time}.",
            "Order for {customer} {event}.",
        ],
        [
            "is running 2 hours behind plan",
            "was not finished by cutoff",
            "stalled waiting on replenishment",
            "is behind because of short stock",
            "has slipped past its promised time",
        ],
    ),
    F(
        "delay/carrier",
        "delay",
        ["Carrier {event}.", "The haulier {event} for {trailer}.", "Collection {trailer} {event}."],
        [
            "rescheduled the collection twice",
            "driver hours expired, will wait until morning",
            "truck broke down on the motorway",
            "cancelled at short notice",
            "cannot attend before Thursday",
        ],
    ),
    F(
        "delay/telegraphic",
        "delay",
        ["late {trailer} {event}", "{trailer} delay {event}", "{loc} slot {event}"],
        ["2h", "delay at gate", "missed slot", "waiting", "behind schedule"],
    ),
    # ------------------------------------------------------------------ safety
    F(
        "safety/slip",
        "safety",
        [
            "{actor} {event} near {loc}.",
            "Slip incident {time}: {actor} {event}.",
            "{actor} {event} by {loc} {time}.",
        ],
        [
            "slipped on an ice patch and hurt their wrist",
            "fell on the wet floor",
            "went over on frost by the door",
            "slipped on a pallet wrap and twisted an ankle",
            "took a fall on the ramp",
        ],
    ),
    F(
        "safety/near_miss",
        "safety",
        [
            "Near miss at {loc}: forklift {event}.",
            "{time} - near miss, forklift {event}, {loc}.",
            "Close call in {loc}, truck {event}.",
        ],
        [
            "almost hit a pedestrian",
            "came round the corner too fast",
            "reversed without the horn and a pedestrian jumped clear",
            "came within a metre of a picker",
            "did not stop at the crossing",
        ],
    ),
    F(
        "safety/hazard",
        "safety",
        [
            "Hazard reported in {loc}: {event}.",
            "{loc} hazard, {event} {time}.",
            "Unsafe condition - {event} ({loc}).",
        ],
        [
            "ammonia smell near the plant room",
            "blocked emergency exit",
            "damaged racking leg with a load at risk",
            "oil leak on the floor",
            "loose cable across the walkway",
        ],
    ),
    F(
        "safety/injury",
        "safety",
        ["{actor} {event} {time}.", "Injury report: {actor} {event}.", "{actor} {event} in {loc}."],
        [
            "trapped fingers in the pallet jack handle",
            "strained their back lifting cartons",
            "cut a hand on the strapping",
            "felt dizzy after a long spell in the freezer",
            "caught a sleeve in the conveyor",
        ],
    ),
    F(
        "safety/ppe",
        "safety",
        [
            "{actor} found {event} in {loc}.",
            "PPE breach: {actor} seen {event}.",
            "Spot check {time}, {actor} {event}.",
        ],
        [
            "working without hi-vis and thermal gloves",
            "not wearing safety shoes",
            "with no helmet near the crane",
            "using a phone while driving the truck",
            "on the freezer floor without a cold suit",
        ],
    ),
    F(
        "safety/cold_stress",
        "safety",
        ["{actor} {event}.", "{actor} {event} {time}, first aid called.", "Welfare issue: {actor} {event}."],
        [
            "was locked in the freezer for ten minutes",
            "showed signs of frostbite on the fingertips",
            "reported numb hands after a long shift",
            "collapsed briefly after the cold room",
            "had hypothermia symptoms",
        ],
    ),
    F(
        "safety/fire",
        "safety",
        ["{event} in {loc}.", "{loc}: {event} {time}.", "Alert - {event}, {loc}."],
        [
            "fire alarm triggered and the area was evacuated",
            "smoke from a charger bay",
            "extinguisher missing from its bracket",
            "sprinkler head leaking",
            "fire door wedged open",
        ],
    ),
    F(
        "safety/telegraphic",
        "safety",
        ["H&S {loc} {event}", "{loc} {event} {time}", "safety: {event} {loc}"],
        ["injury", "near miss", "slip", "hazard", "first aid"],
    ),
]

SHORTHAND = [
    (r"\bpallet\b", ["plt", "pallet", "pallet"]),
    (r"\btemperature\b", ["temp", "temperature"]),
    (r"\bforklift\b", ["FLT", "fork truck", "forklift"]),
    (r"\bwarehouse\b", ["WH", "warehouse"]),
    (r"\bcartons\b", ["ctns", "cartons"]),
    (r"\bapproximately\b", ["approx", "approximately"]),
]


def _typo(word: str, rng: random.Random) -> str:
    if len(word) < 5:
        return word
    i = rng.randrange(1, len(word) - 2)
    kind = rng.random()
    if kind < 0.4:  # swap neighbours
        return word[:i] + word[i + 1] + word[i] + word[i + 2 :]
    if kind < 0.8:  # drop a char
        return word[:i] + word[i + 1 :]
    return word[:i] + word[i] + word[i:]  # double a char


def _noise(text: str, rng: random.Random) -> str:
    for pat, options in SHORTHAND:
        if rng.random() < 0.35:
            text = re.sub(pat, lambda m, options=options: rng.choice(options), text, flags=re.I)
    words = [_typo(w, rng) if rng.random() < 0.05 else w for w in text.split(" ")]
    text = " ".join(words)
    r = rng.random()
    if r < 0.22:
        text = text.lower()
    elif r < 0.25:
        text = text.upper()
    if rng.random() < 0.3:
        text = re.sub(r"[.,;:()]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _slots(rng: random.Random) -> dict[str, str]:
    pid = rng.choice(
        [
            f"plt {rng.randint(1000, 99999)}",
            f"PLT-{rng.randint(1000, 9999)}",
            f"pallet {rng.randint(100, 9999)}",
            f"SSCC ...{rng.randint(10000, 99999)}",
        ]
    )
    trailer = rng.choice(
        [
            f"TRL {rng.randint(1000, 9999)}",
            f"trailer {rng.randint(100, 999)}",
            f"truck {rng.choice('ABCDEFG')}{rng.randint(10, 99)}-{rng.choice('XYZ')}{rng.choice('KLMN')}",
        ]
    )
    return {
        "loc": rng.choice(LOCS),
        "actor": rng.choice(ACTORS),
        "id": pid,
        "trailer": trailer,
        "time": rng.choice(TIMES),
        "product": rng.choice(PRODUCTS),
        "customer": rng.choice(CUSTOMERS),
        "n": str(rng.randint(2, 40)),
    }


def make_report(fam: Family, rng: random.Random) -> str:
    slots = _slots(rng)
    text = rng.choice(fam.templates).format(event=rng.choice(fam.events), **slots)
    text = text[0].upper() + text[1:]
    # Class-independent context. Without it, slots that some families happen to use (pallet ids,
    # products) become shortcuts for the class: a flaw found when a held-out family was misread.
    if rng.random() < 0.4:
        text = f"{text} {rng.choice(CONTEXT).format(**slots)}"
    if rng.random() < 0.25:  # a secondary issue from ANOTHER class; the label stays the main one
        other = rng.choice([c for c in CLASSES if c != fam.label])
        text = f"{text} {rng.choice(SECONDARY[other])}"
    if rng.random() < 0.35:
        text = f"{text} {rng.choice(FILLER)}"
    return _noise(text, rng)


def family_split(seed: int = 0) -> dict[str, str]:
    """family name -> split. Per class: 5 train, 1 validation, 2 test families, chosen at random."""
    rng = random.Random(seed)
    out: dict[str, str] = {}
    for label in CLASSES:
        names = [f.name for f in FAMILIES if f.label == label]
        rng.shuffle(names)
        for n in names[:5]:
            out[n] = "train"
        out[names[5]] = "val"
        for n in names[6:]:
            out[n] = "test"
    return out


def build_corpus(seed: int = 0, per_family: dict[str, int] | None = None):
    """DataFrame of text, label, family, split. Exact duplicate texts are dropped (keeping the first)."""
    import pandas as pd

    per_family = per_family or {"train": 160, "val": 120, "test": 120}
    split = family_split(seed)
    rng = random.Random(seed + 1)
    rows, seen = [], set()
    for fam in FAMILIES:
        s = split[fam.name]
        made, attempts = 0, 0
        while made < per_family[s] and attempts < 60 * per_family[s]:  # a small family may run out of variety
            attempts += 1
            t = make_report(fam, rng)
            if t in seen:
                continue
            seen.add(t)
            rows.append({"text": t, "label": fam.label, "family": fam.name, "split": s})
            made += 1
    return pd.DataFrame(rows)
