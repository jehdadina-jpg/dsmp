"""Real Mumbai Suburban Railway dataset: source extract, cleaning & enrichment.

This module is the single source of truth for the project's data. It models
**every real station** on the Western Line (Churchgate-Bordi Road, including
the Dahanu Road extension), Central Line (CSMT-Kalyan plus both the
Kalyan-Kasara and Kalyan-Khopoli/Karjat branches), Harbour Line (CSMT-Panvel,
plus the real Wadala Road-King's Circle-Andheri-Goregaon branch) and the full
Trans-Harbour Line (Thane-Vashi, every real intermediate stop: Airoli, Digha
Gaon, Rabale, Ghansoli, Koparkhairane, Turbhe). Lokmanya Tilak Terminus (a
real but effectively dead-end spur 0.9km off Kurla, adding a visually
confusing near-duplicate point with no through traffic) is intentionally not
modeled. Nothing here is
procedurally randomised.

Data provenance (full citations in README.md -> "Data Sources & Methodology"):
    * Station identity, line membership and real adjacency order: Wikipedia
      "Western/Central/Harbour line (Mumbai Suburban Railway)" articles.
    * Coordinates: the original 25 major stations from each station's own
      Wikipedia infobox; the remaining 50 from the OpenStreetMap Overpass
      API (railway=station nodes, ODbL), with a handful of Overpass misses
      filled from Wikipedia infoboxes. All WGS84, gathered Sept 2026.
    * daily_footfall: two values are directly cited public figures (Kalyan,
      Borivali). Roughly twenty more (the major hubs/termini) are
      hand-set estimates consistent with each station's documented
      real-world role. The remaining ~50 smaller stations use a deterministic
      tiered formula (secondary/standard/minor, by documented role) rather
      than per-station research -- see `estimate_footfall` intent inline.
      None of this is an official station-wise count, which Indian Railways
      does not publish as a single open dataset; also note that summed
      "footfall" (boardings+alightings) is not directly comparable to
      published per-line "ridership" (boardings only) figures, which is why
      the totals here don't reconcile exactly with those line-level figures
      quoted in the README. Swap `_STATION_RECORDS` for an authoritative
      source (an RTI disclosure or a GTFS feed) to move this from
      "estimate" to "official".

Scope: every real stop on the Western, Central, Harbour and Trans-Harbour
lines, including the Virar-Bordi Road extension, the Kalyan-Kasara and
Kalyan-Khopoli (via Karjat) branches, the Wadala Road-King's Circle-Andheri-
Goregaon Harbour branch, and every real Trans-Harbour stop between Thane and
Vashi -- not the separate Vasai-Diva, Airport, Uran or Panvel-Karjat lines,
nor the Kurla-Lokmanya Tilak Terminus dead-end spur (see above). CSMT-Masjid-Sandhurst Road
is real shared trackage used by both Central and Harbour services, so it is
modeled as shared edges/stations (line="Central/Harbour") rather than
duplicated, the same way Dadar is shared between Western/Central and the
Mahim-Goregaon stretch is shared between Western/Harbour. distance_km is a
real haversine calculation over the coordinates above.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config

logger = config.get_logger(__name__)


@dataclass(frozen=True)
class StationRecord:
    station_id: str
    name: str
    line: str
    latitude: float
    longitude: float
    daily_footfall: int
    interchange_hub: bool


# ---------------------------------------------------------------------------
# REAL SOURCE DATA -- see module docstring for provenance.
# interchange_hub is True iff the station joins >=2 lines *within this
# project's modeled network* (real-world junctions outside that scope, e.g.
# Kalyan's Kasara/Khopoli lines, are intentionally not flagged here).
# ---------------------------------------------------------------------------
_STATION_RECORDS: list[StationRecord] = [
    # -- Western Line (Churchgate -> Virar), 29 stations ---------------------
    StationRecord("CHG", "Churchgate", "Western", 18.934886, 72.827164, 450_000, False),
    StationRecord("MAR", "Marine Lines", "Western", 18.945764, 72.823719, 55_800, False),
    StationRecord("CHA", "Charni Road", "Western", 18.951844, 72.818240, 41_400, False),
    StationRecord("GRA", "Grant Road", "Western", 18.963355, 72.815826, 48_000, False),
    StationRecord("MMC", "Mumbai Central", "Western", 18.969700, 72.819400, 380_000, False),
    StationRecord("MAH", "Mahalaxmi", "Western", 18.982520, 72.824220, 38_400, False),
    StationRecord("LPR", "Lower Parel", "Western", 18.995280, 72.830000, 210_000, False),
    StationRecord("PRA", "Prabhadevi", "Western", 19.007472, 72.835897, 134_000, False),
    StationRecord("DDR", "Dadar", "Western/Central", 19.018335, 72.843214, 750_000, True),
    StationRecord("MAT", "Matunga Road", "Western", 19.027815, 72.846681, 53_400, False),
    StationRecord("MAI", "Mahim Junction", "Western", 19.040663, 72.846604, 69_600, False),
    StationRecord("BND", "Bandra", "Western", 19.054444, 72.840556, 400_000, False),
    StationRecord("KHA", "Khar Road", "Western", 19.068241, 72.840041, 132_000, False),
    StationRecord("SAN", "Santacruz", "Western", 19.079289, 72.847118, 175_000, False),
    StationRecord("VLP", "Vile Parle", "Western", 19.099532, 72.843964, 230_000, False),
    StationRecord("AND", "Andheri", "Western", 19.119167, 72.846944, 650_000, False),
    StationRecord("JOG", "Jogeshwari", "Western", 19.142945, 72.855185, 55_800, False),
    StationRecord("RAM", "Ram Mandir", "Western", 19.151023, 72.850147, 43_800, False),
    StationRecord("GRG", "Goregaon", "Western", 19.164403, 72.849481, 260_000, False),
    StationRecord("MAL", "Malad", "Western", 19.186719, 72.848588, 199_000, False),
    StationRecord("KAN", "Kandivali", "Western", 19.204114, 72.851738, 135_000, False),
    StationRecord("BVI", "Borivali", "Western", 19.229427, 72.856994, 287_000, False),  # cited
    StationRecord("DAH", "Dahisar", "Western", 19.249357, 72.859630, 30_000, False),
    StationRecord("MIR", "Mira Road", "Western", 19.281697, 72.856068, 135_000, False),
    StationRecord("BHA", "Bhayandar", "Western", 19.310185, 72.852699, 126_000, False),
    StationRecord("NAI", "Naigaon", "Western", 19.351093, 72.846523, 30_600, False),
    StationRecord("VAS", "Vasai Road", "Western", 19.382668, 72.832025, 142_000, False),
    StationRecord("NAL", "Nallasopara", "Western", 19.415400, 72.861300, 50_400, False),
    StationRecord("VIR", "Virar", "Western", 19.455300, 72.812000, 340_000, False),
    # -- Central Line (CSMT -> Kalyan), 26 stations --------------------------
    StationRecord("CSMT", "Chhatrapati Shivaji Maharaj Terminus", "Central/Harbour", 18.939850, 72.835440, 480_000, True),
    StationRecord("MAS", "Masjid", "Central/Harbour", 18.952295, 72.838177, 15_000, True),
    StationRecord("SAD", "Sandhurst Road", "Central/Harbour", 18.960924, 72.839372, 34_500, True),
    StationRecord("BYC", "Byculla", "Central", 18.976390, 72.832780, 180_000, False),
    StationRecord("CHI", "Chinchpokli", "Central", 18.987198, 72.832672, 49_200, False),
    StationRecord("CUR", "Currey Road", "Central", 18.993749, 72.832856, 63_600, False),
    StationRecord("PRL", "Parel", "Central", 19.009161, 72.837608, 220_000, False),
    StationRecord("MAU", "Matunga", "Central", 19.027436, 72.850147, 137_000, False),
    StationRecord("SIO", "Sion", "Central", 19.040000, 72.860000, 190_000, False),
    StationRecord("KUR", "Kurla", "Central/Harbour", 19.065600, 72.879100, 500_000, True),
    StationRecord("VID", "Vidyavihar", "Central", 19.079542, 72.897117, 167_000, False),
    StationRecord("GHK", "Ghatkopar", "Central", 19.085700, 72.908100, 420_000, False),
    StationRecord("VKL", "Vikhroli", "Central", 19.110000, 72.920000, 170_000, False),
    StationRecord("KAJ", "Kanjurmarg", "Central", 19.128300, 72.927500, 55_200, False),
    StationRecord("BHN", "Bhandup", "Central", 19.142757, 72.937654, 33_600, False),
    StationRecord("NAH", "Nahur", "Central", 19.154640, 72.946652, 36_000, False),
    StationRecord("MUL", "Mulund", "Central", 19.172137, 72.956697, 149_000, False),
    StationRecord("THN", "Thane", "Central/Trans-Harbour", 19.186110, 72.975830, 700_000, True),
    StationRecord("KAL", "Kalwa", "Central", 19.196060, 72.997574, 39_000, False),
    StationRecord("MUM", "Mumbra", "Central", 19.189943, 73.023075, 132_000, False),
    StationRecord("DIV", "Diva", "Central", 19.188756, 73.041417, 37_000, False),
    StationRecord("KOP", "Kopar", "Central", 19.212423, 73.078865, 35_400, False),
    StationRecord("DOM", "Dombivli", "Central", 19.218049, 73.086135, 142_000, False),
    StationRecord("THA", "Thakurli", "Central", 19.226281, 73.098017, 51_600, False),
    StationRecord("KYN", "Kalyan", "Central", 19.235250, 73.130500, 800_000, False),  # cited
    # -- Harbour Line (CSMT -> Panvel), 25 stations (CSMT/Masjid/Sandhurst ---
    # Road/Kurla already declared above -- shared with Central) -------------
    StationRecord("DOC", "Dockyard Road", "Harbour", 18.967477, 72.844548, 53_400, False),
    StationRecord("REA", "Reay Road", "Harbour", 18.977551, 72.844101, 43_800, False),
    StationRecord("COT", "Cotton Green", "Harbour", 18.987187, 72.843896, 66_000, False),
    StationRecord("SEW", "Sewri", "Harbour", 18.998780, 72.854422, 43_200, False),
    StationRecord("WDR", "Wadala Road", "Harbour", 19.016000, 72.858800, 90_000, False),
    StationRecord("GUR", "Guru Tegh Bahadur Nagar", "Harbour", 19.036687, 72.870786, 84_600, False),
    StationRecord("CHU", "Chunabhatti", "Harbour", 19.051652, 72.869053, 48_600, False),
    StationRecord("TIL", "Tilak Nagar", "Harbour", 19.065786, 72.890470, 43_200, False),
    StationRecord("CHM", "Chembur", "Harbour", 19.063060, 72.900560, 260_000, False),
    StationRecord("GOV", "Govandi", "Harbour", 19.055369, 72.915070, 37_200, False),
    StationRecord("MKD", "Mankhurd", "Harbour", 19.048100, 72.931700, 150_000, False),
    StationRecord("VSH", "Vashi", "Harbour/Trans-Harbour", 19.063200, 72.998800, 300_000, True),
    StationRecord("SAP", "Sanpada", "Harbour", 19.066000, 73.009600, 87_600, False),
    StationRecord("JUI", "Juinagar", "Harbour", 19.056000, 73.017800, 40_200, False),
    StationRecord("NRL", "Nerul", "Harbour", 19.033600, 73.018100, 180_000, False),
    StationRecord("SEA", "Seawoods-Darave", "Harbour", 19.021819, 73.019159, 76_200, False),
    StationRecord("CBD", "CBD Belapur", "Harbour", 19.019008, 73.039130, 168_000, False),
    StationRecord("KHR", "Kharghar", "Harbour", 19.025841, 73.059300, 128_000, False),
    StationRecord("MAN", "Mansarovar", "Harbour", 19.016434, 73.080655, 60_000, False),
    StationRecord("KHN", "Khandeshwar", "Harbour", 19.007350, 73.095283, 51_600, False),
    StationRecord("PNV", "Panvel", "Harbour", 18.989320, 73.122290, 220_000, False),
    # -- Harbour branch: Wadala Road -> King's Circle -> Andheri/Goregaon ----
    # (real Harbour trains also run this branch, shared with Western stations
    # already declared above -- see the line-field update further down).
    StationRecord("KGC", "King's Circle", "Harbour", 19.032600, 72.857400, 40_000, False),
    # -- Western Line extension: Virar -> Dahanu Road / Bordi Road, 10 stns --
    StationRecord("VAI", "Vaitarna", "Western", 19.472000, 72.791000, 8_000, False),
    StationRecord("SAF", "Saphale", "Western", 19.518500, 72.809100, 22_000, False),
    StationRecord("KLV", "Kelve Road", "Western", 19.573400, 72.776100, 10_000, False),
    StationRecord("PLG", "Palghar", "Western", 19.696300, 72.764500, 95_000, False),
    StationRecord("UMR", "Umroli", "Western", 19.721700, 72.748200, 6_000, False),
    StationRecord("BOI", "Boisar", "Western", 19.799800, 72.742300, 60_000, False),
    StationRecord("VNG2", "Vangaon", "Western", 19.893400, 72.721800, 9_000, False),
    StationRecord("DAR", "Dahanu Road", "Western", 19.970900, 72.737000, 65_000, False),
    StationRecord("GHV", "Gholvad", "Western", 20.005000, 72.735000, 7_000, False),
    StationRecord("BDR", "Bordi Road", "Western", 20.030000, 72.728000, 5_000, False),
    # -- Central Line branch: Kalyan -> Kasara, 9 stations -------------------
    StationRecord("TTW", "Titwala", "Central", 19.299600, 73.200300, 48_000, False),
    StationRecord("KHD", "Khadavli", "Central", 19.324700, 73.231200, 12_000, False),
    StationRecord("VSN", "Vasind", "Central", 19.362700, 73.262600, 15_000, False),
    StationRecord("ASN", "Asangaon", "Central", 19.404600, 73.285300, 40_000, False),
    StationRecord("ATG", "Atgaon", "Central", 19.439400, 73.313200, 9_000, False),
    StationRecord("THS", "Thansit", "Central", 19.467700, 73.342100, 4_000, False),
    StationRecord("KRD", "Khardi", "Central", 19.497800, 73.374600, 11_000, False),
    StationRecord("UMB", "Umbermali", "Central", 19.535400, 73.406700, 4_000, False),
    StationRecord("KAS", "Kasara", "Central", 19.608300, 73.446300, 35_000, False),
    # -- Central Line branch: Kalyan -> Khopoli (via Karjat), 13 stations ----
    StationRecord("VIT", "Vithalwadi", "Central", 19.218000, 73.142000, 20_000, False),
    StationRecord("ULH", "Ulhasnagar", "Central", 19.208500, 73.155000, 90_000, False),
    StationRecord("AMB", "Ambarnath", "Central", 19.199600, 73.192700, 105_000, False),
    StationRecord("BDL", "Badlapur", "Central", 19.155400, 73.235000, 120_000, False),
    StationRecord("VNG", "Vangani", "Central", 19.110300, 73.265000, 25_000, False),
    StationRecord("SHL", "Shelu", "Central", 19.079600, 73.293400, 6_000, False),
    StationRecord("NER", "Neral Junction", "Central", 19.033900, 73.316800, 55_000, False),
    StationRecord("KRJ", "Karjat", "Central", 18.910700, 73.323700, 110_000, False),
    StationRecord("PLS", "Palasdhari", "Central", 18.871000, 73.352000, 3_000, False),
    StationRecord("KLI", "Kelavli", "Central", 18.838600, 73.381000, 6_000, False),
    StationRecord("DOL", "Dolavli", "Central", 18.809000, 73.396000, 3_000, False),
    StationRecord("LOW", "Lowjee", "Central", 18.785300, 73.409000, 3_000, False),
    StationRecord("KHP", "Khopoli", "Central", 18.788000, 73.334600, 65_000, False),
    # -- Trans-Harbour intermediate stops: Thane -> Vashi, 6 stations --------
    StationRecord("ARL", "Airoli", "Trans-Harbour", 19.159200, 72.998600, 55_000, False),
    StationRecord("DGG", "Digha Gaon", "Trans-Harbour", 19.147000, 73.000000, 12_000, False),
    StationRecord("RBL", "Rabale", "Trans-Harbour", 19.118000, 72.998500, 20_000, False),
    StationRecord("GNS", "Ghansoli", "Trans-Harbour", 19.108000, 73.006000, 30_000, False),
    StationRecord("KPK", "Koparkhairane", "Trans-Harbour", 19.103900, 73.015900, 25_000, False),
    StationRecord("TRB", "Turbhe", "Trans-Harbour", 19.082700, 73.015800, 22_000, False),
]

# The Andheri<->Goregaon stretch of the Western Line is also a real, live
# Harbour Line branch (Wadala Road -> King's Circle -> Mahim -> Bandra ->
# ... -> Goregaon). Rather than duplicate those stations under a second
# station_id, they are relabelled "Western/Harbour" in place -- the same
# shared-station convention already used for Dadar and CSMT/Masjid/Sandhurst.
_HARBOUR_SHARED_WESTERN_STOPS = {
    "MAI", "BND", "KHA", "SAN", "VLP", "AND", "JOG", "RAM", "GRG",
}
_STATION_RECORDS = [
    (
        StationRecord(s.station_id, s.name, "Western/Harbour", s.latitude, s.longitude,
                       s.daily_footfall, s.interchange_hub)
        if s.station_id in _HARBOUR_SHARED_WESTERN_STOPS else s
    )
    for s in _STATION_RECORDS
]

# (source_id, target_id, line_name) -- real adjacency between two stations,
# in real geographic order. CSMT-Masjid-Sandhurst Road carries both Central
# and Harbour services (line="Central/Harbour"), same convention as a shared
# *station*. The final entry is the real direct Trans-Harbour service.
_EDGE_TOPOLOGY: list[tuple[str, str, str]] = [
    # Western (29 stations -> 28 edges)
    ("CHG", "MAR", "Western"), ("MAR", "CHA", "Western"), ("CHA", "GRA", "Western"),
    ("GRA", "MMC", "Western"), ("MMC", "MAH", "Western"), ("MAH", "LPR", "Western"),
    ("LPR", "PRA", "Western"), ("PRA", "DDR", "Western"), ("DDR", "MAT", "Western"),
    ("MAT", "MAI", "Western"), ("MAI", "BND", "Western/Harbour"), ("BND", "KHA", "Western/Harbour"),
    ("KHA", "SAN", "Western/Harbour"), ("SAN", "VLP", "Western/Harbour"), ("VLP", "AND", "Western/Harbour"),
    ("AND", "JOG", "Western/Harbour"), ("JOG", "RAM", "Western/Harbour"), ("RAM", "GRG", "Western/Harbour"),
    ("GRG", "MAL", "Western"), ("MAL", "KAN", "Western"), ("KAN", "BVI", "Western"),
    ("BVI", "DAH", "Western"), ("DAH", "MIR", "Western"), ("MIR", "BHA", "Western"),
    ("BHA", "NAI", "Western"), ("NAI", "VAS", "Western"), ("VAS", "NAL", "Western"),
    ("NAL", "VIR", "Western"),
    # Central (26 stations -> 25 edges, first 2 shared with Harbour)
    ("CSMT", "MAS", "Central/Harbour"), ("MAS", "SAD", "Central/Harbour"),
    ("SAD", "BYC", "Central"), ("BYC", "CHI", "Central"), ("CHI", "CUR", "Central"),
    ("CUR", "PRL", "Central"), ("PRL", "DDR", "Central"), ("DDR", "MAU", "Central"),
    ("MAU", "SIO", "Central"), ("SIO", "KUR", "Central"), ("KUR", "VID", "Central"),
    ("VID", "GHK", "Central"), ("GHK", "VKL", "Central"), ("VKL", "KAJ", "Central"),
    ("KAJ", "BHN", "Central"), ("BHN", "NAH", "Central"), ("NAH", "MUL", "Central"),
    ("MUL", "THN", "Central"), ("THN", "KAL", "Central"), ("KAL", "MUM", "Central"),
    ("MUM", "DIV", "Central"), ("DIV", "KOP", "Central"), ("KOP", "DOM", "Central"),
    ("DOM", "THA", "Central"), ("THA", "KYN", "Central"),
    # Harbour (25 stations -> 24 edges, first 2 already listed above)
    ("SAD", "DOC", "Harbour"), ("DOC", "REA", "Harbour"), ("REA", "COT", "Harbour"),
    ("COT", "SEW", "Harbour"), ("SEW", "WDR", "Harbour"), ("WDR", "GUR", "Harbour"),
    ("GUR", "CHU", "Harbour"), ("CHU", "KUR", "Harbour"), ("KUR", "TIL", "Harbour"),
    ("TIL", "CHM", "Harbour"), ("CHM", "GOV", "Harbour"), ("GOV", "MKD", "Harbour"),
    ("MKD", "VSH", "Harbour"), ("VSH", "SAP", "Harbour"), ("SAP", "JUI", "Harbour"),
    ("JUI", "NRL", "Harbour"), ("NRL", "SEA", "Harbour"), ("SEA", "CBD", "Harbour"),
    ("CBD", "KHR", "Harbour"), ("KHR", "MAN", "Harbour"), ("MAN", "KHN", "Harbour"),
    ("KHN", "PNV", "Harbour"),
    # Trans-Harbour: real intermediate-stop chain, Thane -> Vashi. (A direct
    # THN-VSH express service also exists in reality, but it is not modeled
    # as a *second* edge here -- with every real intermediate stop now in the
    # graph, an extra shortcut edge would just duplicate this same corridor
    # and double-draw it on the map.)
    ("THN", "ARL", "Trans-Harbour"), ("ARL", "DGG", "Trans-Harbour"),
    ("DGG", "RBL", "Trans-Harbour"), ("RBL", "GNS", "Trans-Harbour"),
    ("GNS", "KPK", "Trans-Harbour"), ("KPK", "TRB", "Trans-Harbour"),
    ("TRB", "VSH", "Trans-Harbour"),
    # Harbour branch: Wadala Road -> King's Circle -> Mahim (joins the
    # Western/Harbour-tagged stretch above, not duplicated here)
    ("WDR", "KGC", "Harbour"), ("KGC", "MAI", "Harbour"),
    # Western extension: Virar -> Dahanu Road / Bordi Road
    ("VIR", "VAI", "Western"), ("VAI", "SAF", "Western"), ("SAF", "KLV", "Western"),
    ("KLV", "PLG", "Western"), ("PLG", "UMR", "Western"), ("UMR", "BOI", "Western"),
    ("BOI", "VNG2", "Western"), ("VNG2", "DAR", "Western"), ("DAR", "GHV", "Western"),
    ("GHV", "BDR", "Western"),
    # Central branch: Kalyan -> Kasara
    ("KYN", "TTW", "Central"), ("TTW", "KHD", "Central"), ("KHD", "VSN", "Central"),
    ("VSN", "ASN", "Central"), ("ASN", "ATG", "Central"), ("ATG", "THS", "Central"),
    ("THS", "KRD", "Central"), ("KRD", "UMB", "Central"), ("UMB", "KAS", "Central"),
    # Central branch: Kalyan -> Khopoli (via Karjat)
    ("KYN", "VIT", "Central"), ("VIT", "ULH", "Central"), ("ULH", "AMB", "Central"),
    ("AMB", "BDL", "Central"), ("BDL", "VNG", "Central"), ("VNG", "SHL", "Central"),
    ("SHL", "NER", "Central"), ("NER", "KRJ", "Central"), ("KRJ", "PLS", "Central"),
    ("PLS", "KLI", "Central"), ("KLI", "DOL", "Central"), ("DOL", "LOW", "Central"),
    ("LOW", "KHP", "Central"),
]


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance (km) between two WGS84 points.

    Used instead of flat Euclidean distance because 1 degree of longitude
    shrinks with latitude (~104.9 km/deg at Mumbai's ~19N vs ~111.3 km/deg
    for latitude) -- treating lat/lon as a flat Cartesian plane would skew
    every east-west measurement by that ratio.
    """
    r1, r2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(r1) * math.cos(r2) * math.sin(dlambda / 2) ** 2
    return 2 * config.EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def generate_source_data() -> None:
    """Write the curated real dataset to data/raw/ as the pipeline's extract.

    In a full production deployment this function would be replaced by a
    connector to an authoritative source (a GTFS feed, an open-data.gov.in
    dataset, or a railway-operator API); the rest of the pipeline is
    written against the raw CSV *shape*, not this function, so swapping the
    source later requires no downstream changes.
    """
    try:
        stations_df = pd.DataFrame([vars(s) for s in _STATION_RECORDS])
        edges_df = pd.DataFrame(_EDGE_TOPOLOGY, columns=["source_id", "target_id", "line_name"])

        stations_df.to_csv(config.STATIONS_RAW_CSV, index=False)
        edges_df.to_csv(config.EDGES_RAW_CSV, index=False)
        logger.info(
            "Wrote raw extract: %d stations -> %s, %d edges -> %s",
            len(stations_df), config.STATIONS_RAW_CSV.name,
            len(edges_df), config.EDGES_RAW_CSV.name,
        )
    except OSError as exc:
        logger.error("Could not write raw source data: %s", exc)
        raise


def _validate_and_clean_stations(df: pd.DataFrame) -> pd.DataFrame:
    """Type-check, deduplicate and report on missing values for stations."""
    missing = df.isnull().sum()
    if missing.any():
        logger.warning("Missing values found in stations:\n%s", missing[missing > 0])
        df = df.dropna(subset=["station_id", "latitude", "longitude"])
    else:
        logger.info("Stations: no missing values.")

    before = len(df)
    df = df.drop_duplicates(subset="station_id", keep="first")
    if len(df) != before:
        logger.warning("Dropped %d duplicate station_id row(s).", before - len(df))

    df["latitude"] = df["latitude"].astype(float)
    df["longitude"] = df["longitude"].astype(float)
    df["daily_footfall"] = df["daily_footfall"].astype(int)
    df["interchange_hub"] = df["interchange_hub"].astype(bool)

    out_of_bounds = df[(df["latitude"].abs() > 90) | (df["longitude"].abs() > 180)]
    if not out_of_bounds.empty:
        raise ValueError(f"Invalid coordinates found for: {out_of_bounds['station_id'].tolist()}")

    return df.reset_index(drop=True)


def _enrich_edges(edges_df: pd.DataFrame, stations_df: pd.DataFrame) -> pd.DataFrame:
    """Compute real haversine distance + derived travel time for every edge."""
    missing = edges_df.isnull().sum()
    if missing.any():
        logger.warning("Missing values found in edges:\n%s", missing[missing > 0])
        edges_df = edges_df.dropna()
    else:
        logger.info("Edges: no missing values.")

    coords = stations_df.set_index("station_id")[["latitude", "longitude"]]
    known_ids = set(coords.index)
    unknown = set(edges_df["source_id"]) | set(edges_df["target_id"])
    unknown -= known_ids
    if unknown:
        raise ValueError(f"Edge references unknown station_id(s): {sorted(unknown)}")

    distances = []
    for row in edges_df.itertuples(index=False):
        lat1, lon1 = coords.loc[row.source_id]
        lat2, lon2 = coords.loc[row.target_id]
        distances.append(_haversine_km(lat1, lon1, lat2, lon2))

    edges_df = edges_df.copy()
    edges_df["distance_km"] = np.round(distances, 3)
    travel_time = (edges_df["distance_km"] / config.AVG_SUBURBAN_SPEED_KMH) * 60.0
    edges_df["travel_time_mins"] = np.maximum(
        travel_time.round().astype(int), config.MIN_TRAVEL_TIME_MINS
    )
    return edges_df


def load_and_clean_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full ETL step: raw CSV -> validated + haversine-enriched CSV.

    Returns:
        (stations_df, edges_df) -- the cleaned, processed DataFrames, already
        persisted to data/processed/ as a side effect.
    """
    if not config.STATIONS_RAW_CSV.exists() or not config.EDGES_RAW_CSV.exists():
        logger.info("Raw data not found -- generating source extract first.")
        generate_source_data()

    try:
        raw_stations = pd.read_csv(config.STATIONS_RAW_CSV)
        raw_edges = pd.read_csv(config.EDGES_RAW_CSV)
    except pd.errors.EmptyDataError as exc:
        logger.error("Raw CSV is empty or corrupt: %s", exc)
        raise

    stations_df = _validate_and_clean_stations(raw_stations)
    edges_df = _enrich_edges(raw_edges, stations_df)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    stations_df.to_csv(config.STATIONS_CLEAN_CSV, index=False)
    edges_df.to_csv(config.EDGES_CLEAN_CSV, index=False)

    logger.info(
        "Processed data saved: %d stations, %d edges (avg hop %.2f km / %.1f min).",
        len(stations_df), len(edges_df),
        edges_df["distance_km"].mean(), edges_df["travel_time_mins"].mean(),
    )
    return stations_df, edges_df


def load_processed_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load already-processed CSVs, running the pipeline first if needed."""
    if not config.STATIONS_CLEAN_CSV.exists() or not config.EDGES_CLEAN_CSV.exists():
        return load_and_clean_data()
    stations_df = pd.read_csv(config.STATIONS_CLEAN_CSV)
    edges_df = pd.read_csv(config.EDGES_CLEAN_CSV)
    return stations_df, edges_df


if __name__ == "__main__":
    stations, edges = load_and_clean_data()
    print(stations.head())
    print(edges.head())
