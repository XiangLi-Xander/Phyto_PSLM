"""
Traditional sequence features for ML classifiers (no transformer-input features).
Features: AAC, DPC, CTD, CKSAAP, physicochemical descriptors.
"""
import numpy as np

AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_INDEX = {aa: i for i, aa in enumerate(AA_LIST)}

# ============================================================================
# Physicochemical groups
# ============================================================================

HYDROPHOBIC = set("AILMFWV")
POS_CHARGED = set("RKH")
NEG_CHARGED = set("DE")
POLAR_UNCHARGED = set("QNST")
AROMATIC = set("FYW")
SMALL = set("GAS")
SPECIAL = set("CP")
TINY = set("AGCS")

GROUPS = {
    "hydrophobic": HYDROPHOBIC,
    "positive": POS_CHARGED,
    "negative": NEG_CHARGED,
    "polar_uncharged": POLAR_UNCHARGED,
    "aromatic": AROMATIC,
    "small": SMALL,
    "special": SPECIAL,
    "tiny": TINY,
}

# ============================================================================
# AAC (20-dim)
# ============================================================================

def aac(seq: str) -> np.ndarray:
    L = max(len(seq), 1)
    vec = np.zeros(20, dtype=np.float32)
    for aa in seq:
        if aa in AA_INDEX:
            vec[AA_INDEX[aa]] += 1
    return vec / L

# ============================================================================
# DPC (400-dim)
# ============================================================================

def dpc(seq: str) -> np.ndarray:
    L = len(seq)
    if L < 2:
        return np.zeros(400, dtype=np.float32)
    vec = np.zeros(400, dtype=np.float32)
    for i in range(L - 1):
        a, b = seq[i], seq[i + 1]
        if a in AA_INDEX and b in AA_INDEX:
            vec[AA_INDEX[a] * 20 + AA_INDEX[b]] += 1
    return vec / (L - 1)

# ============================================================================
# TPC – Tripeptide Composition (compressed: 8000→ use grouped)
# ============================================================================

def _aa_group(aa: str) -> int:
    if aa in HYDROPHOBIC: return 0
    if aa in POS_CHARGED: return 1
    if aa in NEG_CHARGED: return 2
    if aa in POLAR_UNCHARGED: return 3
    if aa in AROMATIC: return 4
    if aa in SMALL: return 5
    if aa in SPECIAL: return 6
    return 7  # tiny/other

def grouped_tpc(seq: str) -> np.ndarray:
    """Grouped tripeptide composition (8^3 = 512-dim)."""
    L = len(seq)
    if L < 3:
        return np.zeros(512, dtype=np.float32)
    vec = np.zeros(512, dtype=np.float32)
    for i in range(L - 2):
        g1 = _aa_group(seq[i])
        g2 = _aa_group(seq[i+1])
        g3 = _aa_group(seq[i+2])
        vec[g1 * 64 + g2 * 8 + g3] += 1
    return vec / (L - 2)

# ============================================================================
# CTD – Composition / Transition / Distribution (8 groups × 21 = 168-dim)
# ============================================================================

def _ctd_single(seq: str, group_set: set) -> np.ndarray:
    """CTD for one group."""
    L = max(len(seq), 1)
    mask = np.array([1 if aa in group_set else 0 for aa in seq], dtype=np.float32)
    n = max(mask.sum(), 1)

    # Composition
    c = n / L

    # Transition: number of 0→1 or 1→0 transitions / (L-1)
    if L > 1:
        t = np.sum(np.abs(np.diff(mask))) / (L - 1)
    else:
        t = 0.0

    # Distribution: positions at 1%, 25%, 50%, 75%, 100% of group residues
    positions = np.where(mask == 1)[0]
    if len(positions) > 0:
        pcts = np.percentile(positions, [0, 25, 50, 75, 100])
        d = pcts / L
    else:
        d = np.zeros(5, dtype=np.float32)

    return np.array([c, t] + d.tolist(), dtype=np.float32)


def ctd(seq: str) -> np.ndarray:
    """CTD features (8 groups × 7 = 56-dim)."""
    feats = []
    for gs in GROUPS.values():
        feats.append(_ctd_single(seq, gs))
    return np.concatenate(feats)

# ============================================================================
# CKSAAP – Composition of k-spaced amino acid pairs (k=0,1,2,3: 400×4=1600-dim)
# Use gap=0 (DPC) + gap=1,2,3 compressed
# ============================================================================

def cksaap(seq: str, max_gap: int = 3) -> np.ndarray:
    """CKSAAP compressed by summing k=0..max_gap into 400-dim."""
    L = len(seq)
    vec = np.zeros(400, dtype=np.float32)
    total = 0
    for gap in range(max_gap + 1):
        span = gap + 2
        for i in range(L - span):
            a, b = seq[i], seq[i + span - 1]
            if a in AA_INDEX and b in AA_INDEX:
                vec[AA_INDEX[a] * 20 + AA_INDEX[b]] += 1
                total += 1
    if total > 0:
        vec /= total
    return vec

# ============================================================================
# Sequence-level physicochemical descriptors
# ============================================================================

def physchem(seq: str) -> np.ndarray:
    """Physicochemical and motif descriptors (14-dim)."""
    L = max(len(seq), 1)
    pos = sum(seq.count(aa) for aa in "RKH")
    neg = sum(seq.count(aa) for aa in "DE")
    net_charge = (pos - neg) / L
    aromatic = sum(seq.count(aa) for aa in "FYW") / L
    hydrophobic = sum(seq.count(aa) for aa in "AILMFWV") / L
    small = sum(seq.count(aa) for aa in "GAS") / L
    proline = seq.count("P") / L
    glycine = seq.count("G") / L
    polar = sum(seq.count(aa) for aa in "QNST") / L
    rg = seq.count("RG") / L
    rgg = seq.count("RGG") / L
    sr = (seq.count("S") + seq.count("R")) / L
    rs = (seq.count("RS") + seq.count("SR")) / L
    # complexity: fraction of unique dipeptides
    dipep_set = set()
    for i in range(L - 1):
        dipep_set.add(seq[i:i+2])
    complexity = len(dipep_set) / max(400 * (1 - (399/400)**L), 1e-8)  # normalized
    return np.array([
        net_charge, aromatic, hydrophobic, small, proline, glycine, polar,
        rg, rgg, sr, rs, float(pos), float(neg), L / 1000.0,
    ], dtype=np.float32)


def extract_features(seqs: list) -> np.ndarray:
    """Extract traditional sequence features (no ESM2, no IUPred3)."""
    feats = []
    for seq in seqs:
        f = np.concatenate([
            aac(seq),           # 20
            dpc(seq),           # 400
            grouped_tpc(seq),   # 512
            ctd(seq),           # 56
            cksaap(seq),        # 400
            physchem(seq),      # 14
        ])
        feats.append(f)
    return np.array(feats, dtype=np.float32)
