import numpy as np

WRIST = 0
THUMB = (1,2,3,4)      # CMC, MCP, IP, TIP
INDEX = (5,6,7,8)      # MCP, PIP, DIP, TIP
MIDDLE = (9,10,11,12)
RING   = (13,14,15,16)
PINKY  = (17,18,19,20)
FINGER_ORDER = ('thumb','index','middle','ring','pinky')

def angle_3pt(a, b, c):
    a = np.asarray(a[:2], dtype=np.float32)
    b = np.asarray(b[:2], dtype=np.float32)
    c = np.asarray(c[:2], dtype=np.float32)

    u = a - b
    v = c - b
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < 1e-6 or nv < 1e-6:
        return 0.0

    cosang = np.dot(u, v) / (nu * nv)
    cosang = np.clip(cosang, -1.0, 1.0)
    ang = np.arccos(cosang)
    if not np.isfinite(ang):
        return 0.0
    return float(ang)

def finger_curl_from_landmarks(lm, use_blend=False, out='dict', use_composite=True):
    # index
    i_mcp, i_pip, i_dip, i_tip = INDEX
    if use_composite:
        index_val = composite_nonthumb_curl(lm, i_mcp, i_pip, i_dip, i_tip)
    else:
        index_val = blended_nonthumb_curl(lm, i_mcp, i_pip, i_dip) if use_blend \
            else angle_3pt(xy(lm, i_mcp), xy(lm, i_pip), xy(lm, i_tip))

    # middle
    m_mcp, m_pip, m_dip, m_tip = MIDDLE
    middle_val = composite_nonthumb_curl(lm, m_mcp, m_pip, m_dip, m_tip) if use_composite \
        else (blended_nonthumb_curl(lm, m_mcp, m_pip, m_dip) if use_blend
              else angle_3pt(xy(lm, m_mcp), xy(lm, m_pip), xy(lm, m_tip)))

    # ring
    r_mcp, r_pip, r_dip, r_tip = RING
    ring_val = composite_nonthumb_curl(lm, r_mcp, r_pip, r_dip, r_tip) if use_composite \
        else (blended_nonthumb_curl(lm, r_mcp, r_pip, r_dip) if use_blend
              else angle_3pt(xy(lm, r_mcp), xy(lm, r_pip), xy(lm, r_tip)))

    # pinky
    p_mcp, p_pip, p_dip, p_tip = PINKY
    pinky_val = composite_nonthumb_curl(lm, p_mcp, p_pip, p_dip, p_tip) if use_composite \
        else (blended_nonthumb_curl(lm, p_mcp, p_pip, p_dip) if use_blend
              else angle_3pt(xy(lm, p_mcp), xy(lm, p_pip), xy(lm, p_tip)))

    # thumb
    t_cmc, t_mcp, t_ip, t_tip = THUMB
    thumb_val = composite_thumb_curl(lm, t_cmc, t_mcp, t_ip, t_tip) if use_composite \
        else (0.7*angle_3pt(xy(lm, t_mcp), xy(lm, t_ip), xy(lm, t_tip))
              + 0.3*angle_3pt(xy(lm, t_cmc), xy(lm, t_mcp), xy(lm, t_ip)))

    return [thumb_val, index_val, middle_val, ring_val, pinky_val] if out=='list' else {
        'thumb':  float(thumb_val),
        'index':  float(index_val),
        'middle': float(middle_val),
        'ring':   float(ring_val),
        'pinky':  float(pinky_val),
    }

def rad2deg(x):
    return float(np.degrees(x))

def xy(lm, idx):
    # returns a 2-vector (x,y) for landmark idx
    return lm[idx, :2]

def blended_nonthumb_curl(lm, mcp, pip, dip, w_pip=0.7):
    pip_ang = angle_3pt(xy(lm, mcp), xy(lm, pip), xy(lm, dip))
    mcp_ang = angle_3pt(xy(lm, WRIST), xy(lm, mcp), xy(lm, pip))
    return w_pip * pip_ang + (1.0 - w_pip) * mcp_ang

def curls_list(lm):
    d = finger_curl_from_landmarks(lm)
    return [d[k] for k in FINGER_ORDER]

def normalize_curls_dict(raw_dict, cal):
    """
    raw_dict: {'thumb':rad, 'index':rad, ...}
    cal: {'thumb':{'min':..., 'max':...}, ...}
    returns list in order [T,I,M,R,P], each in 0..1 (clamped)
    """
    order = ('thumb','index','middle','ring','pinky')
    out = []
    for k in order:
        r = float(raw_dict[k])
        mn = float(cal[k]['min'])
        mx = float(cal[k]['max'])
        # protect against equal min/max
        if mx - mn < 1e-6:
            val = 0.0
        else:
            val = (r - mn) / (mx - mn)
        # clamp
        if val < 0.0: val = 0.0
        if val > 1.0: val = 1.0
        out.append(val)
    return out

def _safe_norm(v):
    n = np.linalg.norm(v)
    return n if n > 1e-6 else 1e-6

def _sin_of_angle(a, b, c):
    """Return sin(theta) at B using 2D cross magnitude; acts as reliability (0 bad … 1 good)."""
    a = np.asarray(a[:2], dtype=np.float32); b = np.asarray(b[:2], dtype=np.float32); c = np.asarray(c[:2], dtype=np.float32)
    u = a - b; v = c - b
    nu = _safe_norm(u); nv = _safe_norm(v)
    # 2D cross magnitude = |u_x v_y - u_y v_x|
    cross_mag = abs(u[0]*v[1] - u[1]*v[0])
    return float(cross_mag / (nu*nv))  # == sin(theta) in [0,1]

def _dist(lm, i, j):
    p = lm[i,:2] - lm[j,:2]
    return float(np.linalg.norm(p))

def _clamp01(x):
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

def composite_nonthumb_curl(lm, mcp, pip, dip, tip,
                            w_pip=0.45, w_mcp=0.15, w_chain=0.25, w_dist=0.15):
    """Blend angles + distances; auto-fallback when angle geometry is unreliable."""
    # Angles
    pip_ang   = angle_3pt(xy(lm, mcp), xy(lm, pip), xy(lm, tip))   # use TIP instead of DIP
    mcp_ang   = angle_3pt(xy(lm, WRIST), xy(lm, mcp), xy(lm, pip))
    chain_ang = angle_3pt(xy(lm, mcp),  xy(lm, pip), xy(lm, tip))  # (same tri as pip_ang but you can keep both; they’ll correlate)

    # Reliability of PIP geometry (edge-on → small)
    rel = _sin_of_angle(xy(lm, mcp), xy(lm, pip), xy(lm, tip))

    # Distance cue (bigger curl → TIP gets closer to MCP)
    d_tip_mcp = _dist(lm, tip, mcp)
    d_w_mcp   = _dist(lm, WRIST, mcp)
    dist_ratio = d_tip_mcp / _safe_norm([d_w_mcp])  # dimensionless
    curl_dist = 1.0 - _clamp01(dist_ratio)          # 0=open, 1=closed-ish

    # Fallback: if rel is low (<~0.2), downweight angles, upweight distance
    if rel < 0.2:
        w_pip, w_mcp, w_chain, w_dist = 0.15, 0.10, 0.15, 0.60

    # Composite raw "curl" (still in angle-ish units; we'll just treat it as raw for calibration)
    return (w_pip*pip_ang + w_mcp*mcp_ang + w_chain*chain_ang + w_dist*curl_dist)

def composite_thumb_curl(lm, cmc, mcp, ip, tip,
                         w_ip=0.5, w_mcp=0.2, w_chain=0.2, w_dist=0.1):
    ip_ang    = angle_3pt(xy(lm, mcp), xy(lm, ip),  xy(lm, tip))
    mcp_ang   = angle_3pt(xy(lm, cmc), xy(lm, mcp), xy(lm, ip))
    chain_ang = angle_3pt(xy(lm, mcp), xy(lm, ip),  xy(lm, tip))
    rel       = _sin_of_angle(xy(lm, mcp), xy(lm, ip), xy(lm, tip))

    d_tip_mcp = _dist(lm, tip, mcp)
    d_w_mcp   = _dist(lm, WRIST, mcp)
    dist_ratio = d_tip_mcp / _safe_norm([d_w_mcp])
    curl_dist  = 1.0 - _clamp01(dist_ratio)

    if rel < 0.2:
        w_ip, w_mcp, w_chain, w_dist = 0.2, 0.1, 0.2, 0.5

    return (w_ip*ip_ang + w_mcp*mcp_ang + w_chain*chain_ang + w_dist*curl_dist)
