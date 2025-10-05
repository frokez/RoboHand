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

def finger_curl_from_landmarks(lm2d, lm3d=None, use_composite=True, out='dict', use_world=True):
    # index
    i_mcp, i_pip, i_dip, i_tip = INDEX
    index_val = composite_nonthumb_curl(lm2d, lm3d, i_mcp, i_pip, i_dip, i_tip, use_world=use_world) \
                if use_composite else angle_3pt_nd(P(lm3d or lm2d, i_mcp, use_world),
                                                  P(lm3d or lm2d, i_pip, use_world),
                                                  P(lm3d or lm2d, i_tip, use_world))

    # middle
    m_mcp, m_pip, m_dip, m_tip = MIDDLE
    middle_val = composite_nonthumb_curl(lm2d, lm3d, m_mcp, m_pip, m_dip, m_tip, use_world=use_world) \
        if use_composite else angle_3pt_nd(P(lm3d or lm2d, m_mcp, use_world),
                                           P(lm3d or lm2d, m_pip, use_world),
                                           P(lm3d or lm2d, m_tip, use_world))

    # ring
    r_mcp, r_pip, r_dip, r_tip = RING
    ring_val = composite_nonthumb_curl(lm2d, lm3d, r_mcp, r_pip, r_dip, r_tip, use_world=use_world) \
        if use_composite else angle_3pt_nd(P(lm3d or lm2d, r_mcp, use_world),
                                           P(lm3d or lm2d, r_pip, use_world),
                                           P(lm3d or lm2d, r_tip, use_world))

    # pinky
    p_mcp, p_pip, p_dip, p_tip = PINKY
    pinky_val = composite_nonthumb_curl(lm2d, lm3d, p_mcp, p_pip, p_dip, p_tip, use_world=use_world) \
        if use_composite else angle_3pt_nd(P(lm3d or lm2d, p_mcp, use_world),
                                           P(lm3d or lm2d, p_pip, use_world),
                                           P(lm3d or lm2d, p_tip, use_world))

    # thumb
    t_cmc, t_mcp, t_ip, t_tip = THUMB
    thumb_val = composite_thumb_curl(lm2d, lm3d, t_cmc, t_mcp, t_ip, t_tip, use_world=use_world) \
                if use_composite else angle_3pt_nd(P(lm3d or lm2d, t_mcp, use_world),
                                                  P(lm3d or lm2d, t_ip,  use_world),
                                                  P(lm3d or lm2d, t_tip, use_world))

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

def composite_nonthumb_curl(lm2d, lm3d, mcp, pip, dip, tip, use_world = True,
                            w_pip=0.45, w_mcp=0.15, w_chain=0.25, w_dist=0.15):
    """Blend angles + distances; auto-fallback when angle geometry is unreliable."""
    # Angles
    lm = lm3d if (use_world and lm3d is not None) else lm2d
    a_wrs = P(lm, WRIST, use_world)
    a_mcp = P(lm, mcp,   use_world)
    a_pip = P(lm, pip,   use_world)
    a_tip = P(lm, tip,   use_world)

    pip_ang   = angle_3pt_nd(a_mcp, a_pip, a_tip)     
    mcp_ang   = angle_3pt_nd(a_wrs, a_mcp, a_pip)
    chain_ang = angle_3pt_nd(a_mcp, a_pip, a_tip)

    rel       = cross_sin_nd(a_mcp, a_pip, a_tip)


    d_tip_mcp = np.linalg.norm(P(lm, tip, use_world) - P(lm, mcp, use_world))
    d_w_mcp   = max(np.linalg.norm(P(lm, WRIST, use_world) - P(lm, mcp, use_world)), 1e-6)
    dist_ratio = d_tip_mcp / d_w_mcp
    curl_dist  = 1.0 - _clamp01(dist_ratio)

    # Fallback: if rel is low (<~0.2), downweight angles, upweight distance
    if rel < 0.2:
        w_pip, w_mcp, w_chain, w_dist = 0.15, 0.10, 0.15, 0.60

    return (w_pip*pip_ang + w_mcp*mcp_ang + w_chain*chain_ang + w_dist*curl_dist)

def composite_thumb_curl(lm2d, lm3d, cmc, mcp, ip, tip, use_world=True,
                         w_ip=0.5, w_mcp=0.2, w_chain=0.2, w_dist=0.1):
    lm = lm3d if (use_world and lm3d is not None) else lm2d

    a_cmc = P(lm, cmc, use_world)
    a_mcp = P(lm, mcp, use_world)
    a_ip  = P(lm, ip,  use_world)
    a_tip = P(lm, tip, use_world)
    a_wrs = P(lm, WRIST, use_world)

    # Angles
    ip_ang    = angle_3pt_nd(a_mcp, a_ip,  a_tip)
    mcp_ang   = angle_3pt_nd(a_cmc, a_mcp, a_ip)
    chain_ang = angle_3pt_nd(a_mcp, a_ip,  a_tip)

    # Reliability (edge-on → small)
    rel = cross_sin_nd(a_mcp, a_ip, a_tip)

    # Distance cue (curled → TIP closer to MCP)
    d_tip_mcp = np.linalg.norm(a_tip - a_mcp)
    d_w_mcp   = max(np.linalg.norm(a_wrs - a_mcp), 1e-6)
    dist_ratio = d_tip_mcp / d_w_mcp
    curl_dist  = 1.0 - _clamp01(dist_ratio)

    # Fallback weights when geometry is weak
    if rel < 0.2:
        w_ip, w_mcp, w_chain, w_dist = 0.2, 0.1, 0.2, 0.5

    return (w_ip*ip_ang + w_mcp*mcp_ang + w_chain*chain_ang + w_dist*curl_dist)

def angle_3pt_nd(a, b, c):
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32); c = np.asarray(c, np.float32)
    u = a - b; v = c - b
    nu = np.linalg.norm(u); nv = np.linalg.norm(v)
    if nu < 1e-6 or nv < 1e-6:
        return 0.0
    cosang = np.dot(u, v) / (nu * nv)
    cosang = np.clip(cosang, -1.0, 1.0)
    ang = np.arccos(cosang)
    return float(ang if np.isfinite(ang) else 0.0)

def cross_sin_nd(a, b, c):
    """sin(theta) at B; use cross magnitude/|u||v|. Works in 3D (true cross) and 2D (z-magnitude)."""
    a = np.asarray(a, np.float32); b = np.asarray(b, np.float32); c = np.asarray(c, np.float32)
    u = a - b; v = c - b
    nu = np.linalg.norm(u); nv = np.linalg.norm(v)
    if nu < 1e-6 or nv < 1e-6:
        return 0.0
    if u.shape[0] == 3:
        cross_mag = np.linalg.norm(np.cross(u, v))
    else:
        cross_mag = abs(u[0]*v[1] - u[1]*v[0])
    return float(cross_mag / (nu*nv))

def P(lm, idx, use_world):
    """Return point idx from lm in proper dimensionality."""
    if lm is None: return None
    return lm[idx, :3] if (use_world and lm.shape[1] >= 3) else lm[idx, :2]
