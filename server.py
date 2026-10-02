import os
import sys
import json
import time
import base64
import urllib.parse
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes, label, convolve, distance_transform_edt
from scipy.interpolate import splprep, splev
from skimage.morphology import skeletonize, medial_axis

PORT = int(os.environ.get("PORT", 8080))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LABELED_DIR = os.path.join(BASE_DIR, "labeled-dataset")

# Definições de cores (BGR) da máscara anotada
COLOR_HYP  = np.array([15, 250, 20])   # Verde (Hipocótilo)
COLOR_ROOT = np.array([5, 10, 245])    # Vermelho (Raiz)
COLOR_COT  = np.array([255, 25, 30])   # Azul (Cotilédone)
KERNEL_3X3 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)

def extract_organ_masks(mask_bgr, tol=12):
    match = lambda c: np.all(np.abs(mask_bgr.astype(int) - c.astype(int)) <= tol, axis=-1)
    return match(COLOR_HYP), match(COLOR_ROOT), match(COLOR_COT)

def preprocess_mask(m):
    return binary_fill_holes(m).astype(bool)

def filter_small_components(m, min_size=30):
    labeled, n = label(m)
    if n == 0:
        return m
    sizes = np.bincount(labeled.ravel())
    keep = [i for i in range(1, n + 1) if sizes[i] >= min_size]
    return np.isin(labeled, keep) if keep else m

def prune_skeleton_by_dt(skeleton, dist, min_length=5, min_dt_ratio=0.5):
    skel = skeleton.copy()
    max_dt = dist[skel].max() if skel.any() else 1.0
    changed = True
    while changed:
        changed = False
        coords = set(map(tuple, np.argwhere(skel)))
        def neighbors(p):
            y, x = p
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    q = (y + dy, x + dx)
                    if q in coords:
                        yield q
        endpoints = [p for p in coords if sum(1 for _ in neighbors(p)) == 1]
        for ep in endpoints:
            branch = [ep]
            visited = {ep}
            current = ep
            while True:
                nbrs = [q for q in neighbors(current) if q not in visited]
                if len(nbrs) != 1:
                    break
                current = nbrs[0]
                visited.add(current)
                deg = sum(1 for _ in neighbors(current))
                branch.append(current)
                if deg > 2:
                    break
            if len(branch) < min_length:
                dt_values = [dist[p] for p in branch]
                if np.mean(dt_values) < min_dt_ratio * max_dt:
                    for p in branch:
                        skel[p] = False
                    changed = True
    return skel

def compute_skeleton_chain(skeleton, dist):
    coords = set(map(tuple, np.argwhere(skeleton)))
    def neighbors(p):
        y, x = p
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                q = (y + dy, x + dx)
                if q in coords:
                    yield q
    visited_global = set()
    chains = []
    for seed in coords:
        if seed in visited_global:
            continue
        component = set()
        stack = [seed]
        while stack:
            p = stack.pop()
            if p in component:
                continue
            component.add(p)
            for q in neighbors(p):
                if q not in component:
                    stack.append(q)
        visited_global |= component
        sub = component
        def nbrs_sub(p):
            y, x = p
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    q = (y + dy, x + dx)
                    if q in sub:
                        yield q
        endpoints = [p for p in sub if sum(1 for _ in nbrs_sub(p)) == 1]
        if not endpoints:
            endpoints = [next(iter(sub))]
        centroid = np.mean(list(sub), axis=0)
        start = max(endpoints, key=lambda p: np.linalg.norm(np.array(p) - centroid))
        visited = {start}
        chain = [start]
        current = start
        while True:
            next_pixels = [q for q in nbrs_sub(current) if q not in visited]
            if not next_pixels:
                break
            nxt = max(next_pixels, key=lambda q: dist[q])
            visited.add(nxt)
            chain.append(nxt)
            current = nxt
        if len(chain) >= 2:
            chains.append(chain)
    return chains

def chain_length_spline(chain, n_points=500):
    if len(chain) < 4:
        length = 0.0
        for i in range(1, len(chain)):
            dy = chain[i][0] - chain[i-1][0]
            dx = chain[i][1] - chain[i-1][1]
            length += np.hypot(dy, dx)
        return length
    pts = np.array(chain)
    try:
        tck, _ = splprep([pts[:, 0], pts[:, 1]], s=0, k=3)
        u = np.linspace(0, 1, n_points)
        y_s, x_s = splev(u, tck)
        diffs = np.diff(np.column_stack([y_s, x_s]), axis=0)
        return float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))
    except Exception:
        length = 0.0
        for i in range(1, len(chain)):
            dy = chain[i][0] - chain[i-1][0]
            dx = chain[i][1] - chain[i-1][1]
            length += np.hypot(dy, dx)
        return float(length)

def get_topology(skel_binary):
    sk_u8 = skel_binary.astype(np.uint8)
    neighbors = convolve(sk_u8, KERNEL_3X3, mode="constant", cval=0) * sk_u8
    endpoints = int((neighbors == 1).sum())
    branchpoints = int((neighbors > 2).sum())
    pixels = int(skel_binary.sum())
    return pixels, endpoints, branchpoints

def create_rgb_overlay(skel_h, skel_r, mask_h, mask_r):
    vis = np.zeros((*mask_h.shape, 3), dtype=np.uint8)
    # Background silhouettes (RGB format for browser display)
    vis[mask_h] = (0, 120, 0)
    vis[mask_r] = (120, 0, 0)
    # 1 px skeletons
    vis[skel_h] = (255, 255, 0)   # Yellow
    vis[skel_r] = (0, 255, 255)   # Cyan
    return vis

def create_organ_overlay(skel, mask, bg_color_rgb, skel_color_rgb):
    vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
    vis[mask] = bg_color_rgb
    vis[skel] = skel_color_rgb
    return vis

def to_base64_png(rgb_image):
    bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".png", bgr)
    return "data:image/png;base64," + base64.b64encode(buf).decode("utf-8")

def process_image(mask_bgr, pixels_per_cm=100.0, min_comp_size=30):
    mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
    mask_hyp_raw, mask_root_raw, mask_cot_raw = extract_organ_masks(mask_bgr)
    
    # Preprocessing and cleanup
    mask_hyp_clean  = filter_small_components(preprocess_mask(mask_hyp_raw),  min_comp_size)
    mask_root_clean = filter_small_components(preprocess_mask(mask_root_raw), min_comp_size)
    
    _, n_hyp_comp = label(mask_hyp_clean)
    _, n_root_comp = label(mask_root_clean)
    
    # 1. Zhang-Suen com Poda a Distância (EDT) + B-splines
    # NOTA: O esqueleto inicial é gerado por skimage.morphology.skeletonize (Zhang-Suen),
    # filtrado via Transformada de Distância Euclidiana (EDT) e interpolado por B-splines.
    t0 = time.time()
    dist_h = distance_transform_edt(mask_hyp_clean)
    dist_r = distance_transform_edt(mask_root_clean)
    
    sk_h_init = skeletonize(mask_hyp_clean)
    sk_r_init = skeletonize(mask_root_clean)
    
    sk_hyp_poda  = prune_skeleton_by_dt(sk_h_init, dist_h, min_length=5, min_dt_ratio=0.5)
    sk_root_poda = prune_skeleton_by_dt(sk_r_init, dist_r, min_length=5, min_dt_ratio=0.5)
    
    chains_h = compute_skeleton_chain(sk_hyp_poda, dist_h)
    chains_r = compute_skeleton_chain(sk_root_poda, dist_r)
    
    len_px_h = sum(chain_length_spline(c) for c in chains_h if len(c) >= 2)
    len_px_r = sum(chain_length_spline(c) for c in chains_r if len(c) >= 2)
    t_poda = (time.time() - t0) * 1000
    
    px_h_p, ep_h_p, bp_h_p = get_topology(sk_hyp_poda)
    px_r_p, ep_r_p, bp_r_p = get_topology(sk_root_poda)
    
    # 2. Zhang-Suen Puro (1984)
    t0 = time.time()
    sk_hyp_zh  = skeletonize(mask_hyp_raw, method="zhang")
    sk_root_zh = skeletonize(mask_root_raw, method="zhang")
    t_zh = (time.time() - t0) * 1000
    px_h_z, ep_h_z, bp_h_z = get_topology(sk_hyp_zh)
    px_r_z, ep_r_z, bp_r_z = get_topology(sk_root_zh)
    
    # 3. Lee & Kashyap Puro (1994)
    t0 = time.time()
    sk_hyp_lee  = skeletonize(mask_hyp_raw, method="lee")
    sk_root_lee = skeletonize(mask_root_raw, method="lee")
    t_lee = (time.time() - t0) * 1000
    px_h_l, ep_h_l, bp_h_l = get_topology(sk_hyp_lee)
    px_r_l, ep_r_l, bp_r_l = get_topology(sk_root_lee)
    
    # 4. MAT Puro (Blum 1967)
    t0 = time.time()
    sk_hyp_mat, _  = medial_axis(mask_hyp_raw, return_distance=True)
    sk_root_mat, _ = medial_axis(mask_root_raw, return_distance=True)
    t_mat = (time.time() - t0) * 1000
    px_h_m, ep_h_m, bp_h_m = get_topology(sk_hyp_mat)
    px_r_m, ep_r_m, bp_r_m = get_topology(sk_root_mat)
    
    # Binary green map for hypocotyl
    bin_green = np.zeros((*mask_hyp_clean.shape, 3), dtype=np.uint8)
    bin_green[mask_hyp_clean] = (34, 197, 94) # Modern emerald green
    
    algorithms = {
        "poda": {
            "name": "Zhang-Suen + Poda EDT (Modelo)",
            "tag": "Modelo de Referência (B-spline)",
            "description": "Afinamento de Zhang-Suen com poda morfológica adaptativa fundamentada no campo de distâncias euclidianas (EDT), eliminando espículas espúrias e medindo comprimento contínuo via B-splines (anteriormente denominado MAT com Poda).",
            "time_ms": round(t_poda, 2),
            "hypocotyl": {
                "pixels": px_h_p,
                "length_cm": round(len_px_h / pixels_per_cm, 2),
                "endpoints": ep_h_p,
                "branchpoints": bp_h_p
            },
            "root": {
                "pixels": px_r_p,
                "length_cm": round(len_px_r / pixels_per_cm, 2),
                "endpoints": ep_r_p,
                "branchpoints": bp_r_p
            },
            "total": {
                "pixels": px_h_p + px_r_p,
                "length_cm": round((len_px_h + len_px_r) / pixels_per_cm, 2),
                "branchpoints": bp_h_p + bp_r_p
            },
            "overlay_composite": to_base64_png(create_rgb_overlay(sk_hyp_poda, sk_root_poda, mask_hyp_clean, mask_root_clean)),
            "overlay_hyp": to_base64_png(create_organ_overlay(sk_hyp_poda, mask_hyp_clean, (0, 120, 0), (255, 255, 0))),
            "overlay_root": to_base64_png(create_organ_overlay(sk_root_poda, mask_root_clean, (120, 0, 0), (0, 255, 255)))
        },
        "zhang_suen": {
            "name": "Zhang-Suen Puro (1984)",
            "tag": "Afinamento Paralelo",
            "description": "Algoritmo de afinamento paralelo clássico iterativo em 8-vizinhança. Preserva detalhes finos e conectividade sem dependência do campo euclidiano.",
            "time_ms": round(t_zh, 2),
            "hypocotyl": {
                "pixels": px_h_z,
                "length_cm": round(px_h_z / pixels_per_cm, 2),
                "endpoints": ep_h_z,
                "branchpoints": bp_h_z
            },
            "root": {
                "pixels": px_r_z,
                "length_cm": round(px_r_z / pixels_per_cm, 2),
                "endpoints": ep_r_z,
                "branchpoints": bp_r_z
            },
            "total": {
                "pixels": px_h_z + px_r_z,
                "length_cm": round((px_h_z + px_r_z) / pixels_per_cm, 2),
                "branchpoints": bp_h_z + bp_r_z
            },
            "overlay_composite": to_base64_png(create_rgb_overlay(sk_hyp_zh, sk_root_zh, mask_hyp_clean, mask_root_clean)),
            "overlay_hyp": to_base64_png(create_organ_overlay(sk_hyp_zh, mask_hyp_clean, (0, 120, 0), (255, 255, 0))),
            "overlay_root": to_base64_png(create_organ_overlay(sk_root_zh, mask_root_clean, (120, 0, 0), (0, 255, 255)))
        },
        "lee": {
            "name": "Lee & Kashyap Puro (1994)",
            "tag": "Afinamento Topológico",
            "description": "Afinamento topológico baseado no número de Euler e invariância de gênero. Gera esqueletos unitários estritamente conexos e com cruzamentos enxutos.",
            "time_ms": round(t_lee, 2),
            "hypocotyl": {
                "pixels": px_h_l,
                "length_cm": round(px_h_l / pixels_per_cm, 2),
                "endpoints": ep_h_l,
                "branchpoints": bp_h_l
            },
            "root": {
                "pixels": px_r_l,
                "length_cm": round(px_r_l / pixels_per_cm, 2),
                "endpoints": ep_r_l,
                "branchpoints": bp_r_l
            },
            "total": {
                "pixels": px_h_l + px_r_l,
                "length_cm": round((px_h_l + px_r_l) / pixels_per_cm, 2),
                "branchpoints": bp_h_l + bp_r_l
            },
            "overlay_composite": to_base64_png(create_rgb_overlay(sk_hyp_lee, sk_root_lee, mask_hyp_clean, mask_root_clean)),
            "overlay_hyp": to_base64_png(create_organ_overlay(sk_hyp_lee, mask_hyp_clean, (0, 120, 0), (255, 255, 0))),
            "overlay_root": to_base64_png(create_organ_overlay(sk_root_lee, mask_root_clean, (120, 0, 0), (0, 255, 255)))
        },
        "mat_puro": {
            "name": "MAT Puro (Blum 1967)",
            "tag": "Eixo Medial Geométrico",
            "description": "Transformada geométrica do eixo medial clássica de Harry Blum baseada em cristas maximais inscritas. Mantém todas as bifurcações induzidas por irregularidades de borda.",
            "time_ms": round(t_mat, 2),
            "hypocotyl": {
                "pixels": px_h_m,
                "length_cm": round(px_h_m / pixels_per_cm, 2),
                "endpoints": ep_h_m,
                "branchpoints": bp_h_m
            },
            "root": {
                "pixels": px_r_m,
                "length_cm": round(px_r_m / pixels_per_cm, 2),
                "endpoints": ep_r_m,
                "branchpoints": bp_r_m
            },
            "total": {
                "pixels": px_h_m + px_r_m,
                "length_cm": round((px_h_m + px_r_m) / pixels_per_cm, 2),
                "branchpoints": bp_h_m + bp_r_m
            },
            "overlay_composite": to_base64_png(create_rgb_overlay(sk_hyp_mat, sk_root_mat, mask_hyp_clean, mask_root_clean)),
            "overlay_hyp": to_base64_png(create_organ_overlay(sk_hyp_mat, mask_hyp_clean, (0, 120, 0), (255, 255, 0))),
            "overlay_root": to_base64_png(create_organ_overlay(sk_root_mat, mask_root_clean, (120, 0, 0), (0, 255, 255)))
        }
    }
    
    return {
        "metadata": {
            "width": int(mask_bgr.shape[1]),
            "height": int(mask_bgr.shape[0]),
            "pixels_per_cm": pixels_per_cm,
            "hypocotyl_pixels": int(mask_hyp_clean.sum()),
            "hypocotyl_components": int(n_hyp_comp),
            "root_pixels": int(mask_root_clean.sum()),
            "root_components": int(n_root_comp)
        },
        "original_mask": to_base64_png(mask_rgb),
        "binary_hypocotyl": to_base64_png(bin_green),
        "algorithms": algorithms
    }

class SkeletonRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=BASE_DIR, **kwargs)
        
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        
        # 1. Samples list endpoint
        if parsed.path == "/api/samples":
            samples = []
            if os.path.isdir(LABELED_DIR):
                files = sorted(os.listdir(LABELED_DIR))
                for f in files:
                    if f.lower().endswith((".png", ".jpg", ".jpeg")):
                        samples.append({
                            "name": f,
                            "url": f"/api/sample/{urllib.parse.quote(f)}"
                        })
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(samples).encode("utf-8"))
            return
            
        # 2. Sample file download/preview
        elif parsed.path.startswith("/api/sample/"):
            fname = urllib.parse.unquote(parsed.path[len("/api/sample/"):])
            fpath = os.path.join(LABELED_DIR, fname)
            if os.path.exists(fpath):
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                with open(fpath, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                self.send_error(404, "Sample image not found")
                return

        return super().do_GET()

    def do_POST(self):
        if self.path == "/api/process":
            try:
                content_len = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_len)
                data = json.loads(body.decode("utf-8"))
                
                pixels_per_cm = float(data.get("pixels_per_cm", 100.0))
                min_comp_size = int(data.get("min_component_size", 30))
                
                img_bgr = None
                if data.get("sample"):
                    sample_name = data["sample"]
                    sample_file = os.path.join(LABELED_DIR, sample_name)
                    if not os.path.exists(sample_file):
                        self.send_error(404, "Sample file not found")
                        return
                    img_bgr = cv2.imread(sample_file, cv2.IMREAD_UNCHANGED)
                elif data.get("image"):
                    raw_b64 = data["image"]
                    if "," in raw_b64:
                        raw_b64 = raw_b64.split(",", 1)[1]
                    img_bytes = base64.b64decode(raw_b64)
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                
                if img_bgr is None:
                    self.send_error(400, "Could not decode input image")
                    return
                
                # Ensure 3-channel BGR
                if len(img_bgr.shape) == 3 and img_bgr.shape[2] == 4:
                    img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
                elif len(img_bgr.shape) == 2:
                    img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
                
                # Process pipeline
                results = process_image(img_bgr, pixels_per_cm, min_comp_size)
                
                resp = json.dumps(results).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(resp)
                return
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
                return
        
        self.send_error(404, "Unknown endpoint")

def run():
    server_address = ("", PORT)
    httpd = ThreadingHTTPServer(server_address, SkeletonRequestHandler)
    print(f"🚀 Servidor de esqueletização iniciado na porta {PORT}!")
    print(f"🔗 Acesse: http://localhost:{PORT}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServidor encerrado.")
        httpd.server_close()

if __name__ == "__main__":
    run()
