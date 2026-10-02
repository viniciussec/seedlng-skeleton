#!/usr/bin/env python3
"""
Benchmark e Avaliação em Lote dos 4 Métodos de Esqueletização
Processa todas as 170 plântulas de labeled-dataset e exporta 4 CSVs individuais
(comprimentos medidos exclusivamente em pixels) e um CSV comparativo consolidado.
"""

import os
import sys
import time
import csv
import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes, label, convolve, distance_transform_edt
from scipy.interpolate import splprep, splev
from skimage.morphology import skeletonize, medial_axis

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "labeled-dataset")

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
        return float(length)
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

def evaluate_sample(mask_bgr, min_comp_size=30):
    """
    Avalia os 4 métodos para uma única máscara e retorna métricas puras em pixels.
    """
    mask_hyp_raw, mask_root_raw, _ = extract_organ_masks(mask_bgr)
    
    mask_hyp_clean  = filter_small_components(preprocess_mask(mask_hyp_raw), min_comp_size)
    mask_root_clean = filter_small_components(preprocess_mask(mask_root_raw), min_comp_size)
    
    # ----------------------------------------------------
    # 1. Zhang-Suen com Poda EDT (Modelo de Referência com B-spline, anteriormente 'MAT com Poda')
    # NOTA: O esqueleto inicial 2D do skimage é Zhang-Suen, podado por EDT e medido via B-splines.
    # ----------------------------------------------------
    t0 = time.time()
    dist_h = distance_transform_edt(mask_hyp_clean)
    dist_r = distance_transform_edt(mask_root_clean)
    
    sk_h_init = skeletonize(mask_hyp_clean)
    sk_r_init = skeletonize(mask_root_clean)
    
    sk_h_poda = prune_skeleton_by_dt(sk_h_init, dist_h, min_length=5, min_dt_ratio=0.5)
    sk_r_poda = prune_skeleton_by_dt(sk_r_init, dist_r, min_length=5, min_dt_ratio=0.5)
    
    chains_h = compute_skeleton_chain(sk_h_poda, dist_h)
    chains_r = compute_skeleton_chain(sk_r_poda, dist_r)
    
    len_px_h_poda = sum(chain_length_spline(c) for c in chains_h if len(c) >= 2)
    len_px_r_poda = sum(chain_length_spline(c) for c in chains_r if len(c) >= 2)
    if len_px_h_poda == 0: len_px_h_poda = float(sk_h_poda.sum())
    if len_px_r_poda == 0: len_px_r_poda = float(sk_r_poda.sum())
    
    t_poda = (time.time() - t0) * 1000
    px_h_p, ep_h_p, bp_h_p = get_topology(sk_h_poda)
    px_r_p, ep_r_p, bp_r_p = get_topology(sk_r_poda)
    
    res_poda = {
        "metodo": "MAT com Poda a Distância",
        "hipocotilo_comprimento_px": round(len_px_h_poda, 2),
        "hipocotilo_pixels": px_h_p,
        "hipocotilo_ramificacoes": bp_h_p,
        "hipocotilo_terminais": ep_h_p,
        "raiz_comprimento_px": round(len_px_r_poda, 2),
        "raiz_pixels": px_r_p,
        "raiz_ramificacoes": bp_r_p,
        "raiz_terminais": ep_r_p,
        "comprimento_total_px": round(len_px_h_poda + len_px_r_poda, 2),
        "ramificacoes_totais": bp_h_p + bp_r_p,
        "tempo_ms": round(t_poda, 2)
    }

    # ----------------------------------------------------
    # 2. Zhang-Suen Puro (1984)
    # ----------------------------------------------------
    t0 = time.time()
    sk_h_zh = skeletonize(mask_hyp_raw, method="zhang")
    sk_r_zh = skeletonize(mask_root_raw, method="zhang")
    t_zh = (time.time() - t0) * 1000
    
    px_h_z, ep_h_z, bp_h_z = get_topology(sk_h_zh)
    px_r_z, ep_r_z, bp_r_z = get_topology(sk_r_zh)
    
    res_zhang = {
        "metodo": "Zhang-Suen Puro (1984)",
        "hipocotilo_comprimento_px": px_h_z,
        "hipocotilo_pixels": px_h_z,
        "hipocotilo_ramificacoes": bp_h_z,
        "hipocotilo_terminais": ep_h_z,
        "raiz_comprimento_px": px_r_z,
        "raiz_pixels": px_r_z,
        "raiz_ramificacoes": bp_r_z,
        "raiz_terminais": ep_r_z,
        "comprimento_total_px": px_h_z + px_r_z,
        "ramificacoes_totais": bp_h_z + bp_r_z,
        "tempo_ms": round(t_zh, 2)
    }

    # ----------------------------------------------------
    # 3. Lee & Kashyap Puro (1994)
    # ----------------------------------------------------
    t0 = time.time()
    sk_h_lee = skeletonize(mask_hyp_raw, method="lee")
    sk_r_lee = skeletonize(mask_root_raw, method="lee")
    t_lee = (time.time() - t0) * 1000
    
    px_h_l, ep_h_l, bp_h_l = get_topology(sk_h_lee)
    px_r_l, ep_r_l, bp_r_l = get_topology(sk_r_lee)
    
    res_lee = {
        "metodo": "Lee & Kashyap Puro (1994)",
        "hipocotilo_comprimento_px": px_h_l,
        "hipocotilo_pixels": px_h_l,
        "hipocotilo_ramificacoes": bp_h_l,
        "hipocotilo_terminais": ep_h_l,
        "raiz_comprimento_px": px_r_l,
        "raiz_pixels": px_r_l,
        "raiz_ramificacoes": bp_r_l,
        "raiz_terminais": ep_r_l,
        "comprimento_total_px": px_h_l + px_r_l,
        "ramificacoes_totais": bp_h_l + bp_r_l,
        "tempo_ms": round(t_lee, 2)
    }

    # ----------------------------------------------------
    # 4. MAT Puro (Blum 1967)
    # ----------------------------------------------------
    t0 = time.time()
    sk_h_mat, _ = medial_axis(mask_hyp_raw, return_distance=True)
    sk_r_mat, _ = medial_axis(mask_root_raw, return_distance=True)
    t_mat = (time.time() - t0) * 1000
    
    px_h_m, ep_h_m, bp_h_m = get_topology(sk_h_mat)
    px_r_m, ep_r_m, bp_r_m = get_topology(sk_r_mat)
    
    res_mat_puro = {
        "metodo": "MAT Puro (Blum 1967)",
        "hipocotilo_comprimento_px": px_h_m,
        "hipocotilo_pixels": px_h_m,
        "hipocotilo_ramificacoes": bp_h_m,
        "hipocotilo_terminais": ep_h_m,
        "raiz_comprimento_px": px_r_m,
        "raiz_pixels": px_r_m,
        "raiz_ramificacoes": bp_r_m,
        "raiz_terminais": ep_r_m,
        "comprimento_total_px": px_h_m + px_r_m,
        "ramificacoes_totais": bp_h_m + bp_r_m,
        "tempo_ms": round(t_mat, 2)
    }

    return res_poda, res_zhang, res_lee, res_mat_puro

def main():
    if not os.path.isdir(DATASET_DIR):
        print(f"❌ Erro: Diretório {DATASET_DIR} não encontrado.")
        sys.exit(1)

    image_files = sorted([f for f in os.listdir(DATASET_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    total = len(image_files)
    print("=" * 75)
    print(f"🌱 Benchmark de Esqueletização em Lote — {total} Plântulas")
    print("📏 Medição estritamente em pixels (px)")
    print("=" * 75)

    rows_poda = []
    rows_zhang = []
    rows_lee = []
    rows_mat_puro = []
    rows_consolidado = []

    t_start_all = time.time()

    for idx, fname in enumerate(image_files, 1):
        fpath = os.path.join(DATASET_DIR, fname)
        img_bgr = cv2.imread(fpath)
        if img_bgr is None:
            print(f"⚠️ [Aviso] Falha ao ler {fname}. Pulando...")
            continue

        h, w = img_bgr.shape[:2]

        r_poda, r_zhang, r_lee, r_mat = evaluate_sample(img_bgr)

        # Adiciona metadados
        for r in (r_poda, r_zhang, r_lee, r_mat):
            r["amostra"] = fname
            r["largura_px"] = w
            r["altura_px"] = h

        rows_poda.append(r_poda)
        rows_zhang.append(r_zhang)
        rows_lee.append(r_lee)
        rows_mat_puro.append(r_mat)

        # Linha consolidada
        rows_consolidado.append({
            "amostra": fname,
            "comp_total_px_mat_poda": r_poda["comprimento_total_px"],
            "comp_total_px_zhang_suen": r_zhang["comprimento_total_px"],
            "comp_total_px_lee": r_lee["comprimento_total_px"],
            "comp_total_px_mat_puro": r_mat["comprimento_total_px"],
            "ramificacoes_mat_poda": r_poda["ramificacoes_totais"],
            "ramificacoes_zhang_suen": r_zhang["ramificacoes_totais"],
            "ramificacoes_lee": r_lee["ramificacoes_totais"],
            "ramificacoes_mat_puro": r_mat["ramificacoes_totais"],
            "tempo_ms_mat_poda": r_poda["tempo_ms"],
            "tempo_ms_zhang_suen": r_zhang["tempo_ms"],
            "tempo_ms_lee": r_lee["tempo_ms"],
            "tempo_ms_mat_puro": r_mat["tempo_ms"]
        })

        if idx % 10 == 0 or idx == total:
            elapsed = time.time() - t_start_all
            avg_per_item = (elapsed / idx) * 1000
            print(f"  [{idx:3d}/{total}] {fname[:38]:<38} | {elapsed:.1f}s ({avg_per_item:.1f}ms/amostra)")

    total_time = time.time() - t_start_all
    print("=" * 75)
    print(f"✅ Processamento concluído em {total_time:.2f} segundos!")
    print("=" * 75)

    # Definição das colunas individuais
    cols_individual = [
        "amostra", "metodo", "largura_px", "altura_px",
        "hipocotilo_comprimento_px", "hipocotilo_pixels", "hipocotilo_ramificacoes", "hipocotilo_terminais",
        "raiz_comprimento_px", "raiz_pixels", "raiz_ramificacoes", "raiz_terminais",
        "comprimento_total_px", "ramificacoes_totais", "tempo_ms"
    ]

    outputs = [
        ("resultados_mat_poda.csv", rows_poda),
        ("resultados_zhang_suen.csv", rows_zhang),
        ("resultados_lee.csv", rows_lee),
        ("resultados_mat_puro.csv", rows_mat_puro),
    ]

    for filename, rows in outputs:
        out_path = os.path.join(BASE_DIR, filename)
        with open(out_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols_individual)
            writer.writeheader()
            writer.writerows(rows)
        print(f"📄 Arquivo salvo: {filename} ({len(rows)} linhas)")

    # Salva consolidado
    cols_consolidado = [
        "amostra",
        "comp_total_px_mat_poda", "comp_total_px_zhang_suen", "comp_total_px_lee", "comp_total_px_mat_puro",
        "ramificacoes_mat_poda", "ramificacoes_zhang_suen", "ramificacoes_lee", "ramificacoes_mat_puro",
        "tempo_ms_mat_poda", "tempo_ms_zhang_suen", "tempo_ms_lee", "tempo_ms_mat_puro"
    ]
    path_consolidado = os.path.join(BASE_DIR, "resultados_comparativo_consolidado.csv")
    with open(path_consolidado, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols_consolidado)
        writer.writeheader()
        writer.writerows(rows_consolidado)
    print(f"📊 Arquivo consolidado salvo: resultados_comparativo_consolidado.csv ({len(rows_consolidado)} linhas)")

    # Tabela Resumo Estatístico
    print("\n" + "=" * 75)
    print("📈 RESUMO ESTATÍSTICO GERAL (MÉDIAS DO DATASET):")
    print("=" * 75)
    header = f"{'Algoritmo':<28} | {'Comp. Total (px)':<18} | {'Ramificações':<14} | {'Tempo Médio':<12}"
    print(header)
    print("-" * 75)
    
    for name, rows in [("MAT com Poda a Distância", rows_poda),
                       ("Zhang-Suen Puro", rows_zhang),
                       ("Lee & Kashyap Puro", rows_lee),
                       ("MAT Puro (Blum)", rows_mat_puro)]:
        tot_lens = [r["comprimento_total_px"] for r in rows]
        tot_bps  = [r["ramificacoes_totais"] for r in rows]
        tot_tms  = [r["tempo_ms"] for r in rows]

        mean_len = np.mean(tot_lens)
        std_len  = np.std(tot_lens)
        mean_bp  = np.mean(tot_bps)
        std_bp   = np.std(tot_bps)
        mean_t   = np.mean(tot_tms)

        print(f"{name:<28} | {mean_len:7.1f} ± {std_len:<6.1f} px | {mean_bp:5.1f} ± {std_bp:<4.1f}  | {mean_t:6.2f} ms")
    print("=" * 75)

if __name__ == "__main__":
    main()
