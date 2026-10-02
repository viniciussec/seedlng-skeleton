#!/usr/bin/env python3
"""
Benchmark e Avaliação em Lote dos 4 Métodos de Esqueletização por Plântula Individual.
Cada imagem do dataset contém ~50 plântulas individuais (teste de vigor/germinação).
O script identifica cada plântula individualmente em cada imagem (totalizando ~8.500 plântulas),
extrai os comprimentos de hipocótilo e raiz estritamente em pixels (px),
contabiliza as ramificações e exporta:
  1. resultados_mat_poda_por_plantula.csv
  2. resultados_zhang_suen_por_plantula.csv
  3. resultados_lee_por_plantula.csv
  4. resultados_mat_puro_por_plantula.csv
  5. resultados_comparativo_por_plantula.csv (lado a lado por plântula)
  6. resultados_resumo_por_imagem.csv (estatísticas agregadas das ~50 plântulas por imagem)
"""

import os
import sys
import time
import csv
import cv2
import numpy as np
from scipy.ndimage import (
    binary_fill_holes, label, convolve, distance_transform_edt, binary_dilation
)
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
    if not coords:
        return []
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
    if not skel_binary.any():
        return 0, 0, 0
    sk_u8 = skel_binary.astype(np.uint8)
    neighbors = convolve(sk_u8, KERNEL_3X3, mode="constant", cval=0) * sk_u8
    endpoints = int((neighbors == 1).sum())
    branchpoints = int((neighbors > 2).sum())
    pixels = int(skel_binary.sum())
    return pixels, endpoints, branchpoints

def identify_seedlings(mask_hyp, mask_root, min_size=50):
    """
    Identifica cada plântula individual na imagem por rotulagem de componentes conectados.
    Faz uma leve dilatação para unir hipocótilo e raiz da mesma plântula no colo (se houver 1-2px de gap na anotação).
    Ordena as plântulas espacialmente (em linhas, do topo para base, da esquerda para a direita).
    """
    plant_bin = binary_dilation(mask_hyp | mask_root, iterations=2)
    plant_bin = binary_fill_holes(plant_bin)
    
    labeled, n_plants = label(plant_bin)
    if n_plants == 0:
        return labeled, []
    
    sizes = np.bincount(labeled.ravel())
    valid_ids = [i for i in range(1, n_plants + 1) if sizes[i] >= min_size]
    
    # Ordenar plântulas por posição na bandeja (linha por linha)
    plant_positions = []
    for pid in valid_ids:
        ys, xs = np.where(labeled == pid)
        y_c = float(ys.mean())
        x_c = float(xs.mean())
        # Agrupa em faixas horizontais de 80px para ordenação natural
        row_key = round(y_c / 80.0) * 1000.0 + x_c
        plant_positions.append((row_key, y_c, x_c, pid))
    
    plant_positions.sort(key=lambda item: item[0])
    ordered_plants = [(item[1], item[2], item[3]) for item in plant_positions]
    return labeled, ordered_plants

def process_image_seedlings(mask_bgr, filename):
    """
    Executa os 4 algoritmos na imagem e extrai as métricas de cada plântula individual.
    """
    mask_hyp_raw, mask_root_raw, _ = extract_organ_masks(mask_bgr)
    
    mask_hyp_clean  = filter_small_components(preprocess_mask(mask_hyp_raw), 30)
    mask_root_clean = filter_small_components(preprocess_mask(mask_root_raw), 30)
    
    # 1. Identificar plântulas individuais
    labeled_plants, ordered_plants = identify_seedlings(mask_hyp_clean, mask_root_clean, min_size=50)
    
    # ----------------------------------------------------
    # 2. Executar Esqueletização Geral para cada Algoritmo
    # ----------------------------------------------------
    # A) Zhang-Suen com Poda EDT (Modelo com B-spline, anteriormente 'MAT com Poda')
    # NOTA: O esqueleto inicial 2D do skimage.morphology.skeletonize é Zhang-Suen (1984),
    # filtrado via Transformada de Distância Euclidiana (EDT) e interpolado por B-splines.
    t0 = time.time()
    dist_h = distance_transform_edt(mask_hyp_clean)
    dist_r = distance_transform_edt(mask_root_clean)
    sk_h_poda = prune_skeleton_by_dt(skeletonize(mask_hyp_clean), dist_h, min_length=5, min_dt_ratio=0.5)
    sk_r_poda = prune_skeleton_by_dt(skeletonize(mask_root_clean), dist_r, min_length=5, min_dt_ratio=0.5)
    t_poda = (time.time() - t0) * 1000

    # B) Zhang-Suen
    t0 = time.time()
    sk_h_zh = skeletonize(mask_hyp_raw, method="zhang")
    sk_r_zh = skeletonize(mask_root_raw, method="zhang")
    t_zh = (time.time() - t0) * 1000

    # C) Lee & Kashyap
    t0 = time.time()
    sk_h_lee = skeletonize(mask_hyp_raw, method="lee")
    sk_r_lee = skeletonize(mask_root_raw, method="lee")
    t_lee = (time.time() - t0) * 1000

    # D) MAT Puro (Blum)
    t0 = time.time()
    sk_h_mat, _ = medial_axis(mask_hyp_raw, return_distance=True)
    sk_r_mat, _ = medial_axis(mask_root_raw, return_distance=True)
    t_mat = (time.time() - t0) * 1000

    rows_poda_sample = []
    rows_zhang_sample = []
    rows_lee_sample = []
    rows_mat_sample = []
    rows_comp_sample = []

    for plant_idx, (y_c, x_c, pid) in enumerate(ordered_plants, 1):
        m_plant = (labeled_plants == pid)
        
        # Hipocótilo e Raiz dessa plântula
        hyp_px_count = int((mask_hyp_clean & m_plant).sum())
        root_px_count = int((mask_root_clean & m_plant).sum())
        
        status = "Normal"
        if hyp_px_count > 0 and root_px_count == 0:
            status = "Sem Raiz"
        elif hyp_px_count == 0 and root_px_count > 0:
            status = "Sem Hipocótilo"

        # --- A) MAT com Poda ---
        sk_h_i = sk_h_poda & m_plant
        sk_r_i = sk_r_poda & m_plant
        
        c_h = compute_skeleton_chain(sk_h_i, dist_h)
        len_h_poda = sum(chain_length_spline(c) for c in c_h if len(c) >= 2) or float(sk_h_i.sum())
        
        c_r = compute_skeleton_chain(sk_r_i, dist_r)
        len_r_poda = sum(chain_length_spline(c) for c in c_r if len(c) >= 2) or float(sk_r_i.sum())
        
        _, _, bp_h_poda = get_topology(sk_h_i)
        _, _, bp_r_poda = get_topology(sk_r_i)
        tot_len_poda = round(len_h_poda + len_r_poda, 2)
        tot_bp_poda = bp_h_poda + bp_r_poda

        rows_poda_sample.append({
            "amostra": filename,
            "plantula_id": plant_idx,
            "pos_x": round(x_c, 1),
            "pos_y": round(y_c, 1),
            "metodo": "MAT com Poda a Distância",
            "hipocotilo_px": round(len_h_poda, 2),
            "hipocotilo_ramificacoes": bp_h_poda,
            "raiz_px": round(len_r_poda, 2),
            "raiz_ramificacoes": bp_r_poda,
            "comprimento_total_px": tot_len_poda,
            "ramificacoes_totais": tot_bp_poda,
            "status": status
        })

        # --- B) Zhang-Suen ---
        sk_h_i = sk_h_zh & m_plant
        sk_r_i = sk_r_zh & m_plant
        len_h_zh = float(sk_h_i.sum())
        len_r_zh = float(sk_r_i.sum())
        _, _, bp_h_zh = get_topology(sk_h_i)
        _, _, bp_r_zh = get_topology(sk_r_i)
        tot_len_zh = round(len_h_zh + len_r_zh, 2)
        tot_bp_zh = bp_h_zh + bp_r_zh

        rows_zhang_sample.append({
            "amostra": filename,
            "plantula_id": plant_idx,
            "pos_x": round(x_c, 1),
            "pos_y": round(y_c, 1),
            "metodo": "Zhang-Suen Puro (1984)",
            "hipocotilo_px": round(len_h_zh, 2),
            "hipocotilo_ramificacoes": bp_h_zh,
            "raiz_px": round(len_r_zh, 2),
            "raiz_ramificacoes": bp_r_zh,
            "comprimento_total_px": tot_len_zh,
            "ramificacoes_totais": tot_bp_zh,
            "status": status
        })

        # --- C) Lee & Kashyap ---
        sk_h_i = sk_h_lee & m_plant
        sk_r_i = sk_r_lee & m_plant
        len_h_lee = float(sk_h_i.sum())
        len_r_lee = float(sk_r_i.sum())
        _, _, bp_h_lee = get_topology(sk_h_i)
        _, _, bp_r_lee = get_topology(sk_r_i)
        tot_len_lee = round(len_h_lee + len_r_lee, 2)
        tot_bp_lee = bp_h_lee + bp_r_lee

        rows_lee_sample.append({
            "amostra": filename,
            "plantula_id": plant_idx,
            "pos_x": round(x_c, 1),
            "pos_y": round(y_c, 1),
            "metodo": "Lee & Kashyap Puro (1994)",
            "hipocotilo_px": round(len_h_lee, 2),
            "hipocotilo_ramificacoes": bp_h_lee,
            "raiz_px": round(len_r_lee, 2),
            "raiz_ramificacoes": bp_r_lee,
            "comprimento_total_px": tot_len_lee,
            "ramificacoes_totais": tot_bp_lee,
            "status": status
        })

        # --- D) MAT Puro ---
        sk_h_i = sk_h_mat & m_plant
        sk_r_i = sk_r_mat & m_plant
        len_h_mat = float(sk_h_i.sum())
        len_r_mat = float(sk_r_i.sum())
        _, _, bp_h_mat = get_topology(sk_h_i)
        _, _, bp_r_mat = get_topology(sk_r_i)
        tot_len_mat = round(len_h_mat + len_r_mat, 2)
        tot_bp_mat = bp_h_mat + bp_r_mat

        rows_mat_sample.append({
            "amostra": filename,
            "plantula_id": plant_idx,
            "pos_x": round(x_c, 1),
            "pos_y": round(y_c, 1),
            "metodo": "MAT Puro (Blum 1967)",
            "hipocotilo_px": round(len_h_mat, 2),
            "hipocotilo_ramificacoes": bp_h_mat,
            "raiz_px": round(len_r_mat, 2),
            "raiz_ramificacoes": bp_r_mat,
            "comprimento_total_px": tot_len_mat,
            "ramificacoes_totais": tot_bp_mat,
            "status": status
        })

        # --- E) Linha Consolidada por Plântula ---
        rows_comp_sample.append({
            "amostra": filename,
            "plantula_id": plant_idx,
            "pos_x": round(x_c, 1),
            "pos_y": round(y_c, 1),
            "status": status,
            "comp_mat_poda_px": tot_len_poda,
            "comp_zhang_suen_px": tot_len_zh,
            "comp_lee_px": tot_len_lee,
            "comp_mat_puro_px": tot_len_mat,
            "ramif_mat_poda": tot_bp_poda,
            "ramif_zhang_suen": tot_bp_zh,
            "ramif_lee": tot_bp_lee,
            "ramif_mat_puro": tot_bp_mat
        })

    # Resumo da Imagem (Bandeja)
    resumo_imagem = {
        "amostra": filename,
        "total_plantulas_detectadas": len(ordered_plants),
        "media_comp_mat_poda_px": round(float(np.mean([r["comprimento_total_px"] for r in rows_poda_sample])), 2) if rows_poda_sample else 0,
        "media_comp_zhang_suen_px": round(float(np.mean([r["comprimento_total_px"] for r in rows_zhang_sample])), 2) if rows_zhang_sample else 0,
        "media_comp_lee_px": round(float(np.mean([r["comprimento_total_px"] for r in rows_lee_sample])), 2) if rows_lee_sample else 0,
        "media_comp_mat_puro_px": round(float(np.mean([r["comprimento_total_px"] for r in rows_mat_sample])), 2) if rows_mat_sample else 0,
        "total_ramif_mat_poda": sum(r["ramificacoes_totais"] for r in rows_poda_sample),
        "total_ramif_zhang_suen": sum(r["ramificacoes_totais"] for r in rows_zhang_sample),
        "total_ramif_lee": sum(r["ramificacoes_totais"] for r in rows_lee_sample),
        "total_ramif_mat_puro": sum(r["ramificacoes_totais"] for r in rows_mat_sample),
        "tempo_total_ms": round(t_poda + t_zh + t_lee + t_mat, 2)
    }

    return (
        rows_poda_sample, rows_zhang_sample, rows_lee_sample, rows_mat_sample,
        rows_comp_sample, resumo_imagem
    )

def main():
    if not os.path.isdir(DATASET_DIR):
        print(f"❌ Erro: Diretório {DATASET_DIR} não encontrado.")
        sys.exit(1)

    image_files = sorted([f for f in os.listdir(DATASET_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    total_imgs = len(image_files)

    print("=" * 80)
    print(f"🌱 Benchmark por Plântula Individual — {total_imgs} Imagens (~50 plântulas/imagem)")
    print("📏 Medições estritamente em pixels (px)")
    print("=" * 80)

    all_rows_poda = []
    all_rows_zhang = []
    all_rows_lee = []
    all_rows_mat = []
    all_rows_comp = []
    all_rows_resumo = []

    t_start = time.time()
    total_seedlings_count = 0

    for idx, fname in enumerate(image_files, 1):
        fpath = os.path.join(DATASET_DIR, fname)
        img_bgr = cv2.imread(fpath)
        if img_bgr is None:
            print(f"⚠️ [Aviso] Falha ao ler {fname}. Pulando...")
            continue

        r_poda, r_zhang, r_lee, r_mat, r_comp, r_resumo = process_image_seedlings(img_bgr, fname)

        all_rows_poda.extend(r_poda)
        all_rows_zhang.extend(r_zhang)
        all_rows_lee.extend(r_lee)
        all_rows_mat.extend(r_mat)
        all_rows_comp.extend(r_comp)
        all_rows_resumo.append(r_resumo)

        n_p = len(r_poda)
        total_seedlings_count += n_p

        if idx % 10 == 0 or idx == total_imgs:
            elapsed = time.time() - t_start
            print(f"  [{idx:3d}/{total_imgs}] {fname[:32]:<32} | {n_p:2d} plântulas | Acumulado: {total_seedlings_count:5d} plântulas ({elapsed:.1f}s)")

    total_time = time.time() - t_start
    print("=" * 80)
    print(f"✅ Concluído com sucesso! {total_seedlings_count} plântulas processadas em {total_time:.2f}s!")
    print("=" * 80)

    # 1. Salvar os 4 CSVs individuais por plântula
    cols_individual = [
        "amostra", "plantula_id", "pos_x", "pos_y", "metodo",
        "hipocotilo_px", "hipocotilo_ramificacoes",
        "raiz_px", "raiz_ramificacoes",
        "comprimento_total_px", "ramificacoes_totais", "status"
    ]

    outputs = [
        ("resultados_mat_poda_por_plantula.csv", all_rows_poda),
        ("resultados_zhang_suen_por_plantula.csv", all_rows_zhang),
        ("resultados_lee_por_plantula.csv", all_rows_lee),
        ("resultados_mat_puro_por_plantula.csv", all_rows_mat),
    ]

    for fname, data_rows in outputs:
        out_path = os.path.join(BASE_DIR, fname)
        with open(out_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols_individual)
            writer.writeheader()
            writer.writerows(data_rows)
        print(f"📄 Arquivo salvo: {fname} ({len(data_rows)} plântulas)")

    # 2. Salvar CSV comparativo consolidado por plântula
    cols_comp = [
        "amostra", "plantula_id", "pos_x", "pos_y", "status",
        "comp_mat_poda_px", "comp_zhang_suen_px", "comp_lee_px", "comp_mat_puro_px",
        "ramif_mat_poda", "ramif_zhang_suen", "ramif_lee", "ramif_mat_puro"
    ]
    path_comp = os.path.join(BASE_DIR, "resultados_comparativo_por_plantula.csv")
    with open(path_comp, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols_comp)
        writer.writeheader()
        writer.writerows(all_rows_comp)
    print(f"📊 Arquivo consolidado salvo: resultados_comparativo_por_plantula.csv ({len(all_rows_comp)} plântulas)")

    # 3. Salvar Resumo Agregado por Imagem
    cols_resumo = [
        "amostra", "total_plantulas_detectadas",
        "media_comp_mat_poda_px", "media_comp_zhang_suen_px", "media_comp_lee_px", "media_comp_mat_puro_px",
        "total_ramif_mat_poda", "total_ramif_zhang_suen", "total_ramif_lee", "total_ramif_mat_puro",
        "tempo_total_ms"
    ]
    path_resumo = os.path.join(BASE_DIR, "resultados_resumo_por_imagem.csv")
    with open(path_resumo, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols_resumo)
        writer.writeheader()
        writer.writerows(all_rows_resumo)
    print(f"📋 Resumo por imagem salvo: resultados_resumo_por_imagem.csv ({len(all_rows_resumo)} imagens)")

    # Estatísticas Finais Globais por Plântula
    print("\n" + "=" * 80)
    print(f"📈 ESTATÍSTICAS GERAIS POR PLÂNTULA (N = {total_seedlings_count} plântulas):")
    print("=" * 80)
    print(f"{'Algoritmo':<28} | {'Comp. Plântula (px)':<20} | {'Ramificações/Plântula':<22}")
    print("-" * 80)

    for name, rlist in [("MAT com Poda a Distância", all_rows_poda),
                        ("Zhang-Suen Puro", all_rows_zhang),
                        ("Lee & Kashyap Puro", all_rows_lee),
                        ("MAT Puro (Blum)", all_rows_mat)]:
        lens = [r["comprimento_total_px"] for r in rlist]
        bps  = [r["ramificacoes_totais"] for r in rlist]
        print(f"{name:<28} | {np.mean(lens):6.1f} ± {np.std(lens):<5.1f} px    | {np.mean(bps):4.2f} ± {np.std(bps):<4.2f}")
    print("=" * 80)

if __name__ == "__main__":
    main()
