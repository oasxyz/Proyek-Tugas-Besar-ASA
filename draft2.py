import random
import time
import tracemalloc
import matplotlib.pyplot as plt
import numpy as np
import sys

# Meningkatkan limit rekursi Python agar DFS tidak crash saat menelusuri pohon
sys.setrecursionlimit(2000000)

# ==========================================
# 1. SETUP DATA DAN PARAMETER
# ==========================================
MAX_CAPACITY = 270

base_items = [
    {"name": "First Aid Kit", "weight": 10, "value": 50},
    {"name": "Medkit", "weight": 20, "value": 40},
    {"name": "Bandage (5x)", "weight": 10, "value": 40},
    {"name": "Energy Drink", "weight": 4, "value": 35},
    {"name": "Painkiller", "weight": 10, "value": 30},
    {"name": "Smoke Grenade", "weight": 14, "value": 35},
    {"name": "Frag Grenade", "weight": 27, "value": 35}, 
    {"name": "Amunisi 5.56mm", "weight": 15, "value": 35},
    {"name": "Amunisi 7.62mm", "weight": 21, "value": 35} 
]

# Modifikasi item_counts: Diturunkan sedikit dari draft lama agar kapasitas 270 tidak langsung penuh 
# oleh Medis/Throwables, sehingga algoritma dipaksa mengambil amunisi sekunder (7.62mm) pada skenario Hybrid.
item_counts = [2, 1, 3, 4, 3, 3, 2, 8, 8]

KATEGORI_ITEM = {
    "First Aid Kit": "Medis", "Medkit": "Medis", "Bandage (5x)": "Medis",
    "Energy Drink": "Medis", "Painkiller": "Medis",
    "Smoke Grenade": "Throwables", "Frag Grenade": "Throwables",
    "Amunisi 5.56mm": "Amunisi", "Amunisi 7.62mm": "Amunisi"
}

# ==========================================
# 2. IMPLEMENTASI 5 ALGORITMA (Pendekatan 0/1 KP)
# ==========================================

def knapsack_dp(capacity, items):
    n = len(items)
    dp_table = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            weight = items[i-1]['weight']
            value = items[i-1]['value']
            if weight <= w:
                dp_table[i][w] = max(value + dp_table[i-1][w - weight], dp_table[i-1][w])
            else:
                dp_table[i][w] = dp_table[i-1][w]
                
    selected_items = []
    w = capacity
    for i in range(n, 0, -1):
        if dp_table[i][w] != dp_table[i-1][w]:
            selected_items.append(items[i-1])
            w -= items[i-1]['weight']
            
    return dp_table[n][capacity], selected_items

def knapsack_greedy(capacity, items):
    sorted_items = sorted(items, key=lambda x: x['value'] / x['weight'], reverse=True)
    total_value = 0
    current_weight = 0
    selected_items = []
    
    for item in sorted_items:
        if current_weight + item['weight'] <= capacity:
            selected_items.append(item)
            total_value += item['value']
            current_weight += item['weight']
            
    return total_value, selected_items

def knapsack_ga(capacity, items, pop_size=50, generations=100, mutation_rate=0.1):
    n = len(items)
    
    def fitness(chromosome):
        total_weight = sum(chromosome[i] * items[i]['weight'] for i in range(n))
        total_value = sum(chromosome[i] * items[i]['value'] for i in range(n))
        if total_weight > capacity:
            return 0  # Penalti
        return total_value

    population = [[random.choice([0, 1]) for _ in range(n)] for _ in range(pop_size)]
    
    for _ in range(generations):
        population = sorted(population, key=lambda x: fitness(x), reverse=True)
        next_gen = population[:2] 
        
        while len(next_gen) < pop_size:
            parent1, parent2 = random.choices(population[:10], k=2)
            crossover_point = random.randint(1, n-1)
            child = parent1[:crossover_point] + parent2[crossover_point:]
            
            for i in range(n):
                if random.random() < mutation_rate:
                    child[i] = 1 - child[i]
            next_gen.append(child)
            
        population = next_gen
        
    best_chromosome = max(population, key=fitness)
    selected_items = [items[i] for i in range(n) if best_chromosome[i] == 1]
    
    return fitness(best_chromosome), selected_items

def knapsack_dfs(capacity, items):
    max_val = 0
    best_items = []
    iterations = [0]
    LIMIT = 1000000 # Cut-off limit mencegah proses berjam-jam

    def solve(idx, curr_w, curr_v, path):
        iterations[0] += 1
        if iterations[0] > LIMIT:
            return
        
        if idx == len(items):
            nonlocal max_val, best_items
            if curr_v > max_val:
                max_val = curr_v
                best_items = list(path)
            return

        # Cabang 1: Tidak mengambil barang
        solve(idx + 1, curr_w, curr_v, path)
        
        # Cabang 2: Mengambil barang (jika muat)
        if curr_w + items[idx]['weight'] <= capacity:
            path.append(items[idx])
            solve(idx + 1, curr_w + items[idx]['weight'], curr_v + items[idx]['value'], path)
            path.pop()

    solve(0, 0, 0, [])
    return max_val, best_items

def knapsack_bfs(capacity, items):
    from collections import deque
    max_val = 0
    best_items = []
    queue = deque([(0, 0, 0, [])]) # (index, weight, value, path)
    iterations = 0
    LIMIT = 1000000 # Cut-off memori limit

    while queue:
        iterations += 1
        if iterations > LIMIT:
            break
            
        idx, curr_w, curr_v, path = queue.popleft()
        
        if idx == len(items):
            if curr_v > max_val:
                max_val = curr_v
                best_items = path
            continue
            
        # Tidak ambil
        queue.append((idx + 1, curr_w, curr_v, path))
        
        # Ambil
        if curr_w + items[idx]['weight'] <= capacity:
            queue.append((idx + 1, curr_w + items[idx]['weight'], curr_v + items[idx]['value'], path + [items[idx]]))
            
    return max_val, best_items

# ==========================================
# 3. FUNGSI UTILITAS PENGOLAHAN DATA
# ==========================================

def rekap_kategori(selected_items):
    rekap = {"Medis": 0, "Amunisi": 0, "Throwables": 0}
    for item in selected_items:
        rekap[KATEGORI_ITEM[item['name']]] += 1
    return rekap

def hitung_bobot(selected_items):
    return sum(item['weight'] for item in selected_items)

def profile_algo(func, capacity, items):
    tracemalloc.start()
    start_time = time.perf_counter()
    
    val, sel_items = func(capacity, items)
    
    exec_time = (time.perf_counter() - start_time) * 1000
    current, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return val, sel_items, exec_time, peak_mem / 1024 # KB

def run_scenario(scenario_name, m_556, m_762, capacity=MAX_CAPACITY):
    print(f"\n{'='*60}\n--- {scenario_name} ---\n{'='*60}")
    
    current_pool = []
    for item, count in zip(base_items, item_counts):
        for _ in range(count):
            new_item = item.copy()
            if new_item['name'] == "Amunisi 5.56mm":
                new_item['value'] *= m_556
            elif new_item['name'] == "Amunisi 7.62mm":
                new_item['value'] *= m_762
            
            if new_item['value'] > 0:
                current_pool.append(new_item)

    algos = [
        ("DP", knapsack_dp),
        ("Greedy", knapsack_greedy),
        ("GA", knapsack_ga),
        ("DFS", knapsack_dfs),
        ("BFS", knapsack_bfs)
    ]
    
    res_val, res_time, res_mem, res_weight, res_cat, res_raw = [], [], [], [], [], []
    
    for name, func in algos:
        val, sel_items, t_ms, mem_kb = profile_algo(func, capacity, current_pool)
        w = hitung_bobot(sel_items)
        
        res_val.append(val)
        res_time.append(t_ms)
        res_mem.append(mem_kb)
        res_weight.append(w)
        res_cat.append(rekap_kategori(sel_items))
        res_raw.append(sel_items)
        
        print(f"[{name:<6}] Value: {val:<5.1f} | Bobot: {w}/{capacity} | Time: {t_ms:>8.2f} ms | Mem: {mem_kb:>8.2f} KB")

    return {
        "values": res_val, "times": res_time, "mems": res_mem, 
        "weights": res_weight, "categories": res_cat, "raw_items": res_raw
    }

# ==========================================
# 4. FUNGSI VISUALISASI MATPLOTLIB
# ==========================================

def add_labels(bars, ax, fmt='{:.2f}', offset=2, rotation=0):
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + offset, fmt.format(yval), 
                ha='center', va='bottom', fontsize=8, fontweight='bold', color='black', rotation=rotation)

def main():
    print("=== EKSPERIMEN 4 SKENARIO KNAPSACK: METADATA PUBG ===")
    
    res_s1 = run_scenario("SKENARIO 1: Full 5.56mm (M416)", 2.0, 0.0)
    res_s2 = run_scenario("SKENARIO 2: Full 7.62mm (AKM)", 0.0, 2.0)
    res_s3a = run_scenario("SKENARIO 3A: Hybrid (AR 5.56 + DMR 7.62)", 1.8, 1.2)
    res_s3b = run_scenario("SKENARIO 3B: Hybrid (AR 7.62 + DMR 5.56)", 1.2, 1.8)

    print("\nMenyiapkan 6 Grafik Komparasi... (Silakan cek window pop-up)")
    
    labels_algo = ['DP', 'Greedy', 'GA', 'DFS', 'BFS']
    x = np.arange(len(labels_algo))
    width = 0.2

    # --- GRAFIK 1: WAKTU EKSEKUSI ---
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    fig1.canvas.manager.set_window_title('Grafik 1 - Waktu Eksekusi')
    
    b1 = ax1.bar(x - 0.3, res_s1["times"], width=width, label='S1', color='#2ca02c')
    b2 = ax1.bar(x - 0.1, res_s2["times"], width=width, label='S2', color='#1f77b4')
    b3a = ax1.bar(x + 0.1, res_s3a["times"], width=width, label='S3A', color='#ff7f0e')
    b3b = ax1.bar(x + 0.3, res_s3b["times"], width=width, label='S3B', color='#d62728')
    
    add_labels(b1, ax1, '{:.1f}', rotation=90)
    add_labels(b2, ax1, '{:.1f}', rotation=90)
    add_labels(b3a, ax1, '{:.1f}', rotation=90)
    add_labels(b3b, ax1, '{:.1f}', rotation=90)
    
    all_times = res_s1["times"] + res_s2["times"] + res_s3a["times"] + res_s3b["times"]
    ax1.set_ylim(0, max(all_times) * 1.4)
    ax1.set_title('Komparasi Waktu Eksekusi (Skala Linear)', fontweight='bold')
    ax1.set_ylabel('Waktu (milidetik)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_algo)
    ax1.legend()
    fig1.tight_layout()

    # --- GRAFIK 2: BEBAN MEMORI RAM ---
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    fig2.canvas.manager.set_window_title('Grafik 2 - Beban Memori RAM')
    
    m1 = ax2.bar(x - 0.3, res_s1["mems"], width=width, label='S1', color='#2ca02c')
    m2 = ax2.bar(x - 0.1, res_s2["mems"], width=width, label='S2', color='#1f77b4')
    m3a = ax2.bar(x + 0.1, res_s3a["mems"], width=width, label='S3A', color='#ff7f0e')
    m3b = ax2.bar(x + 0.3, res_s3b["mems"], width=width, label='S3B', color='#d62728')
    
    add_labels(m1, ax2, '{:,.0f}', rotation=90)
    add_labels(m2, ax2, '{:,.0f}', rotation=90)
    add_labels(m3a, ax2, '{:,.0f}', rotation=90)
    add_labels(m3b, ax2, '{:,.0f}', rotation=90)
    
    all_mems = res_s1["mems"] + res_s2["mems"] + res_s3a["mems"] + res_s3b["mems"]
    ax2.set_ylim(0, max(all_mems) * 1.4)
    ax2.set_title('Komparasi Puncak Alokasi Memori (Skala Linear)', fontweight='bold')
    ax2.set_ylabel('Memori (Kilobytes)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_algo)
    ax2.legend()
    fig2.tight_layout()

    # --- GRAFIK 3: TOTAL VALUE ---
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    fig3.canvas.manager.set_window_title('Grafik 3 - Total Value')
    
    v1 = ax3.bar(x - 0.3, res_s1["values"], width=width, label='S1', color='#2ca02c')
    v2 = ax3.bar(x - 0.1, res_s2["values"], width=width, label='S2', color='#1f77b4')
    v3a = ax3.bar(x + 0.1, res_s3a["values"], width=width, label='S3A', color='#ff7f0e')
    v3b = ax3.bar(x + 0.3, res_s3b["values"], width=width, label='S3B', color='#d62728')
    
    add_labels(v1, ax3, '{:.0f}')
    add_labels(v2, ax3, '{:.0f}')
    add_labels(v3a, ax3, '{:.0f}')
    add_labels(v3b, ax3, '{:.0f}')

    # Efek Zoom Y-Axis untuk Value
    all_vals = res_s1["values"] + res_s2["values"] + res_s3a["values"] + res_s3b["values"]
    ax3.set_ylim(min(all_vals) * 0.9, max(all_vals) * 1.2)
    
    ax3.set_title('Komparasi Total Value Tas', fontweight='bold')
    ax3.set_ylabel('Skor Utilitas')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels_algo)
    ax3.legend(loc='lower right')
    fig3.tight_layout()

    # --- GRAFIK 4: KOMPOSISI ITEM (STACKED BAR) ---
    fig4, ax4 = plt.subplots(figsize=(14, 6))
    fig4.canvas.manager.set_window_title('Grafik 4 - Komposisi Item')
    
    bar_labels = []
    all_cats = []
    for s_name, data in zip(['S1', 'S2', 'S3A', 'S3B'], [res_s1, res_s2, res_s3a, res_s3b]):
        for algo in labels_algo:
            bar_labels.append(f"{s_name}-{algo}")
        all_cats.extend(data["categories"])

    medis = [cat["Medis"] for cat in all_cats]
    amunisi = [cat["Amunisi"] for cat in all_cats]
    throwables = [cat["Throwables"] for cat in all_cats]

    ax4.bar(bar_labels, medis, label='Medis', color='#ff9999')
    ax4.bar(bar_labels, amunisi, bottom=medis, label='Amunisi', color='#66b3ff')
    ax4.bar(bar_labels, throwables, bottom=[i+j for i,j in zip(medis, amunisi)], label='Throwables', color='#99ff99')
    
    ax4.set_title('Distribusi Komposisi Isi Tas (Kategori Barang)', fontweight='bold')
    ax4.set_ylabel('Total Kuantitas Barang')
    ax4.tick_params(axis='x', rotation=45)
    ax4.legend(loc='upper right')
    fig4.tight_layout()

    # --- FUNGSI HELPER PIE CHART ---
    def hitung_peluru(raw_items):
        c_556 = sum(1 for item in raw_items if item['name'] == 'Amunisi 5.56mm')
        c_762 = sum(1 for item in raw_items if item['name'] == 'Amunisi 7.62mm')
        return [c_556, c_762]

    labels_pie = ['5.56mm', '7.62mm']
    colors_pie = ['#2ca02c', '#8b4513']

    # --- GRAFIK 5: PIE CHART SKENARIO 3A ---
    fig5, axs5 = plt.subplots(1, 5, figsize=(18, 4))
    fig5.canvas.manager.set_window_title('Grafik 5 - Rasio Peluru 3A')
    fig5.suptitle('Rasio Pengambilan Amunisi Skenario 3A (AR 5.56 Utama)', fontweight='bold')
    
    for i, ax in enumerate(axs5):
        data_ammo = hitung_peluru(res_s3a["raw_items"][i])
        if sum(data_ammo) > 0:
            ax.pie(data_ammo, labels=labels_pie, autopct='%1.1f%%', colors=colors_pie, startangle=90)
        else:
            ax.text(0.5, 0.5, "0 Peluru", ha='center', va='center')
        ax.set_title(labels_algo[i])
    fig5.tight_layout()

    # --- GRAFIK 6: PIE CHART SKENARIO 3B ---
    fig6, axs6 = plt.subplots(1, 5, figsize=(18, 4))
    fig6.canvas.manager.set_window_title('Grafik 6 - Rasio Peluru 3B')
    fig6.suptitle('Rasio Pengambilan Amunisi Skenario 3B (DMR 7.62 Utama)', fontweight='bold')
    
    for i, ax in enumerate(axs6):
        data_ammo = hitung_peluru(res_s3b["raw_items"][i])
        if sum(data_ammo) > 0:
            ax.pie(data_ammo, labels=labels_pie, autopct='%1.1f%%', colors=colors_pie, startangle=90)
        else:
            ax.text(0.5, 0.5, "0 Peluru", ha='center', va='center')
        ax.set_title(labels_algo[i])
    fig6.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()