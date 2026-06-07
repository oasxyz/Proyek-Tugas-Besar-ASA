import random
import time
import matplotlib.pyplot as plt

# Variabel konstanta untuk mendefinisikan kapasitas maksimal ransel (Knapsack capacity)
MAX_CAPACITY = 400

# base_items: List of dictionaries yang mendefinisikan atribut dasar setiap item.
# Atribut meliputi nama item, bobot (weight) yang memakan kapasitas, dan nilai utilitas (value).
base_items = [
    {"name": "First Aid Kit", "weight": 10, "value": 50},
    {"name": "Medkit", "weight": 20, "value": 40},
    {"name": "Bandage (5x)", "weight": 10, "value": 40},
    {"name": "Energy Drink", "weight": 4, "value": 35},
    {"name": "Painkiller", "weight": 10, "value": 30},
    {"name": "Smoke Grenade", "weight": 14, "value": 35},
    {"name": "Frag Grenade", "weight": 18, "value": 35},
    {"name": "Amunisi 5.56mm", "weight": 15, "value": 35},
    {"name": "Amunisi 7.62mm", "weight": 21, "value": 35} 
]

# item_counts: Array integer yang merepresentasikan kuantitas ketersediaan untuk setiap item di base_items.
# Digunakan untuk mereplikasi skenario Bounded Knapsack Problem.
item_counts = [3, 2, 3, 5, 5, 5, 5, 8, 8]

# KATEGORI_ITEM: Dictionary untuk pemetaan (mapping) nama item ke kategori faksi.
# Berfungsi sebagai metadata tambahan untuk klasifikasi data pada visualisasi grafik komposisi.
KATEGORI_ITEM = {
    "First Aid Kit": "Medis", "Medkit": "Medis", "Bandage (5x)": "Medis",
    "Energy Drink": "Medis", "Painkiller": "Medis",
    "Smoke Grenade": "Throwables", "Frag Grenade": "Throwables",
    "Amunisi 5.56mm": "Amunisi", "Amunisi 7.62mm": "Amunisi"
}


# --- IMPLEMENTASI ALGORITMA ---

def knapsack_dp(capacity, items):
    """
    Menyelesaikan Knapsack Problem menggunakan pendekatan Dynamic Programming (Bottom-Up).
    Membangun tabel 2D untuk menyimpan sub-masalah yang tumpang tindih guna menjamin solusi paling optimal.
    """
    n = len(items)
    # Inisialisasi matriks dp_table berukuran (n+1) x (capacity+1) dengan nilai 0
    dp_table = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
    
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            weight = items[i-1]['weight']
            value = items[i-1]['value']
            
            # Evaluasi state: membandingkan nilai maksimum antara memasukkan item ke-i atau tidak memasukkannya
            if weight <= w:
                dp_table[i][w] = max(value + dp_table[i-1][w - weight], dp_table[i-1][w])
            else:
                dp_table[i][w] = dp_table[i-1][w]
                
    # Proses Backtracking: Menelusuri dp_table dari indeks terbawah untuk mengekstraksi list item yang terpilih
    selected_items = []
    w = capacity
    for i in range(n, 0, -1):
        if dp_table[i][w] != dp_table[i-1][w]:
            selected_items.append(items[i-1])
            w -= items[i-1]['weight']
            
    return dp_table[n][capacity], selected_items


def knapsack_greedy(capacity, items):
    """
    Menyelesaikan Knapsack Problem menggunakan pendekatan heuristik Greedy.
    Memprioritaskan item berdasarkan rasio utilitas terhadap bobot (Value-to-Weight ratio).
    """
    # Mengurutkan array items secara descending berdasarkan rasio (value/weight)
    sorted_items = sorted(items, key=lambda x: x['value'] / x['weight'], reverse=True)
    
    total_value = 0
    current_weight = 0
    selected_items = []
    
    # Iterasi linear untuk memasukkan item selama kapasitas current_weight masih mencukupi
    for item in sorted_items:
        if current_weight + item['weight'] <= capacity:
            selected_items.append(item)
            total_value += item['value']
            current_weight += item['weight']
            
    return total_value, selected_items


def knapsack_ga(capacity, items, pop_size=50, generations=100, mutation_rate=0.1):
    """
    Menyelesaikan Knapsack Problem menggunakan Genetic Algorithm (GA).
    Menerapkan proses evolusioner: inisialisasi populasi, evaluasi fitness, seleksi, crossover, dan mutasi.
    """
    n = len(items)
    
    # Fungsi objektif (Fitness Function): Mengkalkulasi nilai total kromosom.
    # Mengembalikan penalti (0) jika total bobot melebihi MAX_CAPACITY.
    def fitness(chromosome):
        total_weight = sum(chromosome[i] * items[i]['weight'] for i in range(n))
        total_value = sum(chromosome[i] * items[i]['value'] for i in range(n))
        if total_weight > capacity:
            return 0  
        return total_value

    # Inisialisasi populasi awal yang terdiri dari array biner acak (0 atau 1)
    population = [[random.choice([0, 1]) for _ in range(n)] for _ in range(pop_size)]
    
    for _ in range(generations):
        # Proses Seleksi: Mengurutkan kromosom berdasarkan nilai fitness tertinggi
        population = sorted(population, key=lambda x: fitness(x), reverse=True)
        # Elitism: Mengamankan dua individu terbaik langsung ke generasi berikutnya
        next_gen = population[:2] 
        
        while len(next_gen) < pop_size:
            # Seleksi indukan (Parent selection) dari top 10 individu
            parent1, parent2 = random.choices(population[:10], k=2)
            
            # Crossover (Kawin silang): Penggabungan genetik menggunakan satu titik potong (single-point crossover)
            crossover_point = random.randint(1, n-1)
            child = parent1[:crossover_point] + parent2[crossover_point:]
            
            # Mutasi (Mutation): Pembalikan nilai bit (bit-flip) berdasarkan probabilitas mutation_rate
            for i in range(n):
                if random.random() < mutation_rate:
                    child[i] = 1 - child[i]
            next_gen.append(child)
            
        population = next_gen
        
    # Ekstraksi solusi terbaik dari populasi final
    best_chromosome = max(population, key=fitness)
    selected_items = [items[i] for i in range(n) if best_chromosome[i] == 1]
    
    return fitness(best_chromosome), selected_items


# --- FUNGSI UTILITAS PENGOLAHAN DATA ---

def rekap_kategori(selected_items):
    """
    Mengelompokkan array selected_items ke dalam struktur dictionary berdasarkan kategori (Medis, Amunisi, Throwables).
    """
    rekap = {"Medis": 0, "Amunisi": 0, "Throwables": 0}
    for item in selected_items:
        rekap[KATEGORI_ITEM[item['name']]] += 1
    return rekap


def hitung_bobot(selected_items):
    """
    Fungsi akumulator untuk mengembalikan total bobot (weight) dari item yang terpilih.
    """
    return sum(item['weight'] for item in selected_items)


def run_scenario(scenario_name, m_556, m_762, capacity=MAX_CAPACITY):
    """
    Fungsi eksekutor untuk menjalankan satu sesi skenario penuh.
    Membuat pool item dinamis berdasarkan multiplier sinergi dan merekam metrik kinerja ketiga algoritma.
    """
    print(f"\n{'='*60}\n--- {scenario_name} ---\n{'='*60}")
    
    # Generasi kumpulan item (current_pool) dengan mengaplikasikan parameter multiplier
    current_pool = []
    for item, count in zip(base_items, item_counts):
        for _ in range(count):
            new_item = item.copy()
            if new_item['name'] == "Amunisi 5.56mm":
                new_item['value'] *= m_556
            elif new_item['name'] == "Amunisi 7.62mm":
                new_item['value'] *= m_762
            
            # Eliminasi item dengan nilai 0 dari kumpulan data (garbage collector manual)
            if new_item['value'] > 0:
                current_pool.append(new_item)

    # Logging eksekusi dan pencatatan metrik waktu untuk algoritma Dynamic Programming
    start = time.perf_counter()
    dp_val, dp_items = knapsack_dp(capacity, current_pool)
    dp_time = (time.perf_counter() - start) * 1000
    
    # Logging eksekusi dan pencatatan metrik waktu untuk algoritma Greedy
    start = time.perf_counter()
    gr_val, gr_items = knapsack_greedy(capacity, current_pool)
    gr_time = (time.perf_counter() - start) * 1000
    
    # Logging eksekusi dan pencatatan metrik waktu untuk algoritma GA
    start = time.perf_counter()
    ga_val, ga_items = knapsack_ga(capacity, current_pool)
    ga_time = (time.perf_counter() - start) * 1000
    
    print(f"[DP]     Value: {dp_val:.1f} | Bobot: {hitung_bobot(dp_items)}/{capacity} | Time: {dp_time:.4f} ms")
    print(f"[Greedy] Value: {gr_val:.1f} | Bobot: {hitung_bobot(gr_items)}/{capacity} | Time: {gr_time:.4f} ms")
    print(f"[GA]     Value: {ga_val:.1f} | Bobot: {hitung_bobot(ga_items)}/{capacity} | Time: {ga_time:.4f} ms")
    
    # Mengembalikan dictionary komprehensif berisi array data untuk render grafik matplotlib
    return {
        "values": [dp_val, gr_val, ga_val],
        "times": [dp_time, gr_time, ga_time],
        "weights": [hitung_bobot(dp_items), hitung_bobot(gr_items), hitung_bobot(ga_items)],
        "categories": [rekap_kategori(dp_items), rekap_kategori(gr_items), rekap_kategori(ga_items)],
        "raw_items": [dp_items, gr_items, ga_items] 
    }


# --- FUNGSI VISUALISASI MATPLOTLIB ---

def add_labels(bars, ax, fmt='{:.2f}', offset=2):
    """
    Fungsi helper grafis untuk menyematkan teks (label angka) tepat di atas komponen bar chart.
    """
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + offset, fmt.format(yval), 
                ha='center', va='bottom', fontsize=9, fontweight='bold', color='black')


def main():
    print("=== EKSPERIMEN 3 SKENARIO KNAPSACK: METADATA PUBG ===")
    
    # Eksekusi blok fungsi run_scenario untuk menguji 3 variabel independen
    res_s1 = run_scenario("SKENARIO 1: Full 5.56mm (M416 + Mini14)", 2.0, 0.0)
    res_s2 = run_scenario("SKENARIO 2: Full 7.62mm (AKM + SKS)", 0.0, 2.0)
    res_s3 = run_scenario("SKENARIO 3: Hybrid (M416 + SKS)", 1.8, 1.2)

    print("\nMenyiapkan 5 Grafik Komparasi... (Silakan cek window pop-up)")
    
    # Deklarasi variabel konfigurasi koordinat dan penempatan untuk plot Matplotlib
    labels_algo = ['DP', 'Greedy', 'GA']
    x = range(len(labels_algo))
    width = 0.25

    # ================= GRAFIK 1: WAKTU EKSEKUSI =================
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    fig1.canvas.manager.set_window_title('Grafik 1 - Waktu Eksekusi')
    
    b1 = ax1.bar([i - width for i in x], res_s1["times"], width=width, label='S1: 5.56mm', color='#2ca02c')
    b2 = ax1.bar(x, res_s2["times"], width=width, label='S2: 7.62mm', color='#1f77b4')
    b3 = ax1.bar([i + width for i in x], res_s3["times"], width=width, label='S3: Hybrid', color='#ff7f0e')
    
    add_labels(b1, ax1, '{:.2f} ms')
    add_labels(b2, ax1, '{:.2f} ms')
    add_labels(b3, ax1, '{:.2f} ms')
    
    # Modifikasi batas atas sumbu Y (Y-lim) untuk menghindari tabrakan antara teks label dan frame atas
    ax1.set_ylim(0, max(max(res_s1["times"]), max(res_s2["times"]), max(res_s3["times"])) * 1.15)
    ax1.set_title('Komparasi Waktu Eksekusi Algoritma (Makin Rendah Makin Baik)', fontweight='bold')
    ax1.set_ylabel('Waktu (milidetik)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_algo)
    ax1.legend()
    fig1.tight_layout()

    # ================= GRAFIK 2: TOTAL VALUE (ZOOM EFFECT) =================
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    fig2.canvas.manager.set_window_title('Grafik 2 - Total Value')
    
    b1_v = ax2.bar([i - width for i in x], res_s1["values"], width=width, label='S1: 5.56mm', color='#2ca02c')
    b2_v = ax2.bar(x, res_s2["values"], width=width, label='S2: 7.62mm', color='#1f77b4')
    b3_v = ax2.bar([i + width for i in x], res_s3["values"], width=width, label='S3: Hybrid', color='#ff7f0e')
    
    add_labels(b1_v, ax2, '{:.0f}')
    add_labels(b2_v, ax2, '{:.0f}')
    add_labels(b3_v, ax2, '{:.0f}')

    # Ekstraksi rentang data aktual untuk mempersempit skala sumbu Y guna menonjolkan deltabar (Zooming Y-axis)
    all_vals = res_s1["values"] + res_s2["values"] + res_s3["values"]
    min_val = min(all_vals)
    max_val = max(all_vals)
    ax2.set_ylim(min_val - (min_val * 0.1), max_val + (max_val * 0.05)) 
    
    ax2.set_title('Komparasi Total Value Tas (Dengan Efek Zoom Sumbu Y)', fontweight='bold')
    ax2.set_ylabel('Skor Utilitas')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_algo)
    ax2.legend()
    fig2.tight_layout()

    # ================= GRAFIK 3: KOMPOSISI ITEM (STACKED BAR) =================
    fig3, ax3 = plt.subplots(figsize=(12, 6))
    fig3.canvas.manager.set_window_title('Grafik 3 - Komposisi Item')
    bar_labels = ['S1-DP', 'S1-Gr', 'S1-GA', 'S2-DP', 'S2-Gr', 'S2-GA', 'S3-DP', 'S3-Gr', 'S3-GA']
    
    # Agregasi data matrix ke dalam array linear per kategori
    all_cats = res_s1["categories"] + res_s2["categories"] + res_s3["categories"]
    medis = [cat["Medis"] for cat in all_cats]
    amunisi = [cat["Amunisi"] for cat in all_cats]
    throwables = [cat["Throwables"] for cat in all_cats]

    # Eksekusi plot stacked bar dengan kalkulasi offset base (bottom) untuk layer atasnya
    ax3.bar(bar_labels, medis, label='Medis & Booster', color='#ff9999')
    ax3.bar(bar_labels, amunisi, bottom=medis, label='Amunisi', color='#66b3ff')
    ax3.bar(bar_labels, throwables, bottom=[i+j for i,j in zip(medis, amunisi)], label='Throwables', color='#99ff99')
    
    ax3.set_title('Distribusi Komposisi Isi Tas (Kategori Barang)', fontweight='bold')
    ax3.set_ylabel('Total Jumlah Barang')
    ax3.legend(loc='upper left')
    
    # Penambahan text box aksesoris sebagai legend tambahan untuk singkatan metode
    kamus_text = "KAMUS SINGKATAN:\nS = Skenario\nDP = Dynamic Programming\nGr = Greedy Algorithm\nGA = Genetic Algorithm"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax3.text(1.02, 0.5, kamus_text, transform=ax3.transAxes, fontsize=10, verticalalignment='center', bbox=props)
    fig3.tight_layout(rect=[0, 0, 0.85, 1]) 

    # ================= GRAFIK 4: UTILISASI BOBOT TAS =================
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    fig4.canvas.manager.set_window_title('Grafik 4 - Penggunaan Kapasitas')
    
    b1_w = ax4.bar([i - width for i in x], res_s1["weights"], width=width, label='S1: 5.56mm', color='#2ca02c')
    b2_w = ax4.bar(x, res_s2["weights"], width=width, label='S2: 7.62mm', color='#1f77b4')
    b3_w = ax4.bar([i + width for i in x], res_s3["weights"], width=width, label='S3: Hybrid', color='#ff7f0e')
    
    add_labels(b1_w, ax4, '{:.0f}/400')
    add_labels(b2_w, ax4, '{:.0f}/400')
    add_labels(b3_w, ax4, '{:.0f}/400')
    
    # Render indikator garis horizontal untuk merepresentasikan konstanta MAX_CAPACITY
    ax4.axhline(y=MAX_CAPACITY, color='r', linestyle='--', label='Kapasitas Maksimal (400)')
    ax4.set_ylim(min(min(res_s1["weights"]), min(res_s2["weights"]), min(res_s3["weights"])) - 20, 430)
    
    ax4.set_title('Tingkat Efisiensi Penggunaan Ruang Tas Level 3', fontweight='bold')
    ax4.set_ylabel('Total Bobot Digunakan')
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels_algo)
    ax4.legend(loc='lower right')
    fig4.tight_layout()

    # ================= GRAFIK 5: RASIO PELURU KHUSUS SKENARIO 3 =================
    fig5, (ax_dp, ax_gr, ax_ga) = plt.subplots(1, 3, figsize=(12, 5))
    fig5.canvas.manager.set_window_title('Grafik 5 - Rasio Peluru Skenario 3')
    
    def hitung_peluru(raw_items):
        """Fungsi ekstraktor khusus untuk memisahkan kuantitas peluru 5.56mm vs 7.62mm"""
        c_556 = sum(1 for item in raw_items if item['name'] == 'Amunisi 5.56mm')
        c_762 = sum(1 for item in raw_items if item['name'] == 'Amunisi 7.62mm')
        return [c_556, c_762]

    labels_pie = ['5.56mm (AR Utama)', '7.62mm (DMR Sekunder)']
    colors_pie = ['#2ca02c', '#8b4513']

    # Komparasi distribusi rasio dalam format Pie Chart
    data_dp = hitung_peluru(res_s3["raw_items"][0])
    ax_dp.pie(data_dp, labels=labels_pie, autopct='%1.1f%%', colors=colors_pie, startangle=90)
    ax_dp.set_title('Keputusan DP')

    data_gr = hitung_peluru(res_s3["raw_items"][1])
    ax_gr.pie(data_gr, labels=labels_pie, autopct='%1.1f%%', colors=colors_pie, startangle=90)
    ax_gr.set_title('Keputusan Greedy')

    data_ga = hitung_peluru(res_s3["raw_items"][2])
    ax_ga.pie(data_ga, labels=labels_pie, autopct='%1.1f%%', colors=colors_pie, startangle=90)
    ax_ga.set_title('Keputusan GA')

    fig5.suptitle('Rasio Pengambilan Amunisi pada Skenario 3 (Hibrida)', fontweight='bold', fontsize=14)
    fig5.tight_layout()

    # Memicu engine Matplotlib untuk me-render antarmuka grafis
    plt.show()

if __name__ == "__main__":
    main()