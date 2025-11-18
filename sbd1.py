import os
import random
import heapq
import math
import matplotlib.pyplot as plt
from typing import List, Set, Tuple, Optional, IO
from datetime import datetime

class Record:
    """Rekord zawierający zbiór liczb"""
    def __init__(self, numbers: Set[int]):
        self.numbers = numbers
        self.sum = sum(numbers)
    
    # Porównywanie rekordów na podstawie sumy
    def __lt__(self, other):
        return self.sum < other.sum
    
    # Reprezentacja rekordu jako string
    def __str__(self):
        # Zwraca reprezentację zbioru jako posortowany, rozdzielony przecinkami string.
        # Kolejność operacji:
        # 1) sorted(self.numbers) - sortuje elementy zbioru,
        # 2) map(str, ...) - konwertuje każdą liczbę na string,
        # 3) ','.join(...) - łączy elementy przecinkiem w jeden ciąg,
        # 4) f"{{...}}" - umieszcza rezultat w nawiasach klamrowych.
        return f"{{{','.join(map(str, sorted(self.numbers)))}}}"
    
    def to_string(self, max_length: int) -> str:
        """Konwertuje rekord na string o stałej długości z paddingiem"""
        # ljust dodaje znaki '\0' aż do długości max_length – przydatne do zapisu stałych pól binarnych.
        s = str(self)
        return s.ljust(max_length, '\0')
    
    @staticmethod
    def from_string(s: str) -> 'Record':
        """Tworzy rekord ze stringa"""
        # Usuń padding '\0' z końca; pusty lub '{}' oznacza brak rekordu.
        s = s.rstrip('\0')
        if not s or s == '{}':
            return None
        # Split po przecinkach i konwersja na int tworzy zbiór liczb.
        numbers = set(map(int, s.strip('{}').split(',')))
        return Record(numbers)

class DiskSimulator:
    """Symulacja operacji dyskowych z blokowaniem"""
    def __init__(self, filename: str, block_size: int, record_size: int):
        self.filename = filename
        self.block_size = block_size  # liczba rekordów na stronę (b)
        self.record_size = record_size  # rozmiar rekordu w bajtach
        self.page_size = block_size * record_size  # rozmiar strony w bajtach (B)
        self.read_count = 0
        self.write_count = 0
        
    def read_page(self, page_num: int,count_operation: bool = True) -> List[Record]:
        """Odczytuje stronę z dysku"""
        records = []
        
        try:
            with open(self.filename, 'rb') as f:
                # Seek do początku żądanej strony (page_num * page_size)
                f.seek(page_num * self.page_size)
                data = f.read(self.page_size)
                
                for i in range(self.block_size):
                    # Obliczamy przesunięcie bajtowe dla i-tego rekordu na stronie:
                    # start = i * record_size, end = start + record_size
                    start = i * self.record_size
                    end = start + self.record_size
                    if start < len(data):
                        # Dekodujemy fragment bajtów do stringa.
                        # errors='ignore' pomija niepoprawne bajty zamiast podnosić wyjątek.
                        record_str = data[start:end].decode('utf-8', errors='ignore')
                        record = Record.from_string(record_str)
                        if record:
                            records.append(record)
        except FileNotFoundError:
            # Brak pliku oznacza brak stron do odczytu.
            pass

        # Licznik zwiększamy TYLKO gdy faktycznie odczytano dane i count_operation=True
        if count_operation and records:
            self.read_count += 1

        return records
    
    def write_page(self, page_num: int, records: List[Record]):
        """Zapisuje stronę na dysk"""
        self.write_count += 1
        
        # Przygotuj dane do zapisu z paddingiem (każdy rekord ma stałą liczbę bajtów)
        data = b''
        for i in range(self.block_size):
            if i < len(records):
                # to_string już zwraca string wypełniony '\0' do record_size
                record_str = records[i].to_string(self.record_size)
            else:
                # jeśli brak rekordu, zapisz sam padding na to miejsce
                record_str = '\0' * self.record_size  # padding
            # Kodujemy cały rekord (wraz z paddingiem) do bajtów
            data += record_str.encode('utf-8')
        
        # Upewnij się, że katalog istnieje; tryb pliku zależy od istnienia pliku (append/overwrite)
        os.makedirs(os.path.dirname(self.filename) if os.path.dirname(self.filename) else '.', exist_ok=True)
        mode = 'r+b' if os.path.exists(self.filename) else 'wb'
        
        with open(self.filename, mode) as f:
            # Seek do odpowiedniej strony i zapisz dane
            f.seek(page_num * self.page_size)
            f.write(data)
    
    def get_file_size_in_pages(self) -> int:
        """Zwraca rozmiar pliku w stronach"""
        if not os.path.exists(self.filename):
            return 0
        size_bytes = os.path.getsize(self.filename)
        return math.ceil(size_bytes / self.page_size)
    
    def reset_counters(self):
        """Resetuje liczniki operacji dyskowych"""
        self.read_count = 0
        self.write_count = 0

class RunInfo:
    """Informacje o położeniu serii w pliku"""
    def __init__(self, start_page: int, end_page: int):
        self.start_page = start_page
        self.end_page = end_page
        self.length = end_page - start_page

class LargeBufferSort:
    """Sortowanie z użyciem wielkich buforów"""
    def __init__(self, input_file: str, n_buffers: int, block_size: int, record_size: int):
        self.input_file = input_file
        self.n_buffers = n_buffers # liczba buforów (n)
        self.block_size = block_size # liczba rekordów na stronę (b)
        self.record_size = record_size # rozmiar rekordu w bajtach
        self.disk_sim = DiskSimulator(input_file, block_size, record_size)
        self.phase_count = 0
        
    def sort(self, show_phases: bool = False):
        """
        Sortuje plik używając algorytmu z dużymi buforami

        Etap 1 (tworzenie serii)
        1. Wczytaj pierwsze n*b rekordów do buforów i posortuj je
        2. Zapisz serię na dysku (nadpisując plik wejściowy)
        3. Powtarzaj, aż do końca pliku

        Etap 2 (scalanie)
        4. Scal pierwsze n-1 serii używając n-tego bufora jako bufora wyjściowego
        5. Powtarzaj krok 4 dla kolejnych grup serii, aż do końca pliku
        6. Powtarzaj kroki 4 i 5 aż pozostanie jedna seria
        """
        print(f"\n=== Sortowanie z {self.n_buffers} buforami (b={self.block_size}) ===")
        
        # Przygotuj plik logu jeśli pokazujemy fazy
        phase_fh: Optional[IO] = None
        if show_phases:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = os.path.splitext(self.input_file)[0]
            logname = f"{base}_phases_{timestamp}.txt"
            phase_fh = open(logname, 'w', encoding='utf-8')
            phase_fh.write(f"Log faz sortowania dla pliku: {self.input_file}\n")
            phase_fh.write("="*60 + "\n\n")
            print(f"Fazy będą zapisywane do: {logname}")
        
        # Stage 1: Creating runs
        print("\nEtap 1: Tworzenie posortowanych serii...")
        runs = self._create_runs()
        print(f"Utworzono {len(runs)} serii początkowych")
        
        if show_phases:
            # Zapisz i wyświetl serie początkowe
            phase_fh.write("Serie początkowe:\n")
            for i, run in enumerate(runs):
                print(f"Seria {i+1} (strony {run.start_page}-{run.end_page}):")
                phase_fh.write(f"Seria {i+1} (strony {run.start_page}-{run.end_page}):\n")
                self._display_run(run, phase_fh)
                phase_fh.write("\n")
        
        # Stage 2: Merging
        print("\nEtap 2: Scalanie serii...")
        self.phase_count = 0
        self._merge_runs(runs, show_phases, phase_fh)
        
        # Zamknij plik logu jeśli był otwarty
        if phase_fh:
            phase_fh.write("\nKoniec logu faz.\n")
            phase_fh.close()
        
        print(f"\nLiczba faz scalania: {self.phase_count}")
        print(f"Liczba odczytów stron: {self.disk_sim.read_count}")
        print(f"Liczba zapisów stron: {self.disk_sim.write_count}")
        print(f"Łączna liczba operacji dyskowych: {self.disk_sim.read_count + self.disk_sim.write_count}")
    
    def _create_runs(self) -> List[RunInfo]:
        """
        Etap 1: Tworzy posortowane serie z natychmiastowym nadpisywaniem
        Po każdym wczytaniu jednej serii (n*b rekordów), sortuje i OD RAZU nadpisuje w pliku
        """
        runs = []
        input_page = 0
        total_pages = self.disk_sim.get_file_size_in_pages()
        
        while input_page < total_pages:
            # Wczytaj JEDNĄ serię (n stron = n*b rekordów)
            records = []
            start_page = input_page
            
            for i in range(self.n_buffers):
                if input_page >= total_pages:
                    break
                page_records = self.disk_sim.read_page(input_page)
                if not page_records:
                    break
                records.extend(page_records)
                input_page += 1
            
            if not records:
                break
            
            # Sortuj rekordy w pamięci
            records.sort()
            
            # NATYCHMIAST otwórz plik i nadpisz tę serię w miejscu nieposortowanej
            output_page = start_page
            with open(self.input_file, 'r+b') as f:
                for i in range(0, len(records), self.block_size):
                    page_records = records[i:i + self.block_size]
                    
                    # Przygotuj dane do zapisu
                    data = b''
                    for j in range(self.block_size):
                        if j < len(page_records):
                            record_str = page_records[j].to_string(self.record_size)
                        else:
                            record_str = '\0' * self.record_size
                        data += record_str.encode('utf-8')
                    
                    # Nadpisz w pliku
                    f.seek(output_page * self.disk_sim.page_size)
                    f.write(data)
                    self.disk_sim.write_count += 1
                    output_page += 1
            # Plik zamknięty automatycznie po wyjściu z bloku with
            
            runs.append(RunInfo(start_page, output_page))
        
        return runs
    
    def _merge_runs(self, runs: List[RunInfo], show_phases: bool, fh: Optional[IO] = None):
        """
        Etap 2: Scala serie z natychmiastowym nadpisywaniem
        Po scaleniu grupy serii, natychmiast nadpisuje wynik od początku pierwszej serii
        """
        current_runs = runs
        
        while len(current_runs) > 1:
            self.phase_count += 1
            phase_msg = f"\nFaza scalania {self.phase_count}: Scalanie {len(current_runs)} serii..."
            print(phase_msg)
            if show_phases and fh:
                fh.write(phase_msg + "\n")
            
            next_runs = []
            
            # Scal grupy po (n-1) serii
            for i in range(0, len(current_runs), self.n_buffers - 1):
                runs_to_merge = current_runs[i:i + self.n_buffers - 1]
                
                # Wyznacz miejsce zapisu - od początku pierwszej serii w grupie
                output_start_page = runs_to_merge[0].start_page
                
                # Scal i natychmiast zapisz
                output_end_page = self._merge_multiple_runs(runs_to_merge, output_start_page)
                next_runs.append(RunInfo(output_start_page, output_end_page))
            
            current_runs = next_runs
            
            if show_phases and fh:
                fh.write(f"\nPo fazie {self.phase_count}:\n")
                for i, run in enumerate(current_runs):
                    fh.write(f"Seria {i+1} (strony {run.start_page}-{run.end_page}):\n")
                    self._display_run(run, fh)
                    fh.write("\n")
            
            if show_phases:
                print(f"\nPo fazie {self.phase_count}:")
                for i, run in enumerate(current_runs):
                    print(f"Seria {i+1} (strony {run.start_page}-{run.end_page}):")
                    self._display_run(run, fh if show_phases else None)
    
    def _merge_multiple_runs(self, runs: List[RunInfo], output_start_page: int) -> int:
        """
        Scala wiele serii i NATYCHMIAST zapisuje w miejsce scalanych serii
        Otwiera plik raz, zapisuje wszystkie strony, zamyka
        """
        # Inicjalizacja buforów wejściowych
        input_buffers = []
        page_indices = []
        
        for run in runs:
            page_indices.append(run.start_page)
            page_records = self.disk_sim.read_page(run.start_page)
            input_buffers.append(page_records)
        
        # Bufor wyjściowy
        output_buffer = []
        output_page = output_start_page
        
        # Kolejka priorytetowa
        heap = []
        for i, buffer in enumerate(input_buffers):
            if buffer:
                heapq.heappush(heap, (buffer[0], i, 0))
        
        # Otwórz plik JEDEN RAZ do zapisu scalonych danych
        with open(self.input_file, 'r+b') as f:
            # Scalanie
            while heap:
                record, buffer_idx, pos_in_buffer = heapq.heappop(heap)
                
                output_buffer.append(record)
                
                # Gdy bufor wyjściowy jest pełny, zapisz stronę
                if len(output_buffer) >= self.block_size:
                    # Przygotuj dane
                    data = b''
                    for j in range(self.block_size):
                        if j < len(output_buffer):
                            record_str = output_buffer[j].to_string(self.record_size)
                        else:
                            record_str = '\0' * self.record_size
                        data += record_str.encode('utf-8')
                    
                    # Zapisz do pliku
                    f.seek(output_page * self.disk_sim.page_size)
                    f.write(data)
                    self.disk_sim.write_count += 1
                    output_page += 1
                    output_buffer = []
                
                # Weź następny rekord
                next_pos = pos_in_buffer + 1
                if next_pos < len(input_buffers[buffer_idx]):
                    heapq.heappush(heap, (input_buffers[buffer_idx][next_pos], buffer_idx, next_pos))
                else:
                    page_indices[buffer_idx] += 1
                    if page_indices[buffer_idx] < runs[buffer_idx].end_page:
                        next_page = self.disk_sim.read_page(page_indices[buffer_idx])
                        if next_page:
                            input_buffers[buffer_idx] = next_page
                            heapq.heappush(heap, (next_page[0], buffer_idx, 0))
            
            # Zapisz pozostałe rekordy w buforze wyjściowym
            if output_buffer:
                data = b''
                for j in range(self.block_size):
                    if j < len(output_buffer):
                        record_str = output_buffer[j].to_string(self.record_size)
                    else:
                        record_str = '\0' * self.record_size
                    data += record_str.encode('utf-8')
                
                f.seek(output_page * self.disk_sim.page_size)
                f.write(data)
                self.disk_sim.write_count += 1
                output_page += 1
        # Plik zamknięty automatycznie
        
        return output_page
    
    def _display_run(self, run: RunInfo, fh: Optional[IO] = None):
        """Wyświetla zawartość serii; dodatkowo zapisuje do fh gdy podany"""
        all_records = []
        
        for page_num in range(run.start_page, run.end_page):
            records = self.disk_sim.read_page(page_num, count_operation=False)
            all_records.extend(records)
        
        for i, record in enumerate(all_records, 1):
            line = f"  {i}. {record} (suma={record.sum})"
            print(line)
            if fh:
                fh.write(line + "\n")
    
    def _display_file(self, filename: str, fh: Optional[IO] = None):
        """Wyświetla zawartość pliku; dodatkowo zapisuje do fh gdy podany"""
        if not os.path.exists(filename):
            msg = "  (plik nie istnieje)"
            print(msg)
            if fh:
                fh.write(msg + "\n")
            return
        
        disk_sim = DiskSimulator(filename, self.block_size, self.record_size)
        page_num = 0
        all_records = []
        
        while True:
            records = disk_sim.read_page(page_num, count_operation=False)
            if not records:
                break
            all_records.extend(records)
            page_num += 1
        
        for i, record in enumerate(all_records, 1):
            line = f"  {i}. {record} (suma={record.sum})"
            print(line)
            if fh:
                fh.write(line + "\n")

def generate_random_records(n: int, max_numbers: int = 15) -> List[Record]:
    """Generuje losowe rekordy"""
    records = []
    for _ in range(n):
        count = random.randint(1, max_numbers)
        numbers = set(random.randint(1, 100) for _ in range(count))
        records.append(Record(numbers))
    return records

def read_records_from_keyboard() -> List[Record]:
    """Wczytuje rekordy z klawiatury"""
    records = []
    print("\nPodaj rekordy (zbiory liczb, np: 1,2,3). Pusta linia kończy:")
    
    while True:
        line = input("Rekord: ").strip()
        if not line:
            break
        
        try:
            numbers = set(map(int, line.split(',')))
            records.append(Record(numbers))
        except ValueError:
            print("Błędny format! Użyj liczb oddzielonych przecinkami.")
    
    return records

def read_records_from_file(filename: str) -> List[Record]:
    """Wczytuje rekordy z pliku tekstowego"""
    records = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    numbers = set(map(int, line.split(',')))
                    records.append(Record(numbers))
                except ValueError:
                    print(f"Pominięto błędną linię: {line}")
    return records

def save_records_to_disk(records: List[Record], filename: str, block_size: int, record_size: int):
    """Zapisuje rekordy do pliku dyskowego (.dat)"""
    disk_sim = DiskSimulator(filename, block_size, record_size)
    
    for i in range(0, len(records), block_size):
        page_records = records[i:i + block_size]
        disk_sim.write_page(i // block_size, page_records)

def save_records_to_txt(records: List[Record], filename: str):
    """Zapisuje rekordy do pliku tekstowego (.txt)"""
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"Liczba rekordów: {len(records)}\n")
        f.write("="*50 + "\n\n")
        for i, record in enumerate(records, 1):
            f.write(f"{i}. {record} (suma={record.sum})\n")

def display_file_records(filename: str, block_size: int, record_size: int):
    """Wyświetla zawartość pliku"""
    disk_sim = DiskSimulator(filename, block_size, record_size)
    page_num = 0
    record_num = 1
    
    print(f"\n{'='*50}")
    print(f"Zawartość pliku: {filename}")
    print(f"{'='*50}")
    
    while True:
        records = disk_sim.read_page(page_num, count_operation=False)
        if not records:
            break
        
        for record in records:
            print(f"{record_num}. {record} (suma={record.sum})")
            record_num += 1
        
        page_num += 1
    
    print(f"{'='*50}\n")

def export_sorted_to_txt(dat_file: str, txt_file: str, block_size: int, record_size: int):
    """Eksportuje posortowany plik .dat do .txt"""
    disk_sim = DiskSimulator(dat_file, block_size, record_size)
    page_num = 0
    all_records = []
    
    while True:
        records = disk_sim.read_page(page_num, count_operation=False)
        if not records:
            break
        all_records.extend(records)
        page_num += 1
    
    save_records_to_txt(all_records, txt_file)
    print(f"Wyeksportowano posortowane dane do: {txt_file}")

def calculate_theoretical_values(N: int, b: int, n: int) -> Tuple[int, int]:
    """Oblicza teoretyczną liczbę faz i operacji dyskowych"""
    # Liczba początkowych serii
    initial_runs = math.ceil(N / (n * b))
    
    if initial_runs <= 1:
        # Faza 1: 2N/b
        phases = 0
        operations = 2 * N / b
    else:
        # Liczba faz scalania: log_{n}(r)
        phases = math.ceil(math.log(initial_runs, n))
        
        # Całkowity koszt: 2N/b (faza 1) + 2*phases*N/b (fazy scalania)
        operations = 2 * N / b * (1 + phases)
    
    return phases, int(operations)

def generate_large_test_file(filename: str, n_records: int, max_numbers: int = 15):
    """
    Generator dużego pliku testowego z losowymi rekordami
    Zapisuje do pliku tekstowego w formacie: liczba1,liczba2,liczba3,...
    """
    print(f"\nGenerowanie {n_records} losowych rekordów do pliku {filename}...")
    
    with open(filename, 'w') as f:
        for i in range(n_records):
            count = random.randint(1, max_numbers)
            numbers = set(random.randint(1, 100) for _ in range(count))
            f.write(','.join(map(str, numbers)) + '\n')
            
            if (i + 1) % 10000 == 0:
                print(f"  Wygenerowano {i + 1}/{n_records} rekordów...")
    
    print(f"Plik {filename} został utworzony z {n_records} rekordami")

def run_experiment():
    """Przeprowadza eksperyment"""
    print("\n" + "="*60)
    print("EKSPERYMENT: Analiza wydajności sortowania")
    print("="*60)
    
    # Parametry eksperymentu
    test_sizes = [100, 500, 1000, 5000, 10000, 20000, 50000, 100000]
    block_size = 10
    record_size = 128
    n_buffers_small = 5
    n_buffers_large = 50
    
    results = {
        'small': {'N': [], 'r': [], 'phases_practical': [], 'phases_theoretical': [], 
                  'ops_practical': [], 'ops_theoretical': []},
        'large': {'N': [], 'r': [], 'phases_practical': [], 'phases_theoretical': [], 
                  'ops_practical': [], 'ops_theoretical': []}
    }
    
    for buffer_config, n_buffers in [('small', n_buffers_small), ('large', n_buffers_large)]:
        print(f"\n{'='*60}")
        print(f"Testy dla n={n_buffers} buforów")
        print(f"{'='*60}")
        
        for N in test_sizes:
            print(f"\n--- Test dla N={N} rekordów ---")
            
            # Generuj dane
            records = generate_random_records(N)
            filename = f"test_data_{N}.dat"
            save_records_to_disk(records, filename, block_size, record_size)
            
            # Sortuj
            sorter = LargeBufferSort(filename, n_buffers, block_size, record_size)
            sorter.sort(show_phases=False)
            
            # Wartości praktyczne
            phases_practical = sorter.phase_count
            ops_practical = sorter.disk_sim.read_count + sorter.disk_sim.write_count
            
            # Wartości teoretyczne
            phases_theoretical, ops_theoretical = calculate_theoretical_values(N, block_size, n_buffers)
            
            # Liczba serii
            r = math.ceil(N / (n_buffers * block_size))
            
            # Zapisz wyniki
            results[buffer_config]['N'].append(N)
            results[buffer_config]['r'].append(r)
            results[buffer_config]['phases_practical'].append(phases_practical)
            results[buffer_config]['phases_theoretical'].append(phases_theoretical)
            results[buffer_config]['ops_practical'].append(ops_practical)
            results[buffer_config]['ops_theoretical'].append(ops_theoretical)
            
            print(f"Liczba serii (r): {r}")
            print(f"Fazy: praktyczne={phases_practical}, teoretyczne={phases_theoretical}")
            print(f"Operacje: praktyczne={ops_practical}, teoretyczne={ops_theoretical}")
            
            # Usuń plik testowy
            if os.path.exists(filename):
                os.remove(filename)
    
    # Wyświetl wyniki w konsoli
    print("\n" + "="*60)
    print("PODSUMOWANIE WYNIKÓW")
    print("="*60)
    
    for buffer_config in ['small', 'large']:
        n_buf = n_buffers_small if buffer_config == 'small' else n_buffers_large
        print(f"\n{'='*60}")
        print(f"Wyniki dla n={n_buf} buforów:")
        print(f"{'='*60}")
        print(f"{'N':<10} {'r':<8} {'Fazy (P)':<12} {'Fazy (T)':<12} {'Ops (P)':<12} {'Ops (T)':<12}")
        print("-" * 60)
        
        for i in range(len(results[buffer_config]['N'])):
            N = results[buffer_config]['N'][i]
            r = results[buffer_config]['r'][i]
            ph_p = results[buffer_config]['phases_practical'][i]
            ph_t = results[buffer_config]['phases_theoretical'][i]
            op_p = results[buffer_config]['ops_practical'][i]
            op_t = results[buffer_config]['ops_theoretical'][i]
            print(f"{N:<10} {r:<8} {ph_p:<12} {ph_t:<12} {op_p:<12} {op_t:<12}")
    
    # Zapisz wyniki do pliku TXT
    save_experiment_results_to_txt(results, n_buffers_small, n_buffers_large, block_size)
    
    # Wykresy
    plot_results(results, n_buffers_small, n_buffers_large)

def save_experiment_results_to_txt(results, n_small, n_large, block_size):
    """Zapisuje wyniki eksperymentu do pliku TXT w formacie tabeli"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"wyniki_eksperymentu_{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("WYNIKI EKSPERYMENTU: Sortowanie z użyciem wielkich buforów\n")
        f.write("="*80 + "\n")
        f.write(f"Data wykonania: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Blocking factor (b): {block_size}\n")
        f.write("="*80 + "\n\n")
        
        for buffer_config in ['small', 'large']:
            n_buf = n_small if buffer_config == 'small' else n_large
            
            f.write("="*80 + "\n")
            f.write(f"Porównanie danych teoretycznych i praktycznych dla n={n_buf} buforów\n")
            f.write("="*80 + "\n\n")
            
            # Tabela z wynikami
            f.write(f"{'N':<10} | {'r':<8} | {'fazy':<6} | {'operacje':<12} | "
                   f"{'oczekiwane fazy':<18} | {'oczekiwane operacje':<20}\n")
            f.write("-" * 80 + "\n")
            
            data = results[buffer_config]
            for i in range(len(data['N'])):
                N = data['N'][i]
                r = data['r'][i]
                ph_p = data['phases_practical'][i]
                op_p = data['ops_practical'][i]
                ph_t = data['phases_theoretical'][i]
                op_t = data['ops_theoretical'][i]
                
                f.write(f"{N:<10} | {r:<8} | {ph_p:<6} | {op_p:<12} | "
                       f"{ph_t:<18} | {op_t:<20}\n")
            
            f.write("\n")
    print(f"\nWyniki eksperymentu zapisano do: {filename}")

def plot_results(results, n_small, n_large):
    """Tworzy wykresy wyników"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Analiza wydajności sortowania z dużymi buforami', fontsize=16, fontweight='bold')
    
    configs = [('small', n_small), ('large', n_large)]
    
    for idx, (buffer_config, n_buffers) in enumerate(configs):
        data = results[buffer_config]
        
        # Wykres faz
        ax = axes[idx][0]
        ax.plot(data['N'], data['phases_practical'], 'o-', label='Praktyczne', 
                linewidth=2, markersize=8, color='#2E86AB')
        ax.plot(data['N'], data['phases_theoretical'], 's--', label='Teoretyczne', 
                linewidth=2, markersize=8, color='#A23B72')
        ax.set_xlabel('Liczba rekordów (N)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Liczba faz', fontsize=11, fontweight='bold')
        ax.set_title(f'Liczba faz scalania (n={n_buffers}, b=10)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        # Wykres operacji dyskowych
        ax = axes[idx][1]
        ax.plot(data['N'], data['ops_practical'], 'o-', label='Praktyczne', 
                linewidth=2, markersize=8, color='#2E86AB')
        ax.plot(data['N'], data['ops_theoretical'], 's--', label='Teoretyczne', 
                linewidth=2, markersize=8, color='#A23B72')
        ax.set_xlabel('Liczba rekordów (N)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Liczba operacji dyskowych', fontsize=11, fontweight='bold')
        ax.set_title(f'Operacje dyskowe (n={n_buffers}, b=10)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'wyniki_eksperymentu_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Wykresy zapisano do pliku: {filename}")
    plt.show()

def main():
    """Główna funkcja programu"""
    data_file = "data.dat"
    block_size = 4  # liczba rekordów na stronę (b)
    record_size = 128  # rozmiar rekordu w bajtach
    n_buffers = 3  # liczba buforów (n)
    
    while True:
        print("\n" + "="*60)
        print("SORTOWANIE Z UŻYCIEM WIELKICH BUFORÓW")
        print("="*60)
        print("1. Generuj losowe rekordy")
        print("2. Wprowadź rekordy z klawiatury")
        print("3. Wczytaj rekordy z pliku TXT")
        print("4. Wyświetl plik przed sortowaniem")
        print("5. Sortuj plik")
        print("6. Sortuj z wyświetlaniem faz")
        print("7. Wyświetl plik po sortowaniu")
        print("8. Eksportuj posortowane dane do TXT")
        print("9. Zmień parametry (b, n)")
        print("10. Przeprowadź eksperyment")
        print("11. Generator dużych plików testowych")
        print("0. Wyjście")
        print("="*60)
        
        choice = input("Wybór: ").strip()
        
        if choice == '1':
            n = int(input("Liczba rekordów do wygenerowania: "))
            records = generate_random_records(n)
            save_records_to_disk(records, data_file, block_size, record_size)
            
            # Automatycznie zapisz też do TXT
            txt_file = data_file.replace('.dat', '_input.txt')
            save_records_to_txt(records, txt_file)
            
            print(f"Wygenerowano i zapisano {n} rekordów")
            print(f"  - Plik binarny: {data_file}")
            print(f"  - Plik tekstowy: {txt_file}")
            
        elif choice == '2':
            records = read_records_from_keyboard()
            if records:
                save_records_to_disk(records, data_file, block_size, record_size)
                
                # Zapisz też do TXT
                txt_file = data_file.replace('.dat', '_input.txt')
                save_records_to_txt(records, txt_file)
                
                print(f"Zapisano {len(records)} rekordów")
                print(f"  - Plik binarny: {data_file}")
                print(f"  - Plik tekstowy: {txt_file}")
        elif choice == '3':
            filename = input("Nazwa pliku wejściowego TXT: ").strip()
            try:
                records = read_records_from_file(filename)
                save_records_to_disk(records, data_file, block_size, record_size)
                print(f"Wczytano i zapisano {len(records)} rekordów do {data_file}")
            except FileNotFoundError:
                print("Plik nie istnieje!")
            
        elif choice == '4':
            if os.path.exists(data_file):
                display_file_records(data_file, block_size, record_size)
            else:
                print("Plik nie istnieje! Najpierw wygeneruj dane.")
            
        elif choice == '5' or choice == '6':
            if os.path.exists(data_file):
                sorter = LargeBufferSort(data_file, n_buffers, block_size, record_size)
                sorter.sort(show_phases=(choice == '6'))
                
                print("\nSortowanie zakończone!")
                print("  Użyj opcji 7, aby wyświetlić posortowane dane")
                print("  Użyj opcji 8, aby wyeksportować do TXT")
            else:
                print("Plik nie istnieje! Najpierw wygeneruj dane.")
            
        elif choice == '7':
            if os.path.exists(data_file):
                display_file_records(data_file, block_size, record_size)
            else:
                print("Plik nie istnieje!")
            
        elif choice == '8':
            if os.path.exists(data_file):
                txt_file = data_file.replace('.dat', '_sorted.txt')
                export_sorted_to_txt(data_file, txt_file, block_size, record_size)
                print(f"Dane wyeksportowane do: {txt_file}")
            else:
                print("Plik nie istnieje!")
            
        elif choice == '9':
            try:
                block_size = int(input(f"Nowy blocking factor (obecnie b={block_size}): "))
                n_buffers = int(input(f"Nowa liczba buforów (obecnie n={n_buffers}): "))
                print(f"Parametry zmienione: b={block_size}, n={n_buffers}")
            except ValueError:
                print("Błędne wartości!")
            
        elif choice == '10':
            confirm = input("Eksperyment może potrwać kilka minut. Kontynuować? (t/n): ")
            if confirm.lower() == 't':
                run_experiment()
            
        elif choice == '11':
            print("\n" + "="*60)
            print("GENERATOR DUŻYCH PLIKÓW TESTOWYCH")
            print("="*60)
            try:
                n_records = int(input("Liczba rekordów do wygenerowania: "))
                filename = input("Nazwa pliku wyjściowego (np. test_10000.txt): ").strip()
                
                if not filename:
                    filename = f"test_{n_records}.txt"
                
                generate_large_test_file(filename, n_records)
                
                # Opcjonalnie konwertuj do .dat
                convert = input(f"\nKonwertować do formatu .dat? (t/n): ")
                if convert.lower() == 't':
                    print(f"Wczytuje {filename}...")
                    records = read_records_from_file(filename)
                    dat_filename = filename.replace('.txt', '.dat')
                    save_records_to_disk(records, dat_filename, block_size, record_size)
                    print(f"Plik binarny zapisany1 jako: {dat_filename}")
                    
            except ValueError:
                print("Błędna wartość!")
            
        elif choice == '0':
            print("\nDo widzenia!")
            break
        
        else:
            print("Nieprawidłowy wybór!")

if __name__ == "__main__":
    main()