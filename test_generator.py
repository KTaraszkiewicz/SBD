#!/usr/bin/env python3
"""
Generator dużych plików testowych do sortowania
Użycie: python test_generator.py
"""

import random
import argparse

def generate_test_file(filename, n_records, min_numbers=1, max_numbers=15, 
                       min_value=1, max_value=1000):
    """
    Generuje plik testowy z losowymi zbiorami liczb
    
    Args:
        filename: nazwa pliku wyjściowego
        n_records: liczba rekordów do wygenerowania
        min_numbers: minimalna liczba elementów w zbiorze
        max_numbers: maksymalna liczba elementów w zbiorze
        min_value: minimalna wartość liczby w zbiorze
        max_value: maksymalna wartość liczby w zbiorze
    """
    print(f"Generowanie {n_records} rekordów...")
    print(f"Parametry:")
    print(f"  - Liczba elementów w zbiorze: {min_numbers}-{max_numbers}")
    print(f"  - Zakres wartości: {min_value}-{max_value}")
    print(f"  - Plik wyjściowy: {filename}")
    print()
    
    with open(filename, 'w') as f:
        for i in range(n_records):
            # Losuj liczbę elementów w zbiorze
            count = random.randint(min_numbers, max_numbers)
            
            # Generuj unikalny zbiór liczb
            numbers = set()
            while len(numbers) < count:
                numbers.add(random.randint(min_value, max_value))
            
            # Zapisz do pliku (format: liczba1,liczba2,liczba3,...)
            f.write(','.join(map(str, sorted(numbers))) + '\n')
            
            # Wyświetl postęp co 10%
            if (i + 1) % max(1, n_records // 10) == 0:
                progress = (i + 1) / n_records * 100
                print(f"  Postęp: {progress:.0f}% ({i + 1}/{n_records})")
    
    print(f"\n✓ Plik {filename} został utworzony!")
    print(f"  Rozmiar: {n_records} rekordów")

def generate_test_suite():
    """Generuje zestaw plików testowych o różnych rozmiarach"""
    test_cases = [
        ("test_small.txt", 100),
        ("test_medium.txt", 1000),
        ("test_large.txt", 10000),
        ("test_very_large.txt", 50000),
        ("test_huge.txt", 100000)
    ]
    
    print("="*60)
    print("GENERATOR ZESTAWU PLIKÓW TESTOWYCH")
    print("="*60)
    print()
    
    for filename, n_records in test_cases:
        print(f"\n{'='*60}")
        generate_test_file(filename, n_records)
    
    print("\n" + "="*60)
    print("✓ Wszystkie pliki testowe zostały wygenerowane!")
    print("="*60)

def generate_special_cases():
    """Generuje pliki z przypadkami specjalnymi"""
    print("="*60)
    print("GENERATOR PRZYPADKÓW SPECJALNYCH")
    print("="*60)
    print()
    
    # Przypadek 1: Już posortowane
    print("\n1. Generowanie już posortowanych danych...")
    with open("test_sorted.txt", 'w') as f:
        for i in range(1, 101):
            f.write(f"{i}\n")
    print("   ✓ test_sorted.txt (100 rekordów, już posortowane)")
    
    # Przypadek 2: Odwrotnie posortowane
    print("\n2. Generowanie odwrotnie posortowanych danych...")
    with open("test_reverse.txt", 'w') as f:
        for i in range(100, 0, -1):
            f.write(f"{i}\n")
    print("   ✓ test_reverse.txt (100 rekordów, odwrotnie posortowane)")
    
    # Przypadek 3: Wszystkie takie same
    print("\n3. Generowanie identycznych rekordów...")
    with open("test_identical.txt", 'w') as f:
        for i in range(100):
            f.write("5,10,15\n")
    print("   ✓ test_identical.txt (100 identycznych rekordów)")
    
    # Przypadek 4: Duże zbiory
    print("\n4. Generowanie dużych zbiorów (max 15 elementów)...")
    with open("test_large_sets.txt", 'w') as f:
        for i in range(100):
            numbers = set(range(i, i + 15))
            f.write(','.join(map(str, numbers)) + '\n')
    print("   ✓ test_large_sets.txt (100 rekordów, każdy po 15 elementów)")
    
    # Przypadek 5: Małe zbiory
    print("\n5. Generowanie małych zbiorów (1 element)...")
    with open("test_small_sets.txt", 'w') as f:
        for i in range(100):
            f.write(f"{random.randint(1, 100)}\n")
    print("   ✓ test_small_sets.txt (100 rekordów, po 1 elemencie)")
    
    print("\n" + "="*60)
    print("✓ Wszystkie przypadki specjalne zostały wygenerowane!")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(
        description='Generator plików testowych do sortowania z dużymi buforami'
    )
    
    parser.add_argument(
        '-n', '--records',
        type=int,
        help='Liczba rekordów do wygenerowania'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Nazwa pliku wyjściowego'
    )
    
    parser.add_argument(
        '--suite',
        action='store_true',
        help='Generuj pełny zestaw plików testowych'
    )
    
    parser.add_argument(
        '--special',
        action='store_true',
        help='Generuj przypadki specjalne'
    )
    
    parser.add_argument(
        '--min-numbers',
        type=int,
        default=1,
        help='Minimalna liczba elementów w zbiorze (domyślnie: 1)'
    )
    
    parser.add_argument(
        '--max-numbers',
        type=int,
        default=15,
        help='Maksymalna liczba elementów w zbiorze (domyślnie: 15)'
    )
    
    parser.add_argument(
        '--min-value',
        type=int,
        default=1,
        help='Minimalna wartość liczby (domyślnie: 1)'
    )
    
    parser.add_argument(
        '--max-value',
        type=int,
        default=100,
        help='Maksymalna wartość liczby (domyślnie: 100)'
    )
    
    args = parser.parse_args()
    
    # Jeśli brak argumentów, tryb interaktywny
    if not any([args.records, args.suite, args.special]):
        print("="*60)
        print("GENERATOR PLIKÓW TESTOWYCH")
        print("="*60)
        print("\n1. Generuj pojedynczy plik")
        print("2. Generuj zestaw plików testowych")
        print("3. Generuj przypadki specjalne")
        print("4. Wszystko powyższe")
        print()
        
        choice = input("Wybór (1-4): ").strip()
        
        if choice == '1':
            n_records = int(input("Liczba rekordów: "))
            filename = input("Nazwa pliku (np. test_data.txt): ").strip()
            generate_test_file(filename, n_records)
        
        elif choice == '2':
            generate_test_suite()
        
        elif choice == '3':
            generate_special_cases()
        
        elif choice == '4':
            generate_test_suite()
            print()
            generate_special_cases()
        
        else:
            print("Nieprawidłowy wybór!")
    
    else:
        # Tryb z argumentami
        if args.suite:
            generate_test_suite()
        
        if args.special:
            generate_special_cases()
        
        if args.records and args.output:
            generate_test_file(
                args.output,
                args.records,
                args.min_numbers,
                args.max_numbers,
                args.min_value,
                args.max_value
            )

if __name__ == "__main__":
    main()