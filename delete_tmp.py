import os
import glob
import argparse

def find_data_files(path: str):
    pattern = os.path.join(path, "data.*")
    return [f for f in glob.glob(pattern) if os.path.isfile(f)]

def main():
    parser = argparse.ArgumentParser(description="Usuń pliki 'data.*' z bieżącego katalogu")
    parser.add_argument("-y", "--yes", action="store_true", help="Potwierdź usunięcie bez pytania")
    args = parser.parse_args()

    cwd = os.getcwd()
    files = find_data_files(cwd)

    if not files:
        print("Brak plików pasujących do 'data.*' w bieżącym katalogu.")
        return

    print("Znalezione pliki:")
    for f in files:
        print("  ", os.path.basename(f))

    if not args.yes:
        ans = input("Usunąć powyższe pliki? (t/n): ").strip().lower()
        if ans != 't':
            print("Anulowano.")
            return

    removed = 0
    for f in files:
        try:
            os.remove(f)
            removed += 1
        except Exception as e:
            print(f"Nie udało się usunąć {f}: {e}")

    print(f"Usunięto {removed} plik(ów).")

if __name__ == "__main__":
    main()
