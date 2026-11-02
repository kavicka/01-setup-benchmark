# 01 · Setup a benchmark násobenia matíc

🇬🇧 [English version](README.md)

## TL;DR
Reprodukovateľné Python ML prostredie spravované cez `uv` a benchmark, ktorý zistí dostupné zariadenia (CPU, Apple MPS, NVIDIA CUDA) a zmeria ich rýchlosť pri násobení matíc. Uložené výsledky sú z Macu s Apple Silicon a obsahujú CPU aj MPS. Väčšie matice vyžadujú približne **O(n³)** práce.

## Najprv jednoducho
Matica je tabuľka čísel. Neurónové siete takéto tabuľky neustále násobia: napríklad keď spracúvajú vstupy alebo porovnávajú slová vo vete. Táto úloha si preto vyberie jeden základný výpočet a položí dve otázky: **viem projekt nainštalovať na inom počítači a ako rýchlo tento výpočet vykoná môj procesor alebo grafická karta?**

Skript vytvorí dvojice štvorcových matíc s rôznym počtom riadkov, každú dvojicu vynásobí na dostupných zariadeniach a zmeria čas. Výsledkom je tabuľka, súbor `results.json` a graf. **Nižší čas znamená rýchlejší výpočet; vyššie GFLOPS znamená viac vykonanej práce za sekundu.** Meranie opakuje, prvé spustenie nezapočíta a pri GPU počká na dokončenie výpočtu.

Je to príprava na ďalšie ML projekty a ukážka správneho merania. Nejde o tréning modelu: rýchlosť jedného násobenia matíc sama osebe neurčuje rýchlosť celej neurónovej siete.

## Problém
Každý ďalší projekt v tomto repe (autograd, GPT od nuly, LoRA fine-tuning, RAG) stojí na dvoch veciach:
1. prostredie, ktoré sa na každom stroji nainštaluje rovnako, a
2. vedieť, či PyTorch dokáže počítať na dostupnom GPU a ako sa jeho čas líši od CPU.

Násobenie matíc je správna vec na meranie: lineárna vrstva, attention (`Q @ Kᵀ`, `scores @ V`) aj MLP blok sú násobenia matíc a tvoria väčšinu výpočtu v neurónovej sieti.

## Setup
![Setup prostredia](assets/01_setup.png)

```bash
uv init ml-sandbox --python 3.12
cd ml-sandbox
uv add torch numpy matplotlib
uv run python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
```

Čo robí ktorá časť:

| Nástroj / súbor | Úloha |
|---|---|
| `uv` | Nainštaluje samotný Python aj balíky a vytvorí `.venv/` len pre tento projekt. Oveľa rýchlejší ako pip. |
| `.python-version` | Zamkne Python 3.12 pre tento projekt. |
| `pyproject.toml` | Zoznam priamych závislostí (`torch`, `numpy`, `matplotlib`). |
| `uv.lock` | Zamkne presnú verziu **každého** balíka vrátane tranzitívnych. `uv sync` na inom stroji vytvorí identické prostredie. |
| `uv run …` | Spustí príkaz v `.venv` projektu bez ručnej aktivácie. |

## Ako to funguje

### 1. Zistenie zariadení
```python
devices = ["cpu"]
if torch.backends.mps.is_available():   # GPU na Apple Silicon (Metal Performance Shaders)
    devices.append("mps")
if torch.cuda.is_available():           # NVIDIA GPU
    devices.append("cuda")
```
PyTorch dáva každý tenzor na nejaké *zariadenie* (device). Ten istý kód beží všade, meníš len `device=`. Na Macu je GPU backend MPS, CUDA je NVIDIA backend, ktorý používajú Colab a Kaggle.

### 2. Meranie jedného násobenia
```python
a = torch.randn(n, n, device=device)
b = torch.randn(n, n, device=device)
a @ b                 # zahriatie (warm-up)
sync(device)
for _ in range(repeats):
    start = time.perf_counter()
    a @ b
    sync(device)
    times.append(time.perf_counter() - start)
return median(times)
```
Tri pasce pri meraní, ktorým sa kód vyhýba:

- **Warm-up.** Prvé volanie alokuje pamäť a pripravuje GPU kernely. Je vždy pomalšie a skreslilo by výsledok, preto sa spustí raz a nemeria sa.
- **Synchronizácia.** Na GPU sa `a @ b` vráti okamžite a výpočet beží asynchrónne na pozadí. Bez `torch.mps.synchronize()` (alebo `torch.cuda.synchronize()`) by časovač zmeral len to, ako dlho trvalo prácu *zaradiť do fronty*, a GPU by vyzeralo nereálne rýchlo. Na CPU `sync` nerobí nič, lebo CPU počíta synchrónne.
- **Medián z 5 behov.** Jednotlivé merania sú zašumené (iné procesy, zmena frekvencie CPU). Medián ignoruje extrémy lepšie ako priemer.

### 3. Z času na GFLOPS
```python
gflops = 2 * n**3 / t / 1e9
```
Každá z n² výstupných buniek je skalárny súčin dĺžky n: n násobení a n sčítaní. Spolu n² · 2n = **2n³** operácií s pohyblivou desatinnou čiarkou (FLOPs). Delením časom dostaneš FLOPs za sekundu a `1e9` to prevedie na GFLOPS. Vďaka tomu sa dajú priamo porovnať rôzne veľkosti matíc aj rôzne zariadenia.

### 4. Výstupy
- `benchmark.py` vypíše tabuľku a uloží surové čísla do `results.json`.
- `plot.py` načíta `results.json` a nakreslí `assets/matmul_benchmark.png`. Oddelenie merania a kreslenia znamená, že graf môžeš upraviť bez opakovania benchmarku.

```
benchmark.py ──► results.json ──► plot.py ──► assets/matmul_benchmark.png
      └──► tabuľka v termináli
```

## Výsledky
![Výstup benchmarku](assets/02_benchmark.png)

*Historický výstup z linuxového CPU. Aktuálna tabuľka a graf nižšie vychádzajú z `results.json` nameraného na Macu s CPU a MPS.*

| n | CPU čas [ms] | CPU GFLOPS | MPS čas [ms] | MPS GFLOPS |
|---:|---:|---:|---:|---:|
| 256 | 0,03 | 1 286 | 0,43 | 78 |
| 512 | 0,12 | 2 293 | 0,48 | 561 |
| 1000 | 0,66 | 3 044 | 1,23 | 1 620 |
| 2000 | 4,56 | 3 512 | 3,83 | 4 178 |
| 4000 | 43,70 | 2 929 | 24,46 | 5 233 |

![Graf benchmarku](assets/matmul_benchmark.png)

**Ako čítať grafy**
- **Vľavo:** čas rastie s veľkosťou matice; prerušovaná čiara ukazuje teoretickú závislosť `n³`. Zdvojnásobenie strany znamená približne osemnásobok základných operácií, ale skutočný čas ovplyvňuje aj réžia a pamäť.
- **Vpravo:** GFLOPS je počet miliárd operácií za sekundu. Pri malej matici je MPS pomalší než CPU, lebo príprava práce na GPU tvorí veľkú časť času; pri veľkej matici sa paralelný výpočet vyplatí. Toto poradie platí pre nameraný Mac, nie pre každý počítač.

**Prečo na tom neskôr záleží:** attention porovnáva každý token s každým, takže jej cena rastie so štvorcom dĺžky kontextu. Veľkosť modelu a dĺžka kontextu sa násobia, a preto existujú KV cache, kvantizácia a batching (projekty P4 a P10).

## 🏎️ Na autíčkach

200 áut, každé s vlastným malým neurónovým mozgom. Ak sa všetky mozgy spočítajú **jedným dávkovým násobením matíc** namiesto Python cyklu cez autá, výsledok je rovnaký a **21× rýchlejší** (0,73 ms → 0,034 ms pre 200 áut). Rovnaké ponaučenie ako v tomto kroku: daj prácu do dávky a potom ju zmeraj.

![ch01_batch](https://github.com/kavicka/05-racing-lab/raw/main/assets/ch01_batch.gif)

Celá kapitola: [Racing Lab · kap. 01](https://github.com/kavicka/05-racing-lab/blob/main/README.sk.md#kap-01--batching-200-mozgov-v-jednom-násobení-matíc) · kód: [`chapters/ch01_batch.py`](https://github.com/kavicka/05-racing-lab/blob/main/chapters/ch01_batch.py)

## Čo som sa naučil
- `uv` + lockfile robia každý projekt reprodukovateľným jedným príkazom.
- Ako v PyTorch zistiť akcelerátory a presúvať tenzory medzi zariadeniami.
- Ako správne merať GPU kód: warm-up, synchronizácia, medián.
- Násobenie matíc je O(n³) a GFLOPS normalizuje rýchlosť, aby sa dali zariadenia porovnať.

## Spustenie
```bash
uv sync                      # vytvorí .venv z uv.lock
uv run python benchmark.py   # vypíše tabuľku, zapíše results.json
uv run python plot.py        # zapíše assets/matmul_benchmark.png
```

## Obmedzenia
- Zverejnené čísla sú z jedného Macu s Apple Silicon; iný hardvér môže dať iné výsledky.
- Len float32; float16/bfloat16 by na GPU boli rýchlejšie.
- Jedna operácia (štvorcové násobenie matíc) nie je plnohodnotný benchmark tréningu.
