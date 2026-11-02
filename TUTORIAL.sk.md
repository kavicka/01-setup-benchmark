# Tutoriál · 01 · Setup a benchmark násobenia matíc

🇬🇧 [English version](TUTORIAL.md) · 📄 [README (výsledky)](README.sk.md)

Tento tutoriál postaví projekt od prázdneho priečinka, krok za krokom. Pri každom kroku hovorí, **čo** robíme, **prečo** to robíme práve takto a **ako** to funguje pod kapotou. README je správa (čísla, grafy); tento súbor je „ako som sa k tomu dostal“.

**Čo budeš mať na konci:** reprodukovateľné Python prostredie a skript, ktorý zmeria, ako rýchlo PyTorch násobí matice na CPU a na GPU Macu (MPS).

**Potrebuješ:** terminál, nainštalovaný [`uv`](https://docs.astral.sh/uv/), ~2 GB miesta na PyTorch.

---

## Krok 1 · Vytvor projekt cez `uv`

```bash
uv init setup-benchmark --python 3.12
cd setup-benchmark
uv add torch numpy matplotlib
```

**Čo sa stane:**
- `uv init` vytvorí `pyproject.toml` (názov projektu + zoznam závislostí) a `.python-version` (pripne Python 3.12).
- `uv add` stiahne balíčky do súkromného priečinka `.venv/` a zapíše **`uv.lock`**.

**Prečo `uv` a lockfile:** `pyproject.toml` hovorí „potrebujem torch ≥ 2.15“. To je *rozsah*. `uv.lock` zaznamená *presnú* verziu každého balíčka, vrátane ~30 balíčkov, ktoré si torch potiahne. Na inom počítači `uv sync` vytvorí identické prostredie. Bez locku môžu dva počítače nainštalovať rôzne verzie a dostať rôzne výsledky alebo chyby, a ty nevieš prečo.

**Prečo samostatné prostredie pre každý projekt:** každý krok v tomto repe má iné závislosti (neskoršie potrebujú scikit-learn, MLX, …). Samostatné `.venv/` znamenajú, že upgrade balíčka pre krok 12 nikdy nerozbije krok 01.

Over, že to funguje:
```bash
uv run python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
```
`uv run` spustí príkaz vnútri `.venv` bez toho, aby si čokoľvek aktivoval.

---

## Krok 2 · Zisti, aké zariadenia existujú

```python
devices = ["cpu"]
if torch.backends.mps.is_available():   # GPU na Apple Silicon
    devices.append("mps")
if torch.cuda.is_available():           # NVIDIA GPU
    devices.append("cuda")
```

**Ako PyTorch premýšľa o hardvéri:** každý tenzor žije na nejakom *zariadení* (device). `torch.randn(3, 3, device="mps")` vytvorí čísla priamo v pamäti GPU. Operácia beží na zariadení, kde sú jej vstupy. Ten istý kód teda beží na notebooku, Macu aj v Colabe: mení sa len reťazec `device=`.

- **MPS** (Metal Performance Shaders) je GPU backend od Apple.
- **CUDA** je NVIDIA. Na Macu nikdy nie je dostupná, ale vďaka kontrole skript funguje aj v Colabe/Kaggle.

**Prečo detekovať a nenapísať to natvrdo:** skript s `device="mps"` spadne na každom inom stroji než Mac. Detekcia s fallbackom na CPU je vzor, ktorý používa každý ďalší projekt.

---

## Krok 3 · Zmeraj jedno násobenie správne

Toto je jadro projektu. Naivná verzia je zlá:

```python
start = time.perf_counter()
a @ b
print(time.perf_counter() - start)   # ❌ na GPU zle
```

Skutočná funkcia v [`benchmark.py`](benchmark.py) rieši tri problémy:

```python
def bench(device, n, repeats=5):
    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)
    a @ b              # (1) zahriatie, nemeria sa
    sync(device)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        a @ b
        sync(device)   # (2) počkaj, kým GPU dokončí
        times.append(time.perf_counter() - start)
    return sorted(times)[len(times) // 2]   # (3) medián
```

**(1) Zahriatie (warm-up).** Prvé volanie je špeciálne: PyTorch alokuje pamäť a GPU driver pripraví (skompiluje/načíta) kernel. Tento jednorazový náklad nemá nič spoločné s rýchlosťou násobenia, preto ho spustíme raz a zahodíme.

**(2) Synchronizácia.** Na GPU `a @ b` pred návratom **nič nevypočíta**. Iba zaradí prácu do fronty a hneď sa vráti, aby Python mohol pokračovať, kým GPU paralelne počíta. Keď hneď zastavíš stopky, meriaš „ako dlho trvalo zaradiť prácu“, čo sú mikrosekundy bez ohľadu na veľkosť matice. `torch.mps.synchronize()` počká, kým sa fronta vyprázdni. Na CPU je práca synchrónna, takže `sync` nerobí nič:

```python
def sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
```

**(3) Medián z 5.** Jedno meranie môžu narušiť iné programy, systém alebo zmena frekvencie CPU. Medián ignoruje jeden-dva extrémy; priemer by nimi bol stiahnutý.

`time.perf_counter()` sa používa, lebo je to najpresnejšie hodiny v Pythone a nikdy neskáču (na rozdiel od `time.time()`, ktoré sleduje systémový čas).

---

## Krok 4 · Prepočítaj čas na GFLOPS

Samotný čas nevie porovnať maticu 256×256 s maticou 4000×4000. Preto spočítame prácu:

- Výsledok má n × n buniek.
- Každá bunka je skalárny súčin riadku a stĺpca: n násobení + n sčítaní = 2n operácií.
- Spolu: n² · 2n = **2n³ operácií s desatinnými číslami (FLOPs)**.

```python
gflops = 2 * n**3 / t / 1e9
```

FLOPs ÷ sekundy = FLOPs za sekundu; ÷ 10⁹ = **GFLOPS** (miliardy za sekundu). Meria *priepustnosť*: koľko užitočnej práce zariadenie stihne za sekundu, nezávisle od veľkosti matice.

**Prečo záleží na n³:** zdvojnásob stranu a práca narastie 2³ = 8×. Tento kubický rast je dôvod, prečo je veľkosť matíc (= šírka modelu) taká drahá, a neskôr prečo je drahá attention nad dlhým kontextom.

---

## Krok 5 · Prejdi veľkosti a ulož surové výsledky

```python
sizes = [256, 512, 1000, 2000, 4000]
for n in sizes:
    for d in devices:
        t = bench(d, n)
        results[d][n] = {"ms": t * 1000, "gflops": 2 * n**3 / t / 1e9}
json.dump({"torch": ..., "platform": ..., "mps": ..., "results": results}, f)
```

Veľkosti zámerne idú od malých po veľké, lebo správanie GPU sa s veľkosťou mení (Krok 7).

**Prečo zapisovať `results.json` a nie len vypisovať:** meranie a prezentácia sú dve rôzne úlohy. Benchmark je pomalý a závisí od stroja; kreslenie je rýchle a možno ho prerobíš desaťkrát. Uložené surové čísla (plus verzia torchu a platforma, aby si vedel, odkiaľ sú) ti dovolia prekresliť graf bez nového merania a README môže citovať presné čísla. Každý ďalší projekt používa vzor `skript → results.json → plot.py → assets/`.

---

## Krok 6 · Nakresli výsledky

[`plot.py`](plot.py) načíta `results.json` a nakreslí dva panely:

- **Vľavo čas vs n v log-log mierke,** plus prerušovaná referenčná čiara `n³` ukotvená pri najväčšej veľkosti. V log-log osiach sa mocninová závislosť `t ∝ n³` stane priamkou so sklonom 3, takže hneď vidíš, či meranie sedí s teóriou.
- **Vpravo GFLOPS vs n.** Ukazuje *efektivitu*: ako blízko sa zariadenie dostane k svojmu maximu.

```bash
uv run python benchmark.py   # meranie → results.json
uv run python plot.py        # kreslenie → assets/matmul_benchmark.png
```

---

## Krok 7 · Prečítaj výsledok (čo ťa naučí)

Z behu na Macu (`results.json`):

| n | CPU GFLOPS | MPS GFLOPS |
|---:|---:|---:|
| 256 | 1 286 | 78 |
| 4000 | 2 929 | 5 233 |

**Prečo GPU prehráva na malých maticiach:** každé volanie GPU má pevnú réžiu (poslanie príkazu, spustenie kernelu, čakanie na sync), tu zhruba 0,4 ms. Násobenie 256×256 je tak málo práce, že réžia prevládne. Pri 4000×4000 je práce 4000× viac, réžia sa stratí a tisíce paralelných jednotiek GPU vyhrajú.

**Prečo je CPU prekvapivo rýchle:** PyTorch volá vysoko optimalizovanú knižnicu lineárnej algebry (BLAS), ktorá využíva vektorové/maticové jednotky CPU na všetkých jadrách. Pri malých a stredných veľkostiach to porazí réžiu spúšťania na GPU.

**Ponaučenie pre ďalšie projekty:** GPU sa oplatí, keď mu dáš *veľa práce naraz*. Preto tréning používa mini-batche, preto Racing Lab počíta 200 mozgov áut v jednom matmule (kap. 01) a preto malé modely na MPS niekedy nezrýchlia vôbec (meranie KV cache v projekte 12).

---

## Vyskúšaj sám

1. Vymaž `sync(device)` z cyklu a spusti znova. MPS ukáže absurdne vysoké GFLOPS. To je presne chyba, ktorej sync bráni.
2. Pridaj `dtype=torch.float16` do `torch.randn` pre MPS. Polovičná presnosť presúva polovicu bajtov; stúpne priepustnosť?
3. Pridaj `n = 8000`. Udrží CPU tempo, alebo jeho GFLOPS klesnú (limity pamäte/cache)?

## Zhrnutie

| Krok | Myšlienka |
|---|---|
| 1 | `uv` + `uv.lock` = rovnaké prostredie všade |
| 2 | Tenzory žijú na zariadení; detekuj, nepíš natvrdo |
| 3 | Zahriatie, synchronizácia, medián: tri pravidlá merania GPU kódu |
| 4 | Matmul = 2n³ FLOPs; GFLOPS robí veľkosti porovnateľnými |
| 5–6 | Ulož surové čísla, kresli zvlášť |
| 7 | GPU potrebuje veľkú dávku práce, aby prekonalo svoju réžiu |
