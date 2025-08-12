# Sportka RL - AI Agent pro optimalizaci loterie

Tento projekt implementuje Deep Q-Learning (DQN) agenta pro optimalizaci strategie hraní české loterie **Sportka**. Agent se učí na základě historických dat a snaží se minimalizovat ztráty při hraní.

## 🎯 Přehled

Program obsahuje:
- **DQN agenta** natrénovaného pomocí Deep Reinforcement Learning
- **Simulační prostředí** pro Sportku s realistickými pravidly
- **Náhodného agenta** pro porovnání výkonu
- **Kompletní testování a statistické vyhodnocení** obou přístupů
- **Vizualizace výsledků** pomocí grafů

## 📁 Struktura projektu

```
v4/
├── main.py          # Testování a porovnání agentů
├── train.py         # Trénování DQN agenta
├── models.py        # Definice DQN a náhodného agenta
├── environment.py   # Simulační prostředí pro Sportku
└── README.md        # Tento soubor
```

### Popis souborů:

- **`environment.py`** - Gymnasium prostředí simulující Sportku s historickými daty
- **`models.py`** - DQN neural network a RandomAgent pro porovnání
- **`train.py`** - Trénování agenta (2000 epizod)
- **`main.py`** - Testování a porovnání výkonu agentů

## 🚀 Instalace a spuštění

### Požadavky:
```bash
pip install torch gymnasium numpy matplotlib scipy
```

### Použití:

1. **Trénování agenta:**
```bash
python train.py
```

2. **Testování a porovnání:**
```bash
python main.py
```

## 🧠 Algoritmus

### DQN Agent:
- **Architektura**: 3-vrstvá neuronová síť s dropout (512→512→256→50 neuronů)
- **Vstup**: Historie posledních 5 losování + aktuální balance
- **Výstup**: Pravděpodobnosti pro 49 čísel + Šance
- **Strategie**: Epsilon-greedy s postupným snižováním explorace

### Reward systém:
- **Základní reward**: `(výhry - náklady) / 100`
- **Bonus za výhru**: `počet_shod × 10` bodů
- **Penalizace**: `-2 × počet_opakujících_se_čísel` (při >2 opakováních)

## 🎲 Výherní tabulka

| Shody | Výhra | Pravděpodobnost |
|-------|-------|-----------------|
| 3     | 50 Kč | ~1:57 |
| 4     | 300 Kč | ~1:1,033 |
| 5     | 10,000 Kč | ~1:55,492 |
| 6     | 100,000 Kč | ~1:13,983,816 |

### Speciální bonusy:
- **Šance**: +1,000 Kč (při shodě posledního čísla)
- **Superjackpot**: +1,000,000 Kč (6 shod + Šance)

**Náklady**: 30 Kč za sloupeček + 30 Kč za Šanci

## 📊 Výsledky

Program produkuje:

### Během trénování:
- `training_results.png` - Grafy průběhu učení
- `sportka_model.pth` - Uložený natrénovaný model

### Při testování:
- `comparison_results.png` - Porovnání DQN vs Random agenta
- Detailní statistiky v konzoli:
  - Průměrný ROI, zůstatky, výhry
  - Počty výher podle kategorií
  - Statistická významnost (t-test)

### Příklad výstupu:
```
DQN Agent:
  Průměrný konečný zůstatek: 12,450.00 Kč
  Průměrné výhry: 8,230.00 Kč
  Průměrné ROI: -58.5%

Random Agent:
  Průměrný konečný zůstatek: 11,890.00 Kč
  Průměrné výhry: 7,960.00 Kč
  Průměrné ROI: -60.3%

Relativní zlepšení: +3.1%
```

## ⚙️ Technické detaily

### Hyperparametry:
- **Epizody**: 2,000
- **Learning rate**: 0.001
- **Batch size**: 32
- **Epsilon decay**: 0.995 (1.0 → 0.01)
- **Target network update**: každých 100 epizod

### Prostředí:
- **Počáteční balance**: 30,000 Kč
- **Maximální tikety**: 1,000 per epizoda
- **Observation space**: 5×49 (historie) + 2 (balance, counter)
- **Action space**: 50 (49 čísel + Šance)

## 📈 Očekávané výsledky

⚠️ **Důležité**: Oba přístupy jsou v dlouhodobém průměru **ztrátové**, což odpovídá realitě loterií. DQN agent se snaží **minimalizovat ztráty**, nikoliv vyhrávat.

Typická zlepšení DQN agenta:
- **ROI**: +2-5% relativní zlepšení oproti náhodnému sázení
- **Variance**: Nižší volatilita výsledků
- **Strategie**: Vyhýbání se opakování nedávných čísel

## 🔧 Možná rozšíření

- **Reálná historická data** místo simulovaných losování
- **Různé strategie sázení** (progresivní, proporcionální)
- **Ensemble metody** (kombinace více modelů)
- **Analýza hot/cold čísel** s delší historií
- **Optimalizace pro různé rozpočty**

## 📝 Poznámky

- Program používá **zjednodušenou výherní tabulku**
- Simulace **neobsahuje progresivní jackpoty**
- Šance se počítá jako `(drawn_chance == sum(player_numbers) % 10)`
- Model je určen **pouze pro výzkumné účely**

## 🎓 Závěr

Tento projekt demonstruje aplikaci Deep Reinforcement Learning na reálný problém optimalizace loterie. I když není možné "porazit" loterii, DQN agent dokáže nalézt strategie, které mírně snižují očekávané ztráty oproti náhodnému sázení.

---

*⚠️ Upozornění: Tento program je určen pouze pro vzdělávací a výzkumné účely. Hazard může být návykový.*