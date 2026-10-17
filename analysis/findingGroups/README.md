# Hledání skupin žen pro vhodnější podporu screeningu

## Cíl a výsledek

Bez strojového učení, predikčního modelu a vynuceného počtu skupin popsat
pozorované vzorce, odvodit použitelné proměnné a navrhnout testovatelnou podporu.
Nejde o analýzu zmeškaných objednaných termínů, ale účasti po adresné pozvánce.

Byl zpracován **celý dodaný soubor: 1 344 556 pozvánek / 508 527 žen**.
Hlavní srovnání používá **256 466 žen pozvaných v roce 2024**, každou právě jednou.
Vzniklo **20 transparentních provozních profilů**, ne dvacet prokázaných
přirozených typů žen. Čtyři osy popisu jsou historie reakce, preventivní chování,
kontakty s péčí a nedávný screening; věk, registrace a lokalita jsou další vrstvy.

**Úplnost sledování výsledků není doložena.** Míry jsou podíly zaznamenaných
pozitivních příznaků, ne potvrzené odhady skutečné účasti či efektu komunikace.
Pojištění do konce roku 2025 není důkazem úplnosti výsledků.

## Co otevřít

- [Interpretace jednotlivých skupin](interpretation/groups.md): pouze skupiny
  pod sebou, jejich charakteristiky, řešení, kanály a vlastní návrhy A/B testů.
- [Další informace](additionalInformation.md): souhrn zjištění, limity, definice,
  bezpečnost, společný experimentální protokol a neověřené hypotézy.
- [Celý pracovní postup](workflofFingingGroups.md): co bylo skutečně provedeno,
  proč, s jakými kontrolami a jak výsledek reprodukovat.
- [Původní požadavek](prompt.md): zachován beze změny.

## Uspořádání

- [code/](code/): samostatně spustitelné analytické programy a syntetické testy.
- [results/](results/): skutečně spočítané agregace, definice, citlivost a grafy.
- [interpretation/](interpretation/): jediný výsledný interpretační soubor.

Používá se jednotná cesta `analysis/findingGroups`; překlep `findignGroups`
v požadavku nezakládá druhou složku. Interpretace je součást této analýzy,
nikoli nesouvisející složka v kořeni repozitáře.

## Spuštění z kořene repozitáře

Používají se existující závislosti a pytest, žádný nový analytický framework.
Ověřeno lokálně na Pythonu 3.13.3. Data neopouštějí tento počítač.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt

.venv/bin/python analysis/findingGroups/code/01_audit.py \
  --data 'data/Rakhaton2026 (1).csv'
.venv/bin/python analysis/findingGroups/code/02_features.py
.venv/bin/python analysis/findingGroups/code/03_groups.py
.venv/bin/python analysis/findingGroups/code/04_interpret.py
.venv/bin/python analysis/findingGroups/code/05_validate.py

.venv/bin/python -m pytest -q tests analysis/findingGroups/code/test_finding_groups.py
```

Audit úmyslně odmítá přepsat existující databázi. Pro nový nezávislý běh přidejte
**ke všem pěti programům stejný nový `--results /absolutni/cesta`**.
U posledních dvou lze pomocí `--documentation /absolutni/cesta` změnit umístění
interpretačního výstupu i jeho kontroly. Pro samotné přepočítání odvozených
proměnných nad již ověřeným vstupem existuje `02_features.py --rebuild`.
Následně je nutné znovu spustit kroky 03–05.

Hranice nízkého kontaktu má základní hodnotu 2; program automaticky porovnává
0/2/5 kontaktů a zpoždění historického výsledku 0/30/90 dní.
`03_groups.py --minimum-support 500` určuje minimální provozní četnost,
nikoli počet skupin. Výsledky pro 250/500/1000/2000 jsou rovněž spočítané.
Hlavní rok je poslední kalendářní rok pozvánek, **nikoli zaručeně plně
dosledovaný rok**.

## Hlavní výstupy

| Soubor v results/ | Obsah |
|---|---|
| [audit.json](results/audit.json), [schema_audit.csv](results/schema_audit.csv) | Rozsah, hash vstupu, kvalita, chybění a role sloupců |
| [feature_registry.csv](results/feature_registry.csv) | Definice a časová dostupnost vytvořených proměnných |
| [feature_associations.csv](results/feature_associations.csv) | Jednorozměrné vztahy v hlavní kohortě |
| [candidate_profiles.csv](results/candidate_profiles.csv) | Výchozí průniky před bezpečnostním filtrem a zpřesněním |
| [groups.csv](results/groups.csv), [groups.json](results/groups.json) | Velikost, účast, nejistota a přesné mapování výsledných skupin |
| [all_invitation_groups.csv](results/all_invitation_groups.csv) | Zařazení všech 1 344 556 pozvánek; opakované ženy, bez naivních intervalů nezávislosti |
| [group_characteristics.csv](results/group_characteristics.csv), [group_context.csv](results/group_context.csv) | Mediány, vlastnosti, věkové, registrační a územní řezy |
| [annual_groups.csv](results/annual_groups.csv), [age_standardization.csv](results/age_standardization.csv) | Roční kontext a popisná věková standardizace |
| [threshold_sensitivity.csv](results/threshold_sensitivity.csv), [cohort_sensitivity.csv](results/cohort_sensitivity.csv) | Citlivost hranic, historie a výběru kohorty |
| [trial_power_scenarios.csv](results/trial_power_scenarios.csv) | Orientační nábor pro hypotetické zvýšení o 2/5 procentních bodů |
| [export_suppression.json](results/export_suppression.json) | Počty nezveřejněných malých řádků |
| [run_manifest.json](results/run_manifest.json), [validation.json](results/validation.json) | Reprodukční metadata a nezávislé ověření výstupů |
| [groups_180d.png](results/groups_180d.png), [group_sizes.png](results/group_sizes.png) | Účast s intervaly a velikosti skupin |

Identifikátory a přiřazení žen zůstávají pouze v lokální ignorované databázi
`results/private/analysis.sqlite`. Tabulka `features` obsahuje všechny
pozvánky s časově odvozenou historií, `invitation_assignments` přiřazení
všech pozvánek všech 508 527 žen a `main_assignments` hlavní roční kohortu.
Veřejné CSV v této složce jsou pouze agregované; ani agregace
nejsou automaticky schválené k veřejnému publikování. Nic nebylo odesláno.
