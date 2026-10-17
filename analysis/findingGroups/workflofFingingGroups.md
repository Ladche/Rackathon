# Skutečně provedený pracovní postup

## 1. Pochopení zadání

Přečetla jsem aktuální požadavek, původní projektový prompt, popis dřívější
opravy a zdrojové zadání. Text prezentace a obou kopií datového rozhraní
byl lokálně vyčten z Office XML; nebyl poslán externí službě.

Původní úloha zahrnuje predikci. Pro tuto iteraci platí novější upřesnění:
**žádné ML, žádné učení modelu, žádné vynucení šesti skupin**.
Kandidátní skupiny z prezentace jsou pouze hypotézy. Cílem je porozumění
pozorovaným rozdílům a návrh měřitelné podpory.

Vytvořena jednotná struktura `analysis/findingGroups/code`,
`results` a `interpretation`. Původní uživatelský prompt nebyl změněn.
Hlavní modelovací aplikace ani její starší dokumentace nebyly přepisovány.

## 2. Dostupnost dat a výchozí kontrola prostředí

Na začátku CSV chybělo, proto jsem si vyžádala jeho umístění. Následně
byl uživatelem doplněn soubor `data/Rakhaton2026 (1).csv`, 182 763 436 bajtů.
Nebyl zaměněn za malý dřívější vývojový vzorek ani za syntetická data.

Pro závislosti bylo použito lokální prostředí `.venv` a existující
`requirements-dev.txt`. Před změnami byly spuštěny původní testy:
**144 prošlo za 27 sekund**. Existující samostatný linter či build této
analýzy v repozitáři není; nový nástroj nebyl přidáván.

## 3. Audit celého vstupu — 01_audit.py

- CSV se čte po 50 000 řádcích, ne celé jako tabulka textových objektů.
- Používají se stávající aliasy a validační kontrakty hlavní aplikace.
  Import sám nespouští trénování ani nečte data.
- Provádí se kontrola všech binárních příznaků, čísel, dat, věkových
  pásem, návaznosti screeningové historie, registrace a vnoření výsledků.
- Každá pozvánka je uložena do lokální SQLite databáze. Unikátní indexy
  následně kontrolují duplicity i přes hranice bloků.
- Spočteny chybějící hodnoty a kardinality všech dodaných polí, minima,
  maxima a průměry čísel, roční jmenovatele, délka pozorované historie
  a rozpory zdrojových počtů.
- Vstup má SHA-256
  `a75c5aa9f56d1d1526bf5ded005828c3c7318beee5ec89f477a7e24cda02033a`.
- Ověřen rozsah **1 344 556 pozvánek, 508 527 žen, 14. 1. 2020 až
  6. 12. 2024**. Úplný audit prvního běhu trval přibližně 30 sekund.

Nezavedla jsem předpoklad skutečného konce sledování. Výsledky mají
explicitní označení „zaznamenané příznaky, úplnost neověřena“.
Neúspěšný audit nevytvoří úspěšný manifest; další program částečnou
databázi bez takového manifestu odmítne.

## 4. Vytvoření proměnných — 02_features.py

Odvodila jsem:

- skutečně pozorovaný počet dřívějších pozvánek a rozdíl jejich dat,
- počty známých, pozitivních, rychlých a pozdních historických výsledků,
  zvlášť pro rezervy vykázání 0, 30 a 90 dní,
- pět stavů historie plus explicitní stav neznámého časování reakce,
- profil kontaktu a prevence včetně stavu neznámých údajů,
- screeningovou recency, registrační kombinace a zdrojový stav první
  pozvánky oddělený od první řádky v extraktu,
- počet zastoupených preventivních domén a příznaky registrace bez prevence,
- kontaktní pásma, roční a věkové kontexty.

Historické výsledky jsou k dispozici až po skončení celého 365denního
okna plus rezervy, striktně před aktuální pozvánkou. Rychlý pozitivní
výsledek nedostává nedoloženou předčasnou dostupnost. Aktuální výsledky
nejsou použity k zařazení aktuální pozvánky.

Rozdílná okna prevence se nesčítají do společného počtu výkonů.
Žádný údaj o digitální dostupnosti, cestovní vzdálenosti, motivaci nebo
diagnóze nebyl vytvořen z nedostatečných náhražek.

## 5. Volba správného jmenovatele

Nejprve jsem explorovala poslední pozvánku každé ženy napříč roky.
Výsledek ukázal velmi vysokou zaznamenanou účast vůči ročním tabulkám.
Přezkoumání výběru odhalilo zásadní problém: poslední pozvánka může být
poslední právě proto, že žena screening absolvovala.

Proto jsem tuto variantu **neponechala jako hlavní analýzu**. Hlavní
výstupy byly přepočteny na pevný rok 2024, bez podmiňování pozdějším
pozváním. V tomto roce je 256 466 žen / pozvánek. Původní varianta i první
pozorovaná pozvánka napříč roky zůstaly označenými kontrolami výběrového
zkreslení. Všechny zdrojové pozvánky zůstaly zahrnuty v auditu, historii
a ročních agregacích.

## 6. Hledání podložených provozních profilů — 03_groups.py

Nejprve byly spočteny úplné průniky historie × péče, recency × péče
a první pozvánka × historie, plus jednotlivé vztahy věku, registrace,
kontaktů a území. Bez určení „správného K“ se hledaly kombinace, které
vedou k odlišné smysluplné nabídce podpory.

Zjištěna důležitá skupina s velmi nedávným screeningem. Byla oddělena
bezpečnostní předností; není automaticky pokládána za rizikovou neúčast.
U historie bez uplynulého známého okna byly odděleny první a opakované
zdrojové pozvánky. Dřívější rychlá a pozdní účast mají společnou
návratovou nabídku, ale jejich příznaky zůstaly samostatně měřeny.

Provozní minimum je 500 žen. Pouze sousední malé podskupiny bez prevence
ve stejné historii se sdružily. Vzniklo **20 profilů**, nejmenší o 846 ženách.
Počet je výsledek těchto průhledných voleb a datové podpory, ne důkaz
dvaceti přirozených tříd.

Přiřazení všech 1 344 556 pozvánek všech 508 527 žen se ukládá jen lokálně;
hlavní roční kohorta má i samostatnou tabulku. Charakteristiky a míry vznikají až po
zařazení: velikost, podíl, účast po horizontech, Wilsonovy intervaly,
pozdní intervaly reakce, mediány kontaktů a prevence, registrace, věk,
zdrojová první pozvánka a kontext screeningové historie.

## 7. Robustnost a interpretace

Spočítala jsem:

- opakování stejné taxonomie po jednotlivých letech,
- devět kombinací kontaktní hranice 0/2/5 a historické rezervy 0/30/90,
- citlivost četností na minimum podpory 250/500/1000/2000,
- alternativní, výběrově zkreslené kohorty první/poslední pozvánky,
- věkovou standardizaci na společných podporovaných vrstvách včetně pokrytí,
- propojení jednotlivých skupin s dalšími registračními, věkovými a
  územními charakteristikami.

Stabilita není tvrzena, kde historie chybí. Není zaměněna asociace,
klinická potřeba, vysoká neúčast a ovlivnitelnost komunikací.

## 8. Intervence a jejich ověření — 04_interpret.py

Pro každý skutečně nalezený profil vznikla karta s popisným
názvem, přesným pravidlem, velikostí a charakteristikami, interpretací,
řešením, možnostmi kanálu, konkrétním kontrastem A/B a chybějícími údaji.
Na následné upřesnění požadavku byly krátké názvy rozšířeny tak, aby
vystihovaly oba rozměry profilu, a každá karta dostala úvodní slovní
charakteristiku. Pravidla, číselné výsledky a návrhy intervencí se nezměnily.

Navržené kanály zahrnují papírovou poštu, telefonní navigaci, ordinaci,
SMS, e-mail, zabezpečený portál, aplikaci a push. Jde o nabídky k ověření,
ne o údajně známé preference žen. Stejné právo na offline podporu zůstává
zachováno.

Pro 19 podpůrných profilů byl spočten nábor pro hypotetický absolutní
nárůst 2/5 procentních bodů, 80% sílu, běžné i konzervativně upravené alfa.
Nedávný screening má bezpečnější test podle skutečně doporučeného termínu,
pro který v datech není vhodný baseline. Žádný efekt nebyl vymyšlen.

Interpretace obsahuje pouze navazující karty skupin. Souhrnné informace,
společný protokol a limity jsou v `additionalInformation.md`.
Vznikly také dva grafy a reprodukční manifest.

## 9. Validace — 05_validate.py a pytest

Nové syntetické testy ověřují hraniční dny, chybění, priority profilů,
neznámé časování reakce, nepoužití aktuálního či budoucího výsledku,
invarianci pravidel vůči aktuálním labelům, výběr pevného roku,
duplicity mezi bloky, odmítnutí částečné databáze, agregované exporty,
intervaly a velikost experimentu.

Nezávislý validátor proti SQL poddotazům kontroluje počty a pozitivní
historické výsledky u všech 1 344 556 pozvánek pro tři rezervy, nikoli
jen malý vzorek. Dále ověřuje každý hlavní počet a míru, jednoznačnost
a persistenci přiřazení, úplnost segmentace všech zdrojových pozvánek,
podporu skupin, nepřítomnost identifikátorů v agregovaných CSV,
přesný počet interpretačních karet a hash kódu.

Opravená syntetická kontrola sloučení očekává jednu skutečně uplynulou
historii, nikoli nepozorovanou historii. Dřívější selhání tohoto očekávání
nebylo „vyřešeno“ změnou správné časové logiky.

Výsledný příkaz pro celou regresi:

```bash
.venv/bin/python -m pytest -q tests analysis/findingGroups/code/test_finding_groups.py
```

Po doplnění popisů celá regrese zahrnuje **165 testů**, z toho 144 původních
a 21 analytických; tři poslední kontrolují popisné názvy a charakteristiky.
Pandas na uměle vytvořených prázdných číselných
řetězcích hlásí FutureWarning z existujícího normalizátoru; nemění výsledek
a nesouvisí s trénováním. Nejsou přidány široké výjimky ani potlačování chyb.
Původní modelové testy používají syntetické fixture; analytické programy
na reálných datech nefitují žádný model.

## 10. Uchování výsledku

Kód, agregace, pravidla, grafy a interpretace zůstávají v této složce.
Individuální data jsou v lokální ignorované SQLite databázi, aby bylo možné
ověřit a znovu použít přiřazení bez zveřejnění identifikátorů.
Do git konfigurace je přidáno pouze ignorování této privátní podsložky.
Žádné oslovení žen, publikace dat, cloudová úloha ani commit neproběhly.

Poslední úspěšné technické ověření je strojově doloženo v
`results/validation.json` a skutečný vstup, verze a čas běhu v manifestech.
