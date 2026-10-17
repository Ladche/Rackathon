# Analýza účasti na mamografickém screeningu a návrh segmentace

## 1. Co nám data říkají v kostce

Dataset obsahuje **1 344 556 pozvánek** a **508 527 unikátních klientek**. Pozvánky pokrývají období od **14. 1. 2020 do 6. 12. 2024**.

| Ukazatel | Hodnota |
|---|---:|
| Pozvánky celkem | 1 344 556 |
| Unikátní klientky | 508 527 |
| Zaznamenaný screening do 3 měsíců | 7,12 % |
| Zaznamenaný screening do 6 měsíců | 12,54 % |
| Zaznamenaný screening do 12 měsíců | 19,30 % |
| Řádky náležející klientkám s více než jednou pozvánkou | 85,8 % |

Základní interpretace: přibližně **jedna z pěti pozvánek** je spojena se zaznamenaným screeningovým výkonem do 12 měsíců. Nejde ale o podíl klientek, které se někdy účastní screeningu; jde o výsledek navázaný na konkrétní pozvánku a stanovené časové okno.

## 2. Zajímavá zjištění

### A. Preventivní aktivita souvisí s účastí na mamografii

Nejsilnější kladnou číselnou asociaci s účastí do 12 měsíců měla proměnná `GYN_PREVENTION_24M_CNT`, tedy počet preventivních prohlídek u gynekologa za předchozích 24 měsíců.

| Proměnná | Spearmanovo rho vůči `SCREENING_12M_FLAG` |
|---|---:|
| `GYN_PREVENTION_24M_CNT` | 0,3308 |
| `PREVENTION_TOTAL_CNT` | 0,2999 |
| `OTHER_SCREENING_36M_CNT` | 0,2552 |
| `DENTAL_PREVENTION_24M_CNT` | 0,1834 |
| `HEALTHCARE_CONTACT_12M_CNT` | 0,1519 |
| Historický počet pozvánek | -0,2537 |

Průměrný počet preventivních prohlídek u gynekologa byl **1,08** u řádků se zaznamenaným screeningem do 12 měsíců a **0,42** u řádků bez zaznamenaného screeningu. To naznačuje, že kontakt s gynekologickou prevencí může být vhodným kanálem pro podporu účasti.

Jde o asociace, nikoli důkaz příčiny. Samotná korelace neříká, že zvýšení počtu preventivních prohlídek automaticky způsobí vyšší účast na mamografii.

### B. Opakované pozvánky souvisejí s nižší následnou účastí

Průměrný počet předchozích pozvánek byl **0,66** u řádků se zaznamenaným screeningem do 12 měsíců a **1,48** u řádků bez zaznamenaného screeningu.

Historický počet pozvánek měl se screeningem do 12 měsíců zápornou asociaci (Spearmanovo rho **-0,2537**). Opakovaně zvané klientky bez zaznamenané účasti proto mohou být důležitou cílovou skupinou pro další intervenci. Data však sama nevysvětlují, proč se účast nedostavila — například zda hrála roli dostupnost termínu, obavy, jiné priority nebo chybějící záznam.

### C. První pozvánka je významná rozlišovací proměnná

`FIRST_INVITATION` měla ze sledovaných kategoriálních proměnných nejsilnější asociaci s účastí napříč časovými horizonty. Cramérovo V rostlo z **0,248** pro horizont 3 měsíců na **0,360** pro horizont 12 měsíců.

Bez podrobné tabulky četností a měr účasti podle kategorií ale nelze z těchto hodnot určit konkrétní směr ani říct, která kategorie má nejvyšší účast.

### D. Věkové pásmo souvisí s rozdíly v účasti

U `AGE_BAND` byl rozdíl mezi nejvyšší a nejnižší mírou účasti do 12 měsíců **46,81 procentního bodu**; Cramérovo V dosahovalo **0,287**.

To ukazuje, že věkové pásmo může být pro segmentaci užitečné. Z dostupného souhrnu však nevyplývá, která konkrétní věková kategorie měla nejvyšší nebo nejnižší účast. Před návrhem cílené komunikace je potřeba ověřit detailní míry podle jednotlivých pásem.

### E. Region je v modelu důležitý, ale je třeba jej interpretovat opatrně

Random Forest přiřadil proměnné `VZP_TERRITORIAL_AREA` pro horizont 12 měsíců důležitost **0,131**. Proměnná má ale **75 kategorií** a důležitost založená na nečistotě stromů může zvýhodňovat proměnné s mnoha kategoriemi.

Regionální rozdíly mohou být užitečným podnětem k dalšímu zkoumání, ale region VZP není přímým měřením dostupnosti pracovišť, čekacích dob ani dopravní dostupnosti.

## 3. Co ukázala segmentace do pěti clusterů

| Cluster | Podíl pozvánek | Účast do 12 měsíců | Nejčastější pásmo prevence |
|---|---:|---:|---|
| 0 | 18,0 % | 6,09 % | 0 |
| 1 | 20,0 % | 47,963 % | 4–6 |
| 2 | 20,3 % | 20,926 % | 4–6 |
| 3 | 17,8 % | 13,45 % | 2–3 |
| 4 | 23,8 % | 8,195 % | 0 |

Ve všech clusterech byla nejčastější kategorií historie screeningu **„0 in last 5y“** a nejčastějším profilem registrace **„GP+GYN registered“**. To neznamená, že tyto charakteristiky má každá klientka v daném clusteru; jde pouze o nejčastější kategorii.

Zvlášť zajímavé je srovnání clusterů 1 a 2: mají stejné nejčastější pásmo prevence, ale účast do 12 měsíců se liší přibližně o **27 procentních bodů**. Samotný počet preventivních prohlídek tedy zřejmě nevysvětluje všechny rozdíly v chování. Pro praktické pojmenování clusterů je vhodné projít jejich úplné profily a zastoupení jednotlivých kategorií.

## 4. Návrh akčně orientovaných segmentů

Níže jsou pracovní segmenty pro návrh komunikace a pilotních intervencí. Nejde o přesně vymezené skupiny, které lze bez dalších pravidel přímo vyexportovat z dat; před nasazením je potřeba definovat jednoznačná pravidla a ověřit velikost každého segmentu.

### 1. Preventivně aktivní, ale bez zaznamenaného mamografického screeningu

**Možná charakteristika:** klientky s alespoň nějakou gynekologickou preventivní aktivitou, u kterých není v daném okně zaznamenán mamografický screening.

**Možná intervence:**
- využít gynekologickou prevenci jako komunikační kanál;
- připojit stručné doporučení ke screeningu;
- nabídnout jednoduchý postup objednání a jasné informace o tom, kde hledat termín.

**Proč:** gynekologická preventivní aktivita vykazuje nejsilnější kladnou číselnou asociaci s účastí z uvedených počtových proměnných. Pilot by ověřil, zda lze tento kontakt účinně využít i u klientek bez zaznamenané účasti.

### 2. Opakovaně zvané klientky bez zaznamenané účasti

**Možná charakteristika:** více předchozích pozvánek a žádný zaznamenaný screening v relevantním časovém okně.

**Možná intervence:**
- osobnější kontakt místo pouhého opakování stejné pozvánky;
- nabídnout pomoc s nalezením termínu;
- pokud je to vhodné, umožnit klientce sdělit, co jí v účasti brání.

**Proč:** vyšší počet předchozích pozvánek byl spojen s nižší následnou účastí. Další identická připomínka nemusí být pro tuto skupinu nejúčinnější.

### 3. Klientky s nízkou preventivní aktivitou napříč oblastmi

**Možná charakteristika:** nízké počty preventivních prohlídek a dalších preventivních výkonů.

**Možná intervence:**
- jednoduché a srozumitelné vysvětlení účelu screeningu;
- přehledné instrukce, jak postupovat;
- podle možností asistence s objednáním.

**Poznámka:** nízká zaznamenaná aktivita může mít různé příčiny a může částečně odrážet i neúplnost dostupných záznamů. Není vhodné ji automaticky interpretovat jako nezájem.

### 4. Klientky, kterým může pomoci snazší cesta k termínu

**Možná charakteristika:** skupina vybraná podle dalších dostupných ukazatelů a následně ověřená pilotem.

**Možná intervence:**
- nabídnout přímý odkaz nebo jasný postup objednání;
- pomoci s vyhledáním pracoviště či vhodného termínu;
- zkrátit počet kroků mezi pozvánkou a objednáním.

**Omezení:** dataset podle dostupného popisu přímo neměří čekací doby, dopravní dostupnost ani skutečnou kapacitu pracovišť. Tyto bariéry nelze z regionální proměnné samy o sobě spolehlivě určit.

### 5. Klientky s nevyjasněnou překážkou nebo obavou

**Možná charakteristika:** klientky bez zaznamenané účasti, u kterých příčina není známa.

**Možná intervence:**
- nabídnout citlivé a nehodnotící informace;
- umožnit položit otázku nebo požádat o vysvětlení;
- případně nabídnout kontakt s pracovníkem, který pomůže s dalším postupem.

**Důležité:** neúčast sama o sobě není důkazem strachu, nedůvěry ani odmítání. Tyto důvody je potřeba zjistit, ne předpokládat.

### 6. Klientky s vyšší historickou pravděpodobností účasti

**Možná charakteristika:** skupina, u níž historická data naznačují vyšší pravděpodobnost účasti.

**Možná intervence:**
- standardní, srozumitelná pozvánka;
- přiměřená připomínka;
- nákladnější individuální asistenci soustředit tam, kde může mít větší dodatečný přínos.

**Poznámka:** predikovaná pravděpodobnost není jistota. Model by měl sloužit k prioritizaci a vyhodnocování, nikoli k odepírání informací nebo přístupu ke screeningu.

## 5. Jak bych navrhl pilot

Doporučil bych začít dvěma intervencemi a měřit, zda přinášejí **dodatečnou účast**, nikoli pouze více kontaktů.

### Pilot A: gynekologická prevence jako komunikační kanál

- **Cílová skupina:** klientky s preventivní aktivitou u gynekologa, ale bez zaznamenaného screeningu v relevantním období.
- **Kontrola:** standardní pozvánka.
- **Varianta:** standardní pozvánka doplněná o doporučení přes gynekologickou prevenci a jednodušší cestu k objednání.
- **Hlavní metriky:** účast do 3, 6 a 12 měsíců; absolutní rozdíl v procentních bodech; náklady na jednu dodatečnou účast.

### Pilot B: jiný přístup k opakovaně zvaným klientkám

- **Cílová skupina:** klientky s opakovanými pozvánkami a bez zaznamenaného screeningu.
- **Kontrola:** standardní připomínka.
- **Varianta:** osobnější kontakt nebo nabídka pomoci s objednáním.
- **Hlavní metriky:** stejné jako u pilotu A, případně také úspěšnost navázání kontaktu a dokončení objednání.

Pokud je to organizačně a eticky možné, rozdělil bych vhodné klientky náhodně do kontrolní a intervenční skupiny. Náhodné rozdělení pomůže lépe odlišit skutečný efekt intervence od rozdílů mezi klientkami. Předem je vhodné stanovit délku sledování, primární metriku, pravidla vyloučení a způsob práce s opakovanými pozvánkami.

### Jak interpretovat výsledek

Příklad čistě ilustrativního výsledku: pokud se účast zvýší z **10 % na 13 %**, jde o nárůst o **3 procentní body**, tedy přibližně **30 dodatečných účastí na 1 000 oslovených**. To samo o sobě ještě neříká, zda je intervence nákladově efektivní; je potřeba započítat její náklady a nejistotu odhadu.

## 6. Metodická upozornění před rozhodnutím

1. **Opakované řádky na jednu klientku.** Dataset obsahuje více pozvánek pro tutéž klientku. Řádky proto nejsou nezávislé. Při výpočtu nejistoty a při testování intervencí je potřeba zohlednit shlukování podle klientky, případně provést analýzu na úrovni klientek.
2. **Význam labelu.** `SCREENING_3M_FLAG`, `SCREENING_6M_FLAG` a `SCREENING_12M_FLAG` popisují zaznamenaný screeningový výkon po konkrétní pozvánce v příslušném časovém okně. Nejde o celoživotní historii screeningu.
3. **„Nikdy nescreenována“ může být příliš silné tvrzení.** Pokud `NEVER_SCREENED_BEFORE_FLAG` vychází z pětiletého lookback okna, je přesnější interpretovat jej jako „bez zaznamenaného screeningu v posledních pěti letech“, nikoli jako „nikdy v životě nebyla na screeningu“.
4. **Zkontrolovat konzistenci horizontů.** Pokud jsou labely definovány kumulativně stejným způsobem, mělo by platit `SCREENING_3M_FLAG ≤ SCREENING_6M_FLAG ≤ SCREENING_12M_FLAG` pro každý řádek. Případné rozpory je potřeba prověřit.
5. **Asociace není příčina.** Korelace a důležitost proměnné v modelu samy neprokazují, že daný faktor způsobuje účast.
6. **Region není přímá dostupnost péče.** `VZP_TERRITORIAL_AREA` může signalizovat rozdíly mezi oblastmi, ale bez dalších údajů nevysvětluje jejich příčinu.
7. **Ochrana soukromí a férovost.** Segmentaci je vhodné používat pro nabídku podpory a zjednodušení přístupu, nikoli k omezování péče nebo k stigmatizujícímu označování klientek.

## 7. Doporučené další kroky

1. Doplnit detailní profily clusterů a míry účasti podle kategorií, zejména pro `AGE_BAND` a `FIRST_INVITATION`.
2. Převést pracovní segmenty na jednoznačná pravidla a spočítat jejich velikost, překryvy a míru účasti.
3. Ověřit kvalitu a konzistenci labelů a způsob, jakým jsou přiřazeny opakovaným pozvánkám.
4. Vybrat dvě cílové skupiny pro pilot a předem definovat kontrolní skupinu, primární metriku a náklady.
5. Vyhodnotit nejen účast, ale i dodatečný přínos vůči kontrolní skupině a náklady na dodatečnou účast.

Pro detailnější interpretaci segmentace by byly užitečné soubory `cluster_profile.csv`, `categorical_rates__SCREENING_12M_FLAG.csv` a `rule_segments__SCREENING_12M_FLAG.csv`.

---

## Shrnutí doporučení

Nejzajímavější příležitost je spojit **preventivní aktivitu u gynekologa** s jednodušší cestou k mamografickému screeningu a zároveň změnit přístup k **opakovaně zvaným klientkám bez zaznamenané účasti**. Tyto dvě skupiny jsou vhodnými kandidáty pro pilot, ale efekt je nutné ověřit kontrolovaným testem. Při interpretaci výsledků je zásadní zohlednit opakované pozvánky, správně chápat časová okna labelů a nepřisuzovat klientkám důvody neúčasti, které data přímo neobsahují.
