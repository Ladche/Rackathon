# Kdo potřebuje více než standardní pozvánku na mamografický screening?

## Metareport čtyř analýz pro Rakathon 2026

**Datum:** 17. 10. 2026  
**Rozhodovací otázka:** Které adresně oslovené klientky VZP mají nejvyšší riziko neúčasti, komu nabídnout intenzivnější podporu a jak prokázat její dodatečný přínos?

> **Hlavní doporučení:** Propojit predikční skóre s vysvětlitelnými pravidly segmentace a klinickou kontrolou vhodnosti. Pilotně ověřit podporu přes ordinaci u početné skupiny opakovaně zvaných žen s lékařskou prevencí a navigační podporu u žen s minimálním kontaktem s péčí. Historická neúčast určuje potřebu podpory, nikoli její prokázanou účinnost.

---

## 1. Manažerské shrnutí

1. **KDO:** Nejvyšší prakticky relevantní historické riziko mají ženy s alespoň dvěma staršími pozvánkami bez zaznamenané účasti v již uzavřených ročních oknech, zejména při absenci další prevence. Ve skupině s nízkým kontaktem s péčí byla účast do 180 dní jen **1,0 %**, do 365 dní **2,5 %**. Nízká účast ale neprokazuje odmítání, strach ani nedoručení.
2. **PREDIKCE:** Notebook dokládá model XGBoost s **ROC-AUC 0,8224** a CatBoost s **0,8211** pro účast do 365 dní. Jde o průzkumné výsledky při náhodném dělení pozvánek, nikoli o ověřenou výkonnost při budoucím nasazení. Původní sloupec `RISK_SCORE` znamená skóre **účasti**; skóre neúčasti je jeho doplněk do jedné.
3. **PROČ / SOUVISLOSTI:** Napříč metodami se opakují gynekologická prevence, historie a recence mamografie, věk, jiná prevence a kontakt s péčí. Počet starších pozvánek má zápornou asociaci s následnou účastí. Nejde o příčiny neúčasti ani o důkaz, že změna atributu změní chování.
4. **SEGMENTY:** Detailní segmentační podklad popisuje **256 466 žen v roce 2024**, každou jednou. Čtyři segmenty s opakovaně nulovou historickou účastí představují **146 445 žen, tedy 57,1 % této kohorty**, a soustřeďují **65,7 % jejích záznamů bez screeningu do 365 dní**. Jde o objem pro prioritizaci, nikoli o počet účastí, které lze intervencí získat.
5. **CO NABÍDNOUT:** Propojit podporu s doloženým kontaktem s ordinací; u ostatních nabídnout srozumitelnou informaci, navigátorku a jednoduché telefonní či online objednání. Standardní pozvánka a běžná péče zůstávají dostupné všem.
6. **JAK OVĚŘIT:** Dva oddělené randomizované A/B piloty, rozdělení žen 1:1, primárně **zaznamenaný screening do 180 dní**, sekundárně do 90 a 365 dní. Rozhodovat podle rozdílu proti souběžné kontrole, nejistoty, nákladů a bezpečnosti, nikoli podle počtu odeslaných zpráv.
7. **BEZPEČNOST:** **7 600 žen** má uvedenou mamografii před méně než 24 měsíci. Nejdříve ověřit přesné datum a další vhodný termín; nízkou pravděpodobnost dalšího výkonu nelze automaticky řešit intenzivnějším zvaním.

**Co je doloženo a co nikoli:** Doloženy jsou uložené výsledky modelů, asociace a četnosti segmentů. Není doložena kalibrace pravděpodobností, nezávislá časová validace, dosažitelnost digitálními kanály ani účinek navržených intervencí.

## 2. Zadání, zdroje a způsob syntézy

Prezentace výslovně rozlišuje tuto výzvu od „no-show“ na objednaný termín. Cílem je screening v definovaném období **po adresném oslovení**, nikoli docházka na známou rezervaci. Požaduje model/skóre, hlavní faktory, srozumitelné segmenty, intervence a experiment. Tento report všechny tyto části propojuje. [Z]

### 2.1 Vstupní podklady

| Značka | Podklad | Přínos a omezení |
|---|---|---|
| [Z] | [Rakathon.pdf](./Rakathon.pdf) | Závazné zadání a kontext; prezentační statistiky mají jiný základ než analytický extrakt. |
| [D] | [Rakathon_Datove_rozhrani.docx](./Rakathon_Datove_rozhrani.docx) | Granularita, výběr kohorty, časová okna a význam prediktorů i labelů. |
| [A1] | [Honza report.md](./Honza%20report.md) | Popis celého extraktu, asociace, regrese, důležitosti Random Forest a pět clusterů. |
| [A2] | [Honza report 2.md](./Honza%20report%202.md) | Praktická interpretace, pracovní segmenty a návrhy pilotů; z velké části navazuje na stejné výsledky jako [A1]. |
| [A3] | [message.txt](./message.txt) | Detailní pravidlová segmentace roku 2024, počty, účast, komunikační návrhy a experimentální omezení. |
| [A4] | [program.ipynb](./program.ipynb) | Příprava vstupů, XGBoost/CatBoost, uložené metriky a vícemetodová vysvětlitelnost. |

**Pravidlo syntézy:** Definice přebíráme z [D], požadavky ze [Z], souhrnné četnosti z [A1], segmentové četnosti z [A3] a výsledky modelů z [A4]. [A2] pomáhá převést zjištění do praxe. Shoda podkladů není nezávislá replikace: pracují se stejnými nebo odvozenými daty. Neprůměrujeme nesrovnatelné kohorty ani různé typy důležitosti.

**Rozsah ověření:** Metareport vznikl čtením dodaných podkladů a uložených výstupů notebooku. Součty a vybrané odvozené ukazatele byly přepočítány. Ve workspace nejsou vstupní řádková data ani pomocný modul vysvětlitelnosti odkazovaný notebookem; modely ani přiřazení segmentů proto nebyly znovu spuštěny. Níže odlišujeme doložený výsledek, jeho interpretaci a návrh dalšího postupu.

## 3. Co přesně měříme a na jakých datech

### 3.1 Jednotka analýzy a cíl

- **T0:** datum odeslání pozvánky.
- **Řádek:** jedna pozvánka jedné klientce, nikoli jedna jedinečná žena za celé období.
- **Účast:** zaznamenaný kvalifikující screeningový výkon do 90 / 180 / 365 dní po T0. Labely `SCREENING_3M_FLAG`, `SCREENING_6M_FLAG`, `SCREENING_12M_FLAG` mají hodnotu 1 pro účast.
- **Neúčast v tomto reportu:** absence takového záznamu v daném okně; ne důkaz vědomého odmítnutí nebo celoživotní neúčasti.
- **Predikční horizont doloženého modelu:** 365 dní. **Primární horizont navrženého experimentu:** 180 dní, aby byl použitelný pro dřívější rozhodnutí; 365 dní zůstává důležitou sekundární kontrolou. Model pro 365 dní není automaticky modelem pro 180 dní.

### 3.2 Celý analytický extrakt

| Ukazatel | Hodnota |
|---|---:|
| Pozvánky | 1 344 556 |
| Unikátní klientky | 508 527 |
| Období pozvánek | 14. 1. 2020 až 6. 12. 2024 |
| Klientky s více pozvánkami | 317 557 |
| Podíl řádků náležejících klientkám s více pozvánkami | 85,8 % |
| Screening do 90 dní | 95 769 / 1 344 556 = 7,12 % |
| Screening do 180 dní | 168 621 / 1 344 556 = 12,54 % |
| Screening do 365 dní | 259 536 / 1 344 556 = 19,30 % |

Zdroj: [A1], velikost kohorty potvrzuje [D]. Přibližně jedna z pěti **pozvánek** je spojena se screeningem do roka; nejde o podíl všech žen, které někdy absolvovaly screening.

Datové rozhraní popisuje výběr přibližně 1 504 546 pozvánek, vyřazení 159 535 kvůli podmínce nepřetržitého pojištění u VZP v letech 2015–2025 a následné odstranění 455 potenciálně identifikujících kombinací. Výsledkem je 1 344 556 řádků. Tato selekce omezuje přenositelnost na jiné pojištěnky, nově pojištěné a obecnou populaci. [D]

### 3.3 Tři různé statistické základy: nezaměňovat

| Základ | Počet | Účast a interpretace |
|---|---:|---|
| Prezentace, pozvánky v roce 2024 | 292 176 | Uvádí celkovou účast 18,9 %, u poprvé oslovených 44,5 % a při dlouhodobém opakování 3,2 %. Na snímku není konkrétní horizont ani úplná definice podskupin. [Z] |
| Celý analytický extrakt, roky 2020–2024 | 1 344 556 pozvánek | Účast do 365 dní 19,30 %. [A1] |
| Součet segmentů roku 2024 | 256 466 žen / pozvánek | Účast do 90 dní 18 180 = 7,09 %; do 180 dní 31 837 = 12,41 %; do 365 dní 49 265 = 19,21 %. Každá žena je podle podkladu uvedena jednou. [A3], přepočet |

Rozdíl mezi prezentačním počtem a segmentační kohortou je **35 710**. Rozdílná selekce je možné vysvětlení, nikoli doložený rozpad tohoto rozdílu. Bez řádkových dat a přesných filtrů nelze ověřit, zda jej celý vysvětlují podmínky extraktu. Nepřebíráme proto 18,9 % jako kontrolní míru pilotu a 3,2 % jako míru našich segmentů.

## 4. KDO: predikce rizika neúčasti

### 4.1 Doložené modely a výsledky

Notebook používá 91 vstupů po zakódování: historii mamografie, kontakt s péčí, délku pojištění, registrace, preventivní výkony, věkové pásmo a územní pracoviště VZP. Cílem je `SCREENING_12M_FLAG`. Zdrojový příznak první pozvánky ani odvozená historie reakcí nejsou v uvedené matici těchto modelů; doplňuje je pravidlová vrstva. [A4]

| Model | Uložená ROC-AUC | Co lze říci |
|---|---:|---|
| XGBoost | 0,8224 | O něco vyšší AUC v daném testu; pracovní kandidát pro další validaci. |
| CatBoost | 0,8211 | Velmi podobný výsledek; rozdíl 0,0013 bez intervalu nejistoty nedokládá jednoznačnou převahu XGBoostu. |

Data byla rozdělena náhodně na 80 % trénink / 20 % test; test má **268 912 pozvánek**, z toho **216 866 bez účasti** a **52 046 s účastí**. Ladění používá třífoldovou křížovou validaci. U XGBoostu jsou jako nejlepší uloženy parametry 300 stromů, hloubka 6, learning rate 0,05 a subsample i colsample_bytree 1,0. [A4]

**Význam AUC:** Jde o schopnost seřadit případy podle skóre, nikoli o 82% přesnost, kalibraci pravděpodobností ani předpokládaný přínos intervence. Při přechodu od účasti k neúčasti se musí současně obrátit label i skóre; takto vypočtená AUC zůstává stejná.

### 4.2 Oprava směru skóre a interpretace klasifikace

V původních buňkách notebooku platí:

```text
PARTICIPATION_SCORE_365 = predict_proba(X)[třída 1]
NONPARTICIPATION_SCORE_365 = 1 - PARTICIPATION_SCORE_365
```

Původní `RISK_SCORE` odpovídá prvnímu řádku, tedy **vyšší hodnota znamená vyšší skóre účasti**, nikoli vyšší riziko neúčasti. Novější část vysvětlitelnosti již pracuje se správným směrem neúčasti. Při implementaci musí být index sloupce ověřen přes pořadí tříd modelu.

Navíc modely používají vážení tříd (`scale_pos_weight=1.5`, resp. automatické vyvážení). Jejich syrový výstup proto označujeme jako **skóre**, ne jako ověřenou absolutní pravděpodobnost. Teprve po kalibraci na vhodných validačních datech lze reportovat `P(neúčast do 365 dní)`.

Detailní klasifikační report pochází z **CatBoostu**, nikoli z vítězného XGBoostu. Při prahu 0,5 pro skóre účasti uvádí:

| Třída | Precision | Recall | F1 |
|---|---:|---:|---:|
| Neúčast | 0,93 | 0,77 | 0,84 |
| Účast | 0,43 | 0,74 | 0,55 |

Správná interpretace: mezi pozvánkami označenými modelem jako neúčast bylo přibližně **93 % skutečně bez zaznamenané účasti** a zachyceno bylo přibližně **77 % všech neúčastí** v tomto testu. Nejde o prokázanou účinnost připomínek ani o důkaz, že kontaktování je „zbytečné“ u ostatních. Textová poznámka notebooku uvádí 76 %; přebíráme 77 % z uložené tabulky. Celková accuracy 0,76 není vhodná hlavní metrika: samotné předpovídání neúčasti všem by při této prevalenci mělo accuracy přibližně 80,6 %.

### 4.3 Použitelné pravidlové skóre pro segmenty

Vedle individuálního modelu lze již z dodaných agregací jednoznačně definovat popisné segmentové skóre:

```text
HISTORICAL_NONPARTICIPATION_RATE_h(segment)
    = 1 - počet screeningů do h dní / počet žen v segmentu
```

Při horizontu 365 dní tak činí přibližně **97,5 %** pro `repeated_zero__low_contact`, **95,7 %** pro `repeated_zero__care_no_prevention`, **93,9 %** pro `repeated_zero__other_prevention` a **90,8 %** pro `repeated_zero__medical_prevention`. Všechny jmenovatele i čitatele uvádí příloha A.

Jde o historickou skupinovou četnost, **ne individuální predikci ověřenou mimo vzorek**. Lze ji použít jako transparentní referenční skóre pro budoucí porovnání s modelem. Nesmí překonat klinickou kontrolu: skupina `recent_screen` má rovněž vysokou četnost bez dalšího výkonu, ale jinou potřebu podpory.

### 4.4 Co ověřit před použitím individuálních pravděpodobností

1. **Rozdělení podle času a klientky:** náhodné dělení pozvánek umožňuje, aby tatáž žena byla v tréninku i testu, a mísí roky. Navržený postup: trénink na dřívějších letech, validace pro ladění a kalibraci v dalším období, konečný uzamčený test na roce 2024. U každého kroku povolit jen labely, jejichž okno a vykazovací prodleva již k danému okamžiku uplynuly.
2. **Dva odlišné scénáře generalizace:** zvlášť vyhodnotit budoucí pozvánky známým klientkám a skupinově oddělený test dosud neviděných klientek. Časový test může obsahovat známé ženy, pokud je to zamýšlený provozní scénář; nesmí však používat jejich budoucí informace.
3. **Nový nezávislý test:** vítěz byl vybrán podle nynější testovací AUC, takže tato sada již není zcela nedotčeným závěrečným testem. Uložené AUC nelze zpětně prohlásit za výsledek navržené přísnější validace.
4. **Předzpracování pouze z tréninku:** imputace a kódování jsou nyní připraveny před dělením. V provozní pipeline se musí naučit jen na tréninku. Chybějící label nesmí přejít na neúčast, chybějící prediktor automaticky na skutečnou nulu.
5. **Kalibrace a rozhodovací metriky:** doplnit kalibrační křivku, Brier score, PR-AUC pro neúčast, precision/recall a lift při reálné kapacitě podpory, intervaly nejistoty a výsledky podle roku, věku a regionu. Nejsou nyní doloženy.
6. **Zvolený práh:** určit na validaci podle kapacity a nákladů. Vyšší práh skóre neúčasti zpravidla zmenší výběr; pro větší záchyt neúčastí jej snižujeme. Při prahování původního skóre účasti je směr opačný. Výběr prahu neladit na závěrečném testu.

**Závěr k predikci:** Existuje slibný doložený model a explicitní skórovací pravidlo. Neexistuje zatím důkaz produkční kalibrace, generalizace ani hotový ověřený export individuálních rizik. Skóre neslouží k odepření péče.

## 5. PROČ: faktory související s účastí a jejich vysvětlení

### 5.1 Robustní společné závěry

| Faktor | Doložený výsledek | Praktický závěr a hranice interpretace |
|---|---|---|
| Gynekologická prevence | Spearmanovo rho s účastí do 365 dní **0,3308**; průměr 1,083 pro účast vs. 0,424 bez účasti. Ve 2 000řádkovém SHAP vzorku XGBoostu nejvyšší průměrný absolutní příspěvek **0,552**. | Kandidát pro oslovení přes ordinaci. Neprokazuje účinek dodatečné preventivní prohlídky. |
| Historie pozvánek | Počet předchozích pozvánek rho **−0,2537**; průměr 0,664 pro účast vs. 1,482 bez účasti. | Odlišit opakovaně nulové výsledky od pouhého příznaku opakované pozvánky. |
| První pozvánka | Cramérovo V pro 365 dní **0,3598**, nejsilnější z uvedených kategoriálních asociací. | Samotné V nedává směr; [A3] doloží vysokou účast konkrétní první skupiny s lékařskou prevencí, nikoli všech poprvé zvaných. |
| Šíře prevence | Počet aktivních oblastí rho **0,3098**; součet výkonů rho **0,2999**; jiný screening rho **0,2552**. | Rozlišit absenci mamografie od absence veškeré sledované prevence. Součty propojují různá časová okna a nejsou klinickým skóre. |
| Historie a recence mamografie | Společná permutace korelované skupiny historie/recence ve vzorku 400 řádků snížila AUC o **0,0922**. | Silný predikční signál, ale recentní screening může znamenat správné načasování další péče, ne problém k „nápravě“. |
| Věk | Cramérovo V **0,2873**; společná permutace věkových kategorií snížila AUC o **0,0712**. | Použít pro stratifikaci a kontrolu férovosti, ne jako samostatný důvod neposkytnutí podpory. |
| Kontakt s péčí a registrace | Kontakt rho **0,1519**; společná permutace kontaktu snížila AUC o **0,0143**. | Pomáhá volit cestu podpory; počet kontaktů neurčuje diagnózu ani dostupnost konkrétní ordinace. |
| Region | Random Forest přisuzuje územnímu pracovišti souhrnnou důležitost **0,1314** při 75 kategoriích. Skupinová permutace v notebooku jen **0,0031** poklesu AUC. | Regionální bariéra je hypotéza. Bez kapacit, dojezdů a čekacích dob není doložena. |

Zdroje: asociace a Random Forest [A1], modelové vysvětlení [A4], interpretace [A2]. Spearmanovo rho, Cramérovo V, SHAP a permutační pokles AUC měří různé věci; jejich velikosti nelze sčítat ani přímo porovnávat.

Rozdíl v regionální důležitosti není třeba násilně „sjednotit“: jde o jiné modely, vstupy a metriky. Impurity importance zvýhodňuje proměnné s mnoha kategoriemi; navíc permutace pochází z malého vzorku a regionální pořadí je nejisté. Praktické doporučení proto stojí především na prevenci a historii, nikoli na regionálním žebříčku.

### 5.2 Co navíc přináší vícemetodová vysvětlitelnost

- **Konsenzus pořadí:** v notebooku vedou recence screeningu a gynekologická prevence, následují věkové kategorie, jiný screening a kontakt s péčí. Samostatné dummy kategorie mohou vytvářet neplatné kombinace, proto pro věk a region upřednostňujeme skupinové permutace.
- **Stabilita:** v rozšířeném SHAP vzorku 400 řádků zůstává prvních deset uvedených vstupů v TOP 10 ve všech 100 bootstrapech. To dokládá stabilitu v tomto vzorku, nikoli stabilitu při novém tréninku nebo v jiném roce.
- **Tvar vztahu:** ALE pro gynekologickou prevenci má v nejnižším intervalu kladný centrovaný příspěvek k neúčasti přibližně +0,025, ve vyšších intervalech přibližně −0,041 až −0,043. Obdobně klesá příspěvek s jiným screeningem. Jsou to modelové křivky, nikoli účinky intervence.
- **Testované interakce:** nejsilnější ze tří zkoumaných dvojic je gynekologická prevence × jiný screening, RMS neaditivity 2D PDP **0,0116** na pravděpodobnostní škále. Nejde o statistický test ani prohledání všech dvojic.
- **Náhradní vysvětlení:** malý strom napodobuje model s R² **0,675** a MAE **0,080**. Lokální lineární aproximace má R² přibližně **0,47–0,71** a někdy vrací hodnotu nad 1. Je to přibližné vysvětlení chování modelu, nikoli validní nový pravděpodobnostní model.
- **Směr lokálního SHAP:** pro vysvětlovanou neúčast zvyšují kladné příspěvky modelové riziko. Absence gynekologické prevence a kontaktu s péčí ho u ukázkového vysoce rizikového profilu zvyšují; u jiného profilu gynekologická prevence přispívá opačně. SHAP je zde v log-odds, ne v procentních bodech.

U recence je zvlášť nutná kontrola: notebook imputuje chybějící hodnoty maximem a uložené vysvětlení obsahuje hodnotu 76 měsíců, zatímco rozhraní popisuje pětileté okno. Bez původních dat nelze rozlišit skutečný údaj, odlišný výpočet a imputovaný kód. Nevyvozujeme proto jednoduché pravidlo „čím déle od mamografie, tím vždy vyšší riziko“ ani klinický závěr z hranic náhradního stromu. [D, A4]

### 5.3 Co zůstává neznámé

Data přímo neměří obavy, pracovní povinnosti, důvěru, zdravotní gramotnost, digitální preference, doručení pozvánky, existující rezervaci ani skutečnou čekací dobu. Získat je lze krátkým dobrovolným dotazem a doplněním provozních dat. Formulace „odmítá“, „nechce“ nebo „má dopravní bariéru“ nejsou z uvedených atributů oprávněné.

## 6. SEGMENTY: srozumitelná pravidla místo neověřených person

### 6.1 Rozhodovací logika při T0

Pravidlová segmentace [A3] je pro komunikaci vhodnější než pouhé číslo clusteru. Používá dvě osy a předřazenou kontrolu:

**Krok 0 — bezpečnost:** pokud je `MONTHS_SINCE_LAST_SCREEN < 24`, zařadit do `recent_screen` a ověřit vhodný termín. Podmínka není automatickou klinickou výjimkou; má přednost před dalšími pravidly.

**Krok 1 — jen časově známá historie reakcí:** pro starší pozvánku počítat výsledek pouze tehdy, když před aktuálním T0 uplynulo 365 dní plus použitá 30denní rezerva. Rezerva je analytické pravidlo [A3], ne doložená garance úplnosti vykazování.

| Historická skupina | Pravidlo |
|---|---|
| `past_response` | Alespoň jedna účast v takto uzavřeném starším okně. |
| `repeated_zero` | Alespoň dvě taková známá okna, žádné s účastí. |
| `single_zero` | Právě jedno takové známé okno bez účasti. |
| `unobserved_first` | Žádné takové známé okno a zdroj uvádí první pozvánku. |
| `unobserved_repeat` | Žádné takové známé okno a zdroj uvádí opakovanou pozvánku. |

**Krok 2 — prevence a kontakt s péčí, v tomto pořadí:**

| Profil | Jednoznačné pravidlo při úplných vstupech |
|---|---|
| `medical_prevention` | Alespoň jedna preventivní prohlídka u praktika nebo gynekologa za 24 měsíců. |
| `other_prevention` | Nula u praktika i gynekologa; alespoň jedna stomatologická prevence za 24 měsíců nebo jiný screening za 36 měsíců. |
| `care_no_prevention` | Nula ve všech čtyřech oblastech a více než dva kontakty typu poskytovatel/den za 12 měsíců. |
| `low_contact` | Nula ve všech čtyřech oblastech a nejvýše dva takové kontakty za 12 měsíců. |

Segment vzniká spojením obou os; u `past_response` jsou poslední dva profily v podkladu sloučeny jako `no_prevention` kvůli malé četnosti. **Chybějící či rozporné vstupy nesmějí tiše znamenat nulu:** v budoucí implementaci patří k datovému ověření, ne k automatickému označení nízké prevence. Stejně postupovat při neznámém příznaku první pozvánky.

Důležité rozlišení: `unobserved_repeat` není opakovaná neúčast a první pozvánka podle zdroje není první celoživotní zkušenost s mamografií.

### 6.2 Čtyři prioritní skupiny s opakovaně nulovou účastí

Všechny níže mají alespoň dvě starší známá roční okna bez účasti a nejsou v předřazené skupině nedávného screeningu.

| Segment | Ženy 2024 | Podíl kohorty | Účast do 180 dní | Účast do 365 dní | Bez záznamu do 365 dní |
|---|---:|---:|---:|---:|---:|
| S1: prevence u praktika či gynekologa | 78 213 | 30,5 % | 5,4 % | 9,2 % | 71 027 |
| S2: zubní prevence či jiný screening, ne lékařská prevence | 31 057 | 12,1 % | 2,9 % | 6,1 % | 29 177 |
| S3: kontakt s péčí bez sledované prevence | 20 861 | 8,1 % | 2,1 % | 4,3 % | 19 971 |
| S4: nízký kontakt bez sledované prevence | 16 314 | 6,4 % | 1,0 % | 2,5 % | 15 904 |
| **Celkem** | **146 445** | **57,1 %** | **3,9 %** | **7,1 %** | **136 079** |

Zdroj: [A3], součty a souhrnné míry přepočteny z čitatelů. Roční kohorta má celkem 207 201 záznamů bez screeningu do 365 dní; uvedených 136 079 tvoří **65,7 %**. Součet zaokrouhlených dílčích podílů se může lišit od celku.

Popisné 95% Wilsonovy intervaly účasti do 180 dní jsou S1 **5,3–5,6 %**, S2 **2,7–3,1 %**, S3 **1,9–2,3 %**, S4 **0,9–1,2 %**. Nezahrnují neúplnost záznamů ani selekční zkreslení. V roce 2023 byly míry těchto skupin přibližně 5,5 / 3,2 / 2,2 / 1,3 %; podobný vzorec podporuje jejich praktickou relevanci, není však nezávislou replikací, protože se ženy mezi roky opakují. [A3]

**Priorita není totéž co nejvyšší riziko:** S4 má nejnižší účast, S1 však největší objem a existující kontakt s prevencí. O skutečném dodatečném přínosu podpory rozhodne experiment.

### 6.3 Další užitečné skupiny

- **První pozvánka + lékařská prevence:** 32 478 žen; účast do 180 dní **42,4 %**, do 365 dní **60,4 %**. Zde dává smysl začít nízkonákladovou orientací a jednoduchým objednacím krokem, ne automaticky intenzivní navigací.
- **Opakovaná pozvánka bez vyhodnotitelné historie + lékařská prevence:** 24 033 žen; **23,9 % / 36,2 %**. Neoznačovat za dlouhodobě nereagující.
- **Jedno starší nulové okno + lékařská prevence:** 14 890 žen; **15,0 % / 23,5 %**. Jedna historická neúčast není totéž co opakovaný vzorec.
- **Dřívější účast + lékařská prevence:** 6 592 žen; **25,8 % / 38,9 %**. Navázat na dřívější zkušenost a ověřit správné načasování, ne předpokládat, že současná účast je zaručena.
- **Nedávná mamografie:** 7 600 žen; **2,2 % / 2,4 %**. Medián recence je 0 měsíců; možné zaokrouhlení či souběh výkonu a zvaní vyžadují kontrolu přesných dat.

Čísla vždy znamenají účast do 180 / 365 dní. Úplný soupis všech 20 skupin je v příloze A.

### 6.4 Jak využít pět statistických clusterů

[A1] rozděluje celý extrakt do pěti clusterů s účastí do 365 dní přibližně **6,09 / 47,96 / 20,93 / 13,45 / 8,20 %**. Rozdíl mezi clustery 1 a 2 je zhruba 27 procentních bodů, přestože oba mají jako nejčastější pásmo součtu prevence 4–6.

To podporuje potřebu vícerozměrné segmentace. Podklady ale neobsahují úplné profily potřebné k reprodukci přiřazení. **Cluster 0 není automaticky S4** a častá charakteristika clusteru neplatí pro každou jeho členku. Clustery ponecháváme jako průzkumný důkaz heterogenity, nikoli jako hotová komunikační pravidla.

## 7. CO: konkrétní podpora podle segmentu

### 7.1 Intervenční matice

| Segment / situace | Hypotéza k ověření, ne zjištěný důvod | Nabídka podpory | Cesta kontaktu |
|---|---|---|---|
| S1: opakovaně bez účasti, lékařská prevence | Existující preventivní kontakt může usnadnit návaznost na mamografii. | Doporučení vlastní ordinace, jediný jasný objednací krok, možnost probrat otázky. **Pilot P1.** | Praktik/gynekolog podle skutečně dostupné vazby; dopis, oprávněný telefon či digitální kanál. |
| S2: jiná prevence | Zkušenost s jinou prevencí nemusí zahrnovat orientaci v mamografii. | Krátký rozhodovací průvodce průběhem, přínosy a omezeními; přímé objednání a prostor pro dotazy. | Srozumitelný dopis; podle preference zabezpečený portál nebo další oprávněný kanál. |
| S3: péče bez prevence | Existující péče nabízí příležitost ke konzultaci. | Krátký rozhovor při kontaktu s ordinací a nabídka pomoci s termínem; jinak navigátorka. | Ordinace po potvrzení spolupráce, alternativně navigační linka. |
| S4: minimální kontakt | K účasti může pomoci snazší vstup do péče a společné objednání. | Jednoduchá informace o screeningu i bez příznaků, nabídka zpětného volání a asistovaného objednání. **Pilot P2.** | Dopis s dostupnou bezplatnou linkou; oprávněný telefon/SMS podle ověření kontaktu. |
| První pozvánka / vyšší historická účast | Části žen může stačit jasná orientace. | Standardní pozvánka s konkrétním postupem a možností vyžádat pomoc. | Preferovaný dostupný kanál, vždy offline alternativa. |
| Dřívější účast, nyní bez aktuálního screeningu | Může jít o prodlení, změnu situace či jiný vhodný termín. | Neutrální připomenutí, kontrola termínu, možnost konzultace a objednání. | Dopis, ordinace nebo preferovaný oprávněný kanál. |
| `recent_screen` | Další screening nemusí být právě splatný. | Ověřit datum, indikaci a doporučený další termín; pak osobní plán. | Zabezpečená komunikace nebo rovnocenná offline cesta. |
| Podezření na regionální bariéru | Bariéra nebyla přímo změřena. | Nejdříve zjistit čekací dobu/dojezd; poté nabídnout konkrétní dostupné pracoviště a termín. | Navigace podle ověřených provozních dat, ne pouze názvu regionu. |

### 7.2 Společná komunikační politika

1. **Při T0** zkontrolovat klinickou vhodnost, známý aktuální výkon a existující rezervaci, pokud jsou tyto informace dostupné. Nabídnout podporu bez stigmatizace a bez zdravotního nátlaku.
2. **Kolem 30. dne** nejvýše jeden předem definovaný navazující kontakt v pilotním balíčku, pokud situace není vyřešena a kontakt je vhodný. Absence vykázaného výkonu kvůli prodlevě sama neprokazuje potřebu další výzvy.
3. **Zastavení dalších výzev:** po známém výkonu, potvrzené rezervaci řešit pouze potřebné organizační informace; respektovat odhlášení a individuální klinické doporučení.
4. **Rovnocennost kanálů:** SMS, e-mail ani aplikaci nelze předpokládat z tohoto datasetu. Ověřit kontakt, oprávnění a preferenci; bez digitální dostupnosti musí existovat telefonní nebo písemná cesta.
5. **Soukromí:** zdravotní historii ani segment nepsat do nezabezpečené notifikace. Před sdělením zdravotních podrobností telefonicky ověřit totožnost.

**Návrh obecného textu pro dopis po schválení klinickým a právním garantem:**

> Nabízíme Vám pomoc s objednáním na mamografický screening, pokud je pro Vás nyní vhodný. Můžete využít objednací kontakt uvedený v dopise nebo požádat o pomoc s výběrem termínu. Pokud již máte termín nebo jste vyšetření nedávno absolvovala, nejprve s námi ověřte další postup.

Do finální šablony doplnit skutečné kontakty a dostupné služby; report nepředpokládá, že jsou již zřízené.

## 8. JAK OVĚŘIT: dva prioritní A/B piloty

### 8.1 Cílové skupiny a ramena

| Pilot | Cílová skupina při T0 | A: kontrola | B: intervence |
|---|---|---|---|
| **P1 — návaznost na ordinaci** | S1, po kontrole vhodnosti a provozní dosažitelnosti. Historický základ 78 213 žen. | Stávající adresná pozvánka a běžná péče. | Stejný standard plus potvrzené doporučení ordinace, jednoduchý objednací krok a nabídka pomoci telefonicky nebo online. |
| **P2 — navigační podpora** | S4, po stejné kontrole. Historický základ 16 314 žen. | Stávající adresná pozvánka a běžná péče. | Stejný standard plus nabídka navigátorky, zpětného volání a společného objednání; offline možnost požádat o pomoc. |

P1 ověřuje objemnou skupinu s existujícím preventivním kontaktem, P2 skupinu s nejnižší doloženou účastí mimo kontrolu nedávného screeningu. Každý testuje **celý podpůrný balíček**, ne izolovaný účinek SMS, lékaře a objednání. Konkrétní kanál i záložní cestu určit podle předem zjištěných oprávnění a preferencí; neměnit je zpětně podle výsledků.

Nejde o porovnání S1 proti S4 ani o porovnání roku 2023 proti roku 2024. Každý segment má vlastní souběžně randomizovanou kontrolu.

### 8.2 Randomizace a ochrana před zkreslením

- **Jednotka:** žena, rozdělení 1:1 při T0; segment, způsobilost a případné skóre zmrazit před přiřazením.
- **Stratifikace:** uvnitř pilotu předem stanovené širší věkové skupiny, region a období náboru; případně dostupnost kanálu, pokud to dovolí četnosti.
- **Trvalé rameno:** další pozvánka nesmí ženu znovu randomizovat. Žena nesmí současně vstoupit do druhého pilotu.
- **Pevná politika:** navazující kontakt kolem 30. dne je součástí přidělené strategie, nikoli nová nerandomizovaná skupina vybraná podle reakce.
- **Intention-to-treat (ITT):** všechny randomizované ženy analyzovat v původním rameni včetně nedoručených nabídek a nevyužité asistence.
- **Kontaminace:** sledovat podporu poskytnutou kontrolám. Pokud doporučení změní chování celé ordinace, zvážit randomizaci ordinací; potom přepočítat velikost i analýzu na shluky.
- **Stejné zjišťování výsledku:** zdroj výkonů a datum úplnosti musí být stejné pro A i B; pokud možno vyhodnocení bez znalosti ramene.

### 8.3 Výsledky a statistický plán

**Primární výsledek:** podíl randomizovaných žen se zaznamenaným kvalifikujícím screeningem do **180 dní od T0**.

**Primární efekt:** absolutní rozdíl `účast_B − účast_A` v procentních bodech, s intervalem spolehlivosti a počty v obou ramenech. Relativní nárůst je doplněk, ne náhrada absolutního efektu.

**Sekundární výsledky:**

- účast do 90 a 365 dní; rozlišit rychlejší účast od trvalého zvýšení celkové účasti;
- objednání a čas do objednání/výkonu až po doplnění termínových dat;
- úspěšnost kontaktu, nedoručení, využití asistence;
- dodatečné náklady, cena na jednu dodatečnou účast;
- odhlášení, stížnosti, neindikované duplicity a rozdíly v dostupnosti podpory;
- heterogenita podle věku, regionu a kanálu jako předem specifikovaná, jinak průzkumná analýza.

V segmentační kohortě vzniklo **17 428 z 49 265** ročních účastí až v intervalu 181–365 dní, tedy **35,4 %**. Proto samotný úspěch ve 180 dnech ještě nemusí znamenat vyšší účast po roce. Tento výpočet předpokládá správně kumulativní labely. [A3], přepočet

**Vícenásobné testování:** pro dva předem vybrané primární testy doporučujeme Bonferroniho rozdělení rodinné hladiny 0,05, tedy oboustranně **α = 0,025 na pilot**, a odpovídající **97,5% intervaly** pro potvrzující závěry. Sekundární ukazatele označit jako sekundární/průzkumné. Průběžné nahlížení nesmí vést k neplánovanému zastavení pro „významnost“.

Předem stanovit prakticky významný minimální přínos a rozpočtový limit. Škálovat jen při slučitelném důkazu dodatečné účasti, přijatelné ceně a bez nepřijatelného bezpečnostního či nerovnostního dopadu. Neprůkazný výsledek neznamená prokázanou nulovou účinnost.

### 8.4 Orientační velikost

Pro plánovací scénář **+2 procentní body**, sílu **80 %**, nezávislé ženy, alokaci 1:1 a α = 0,025 vychází normální aproximace dvou podílů bez korekce na kontinuitu:

| Pilot | Historická účast do 180 dní | Hypotetická účast B | Přibližně žen na rameno | Celkem |
|---|---:|---:|---:|---:|
| P1 | 4 253 / 78 213 = 5,44 % | 7,44 % | 2 862 | 5 724 |
| P2 | 164 / 16 314 = 1,01 % | 3,01 % | 933 | 1 866 |

**Celkem přibližně 7 590 žen.** Jde o nový plánovací přepočet, nikoli očekávaný výsledek. Zvlášť u P2 znamená +2 p. b. velký relativní nárůst; jeho dosažitelnost není doložena. Pro menší cílový efekt bude potřeba více žen.

Použitý vzorec pro počet na rameno:

```text
n = ceil(
  [z_(1−α/2) × sqrt(2 × p_bar × (1−p_bar))
   + z_(0,80) × sqrt(p_A × (1−p_A) + p_B × (1−p_B))]^2
  / (p_B−p_A)^2
)
p_bar = (p_A + p_B) / 2
```

Finální výpočet musí zohlednit současný baseline, skutečnou dostupnou populaci, neúplnost výsledků a případné shlukování. Při randomizaci ordinací použít nejméně korekci `1 + (průměrná velikost shluku − 1) × ICC` a zohlednit i různou velikost ordinací. Historická velikost segmentu není garance náboru po klinických a kontaktních filtrech.

[A3] uvádí jiný scénář: +5 p. b. a α přibližně 0,00263, což odpovídá konzervativnímu rozdělení 0,05 mezi 19 testů. Ten zde nepřebíráme, protože navrhujeme **dva prioritní piloty**, ne souběžně testovat všech 19 intervenčních segmentů.

### 8.5 Doba sledování a chybějící výsledky

- Finální primární vyhodnocení až **180 dní po poslední zařazené ženě plus ověřená vykazovací prodleva**. Výsledek do 365 dní obdobně.
- Neuzavřená okna nebo výpadek zdroje nejsou automaticky neúčast. Předem určit pravidla pro úmrtí, změnu pojištění, klinickou nezpůsobilost a chybějící sledování; transparentně uvést jejich výskyt podle ramen a citlivostní analýzy.
- Původní extrakt vybírá dlouhodobě nepřetržitě pojištěné; budoucí experiment nemusí mít stejně úplné sledování.
- Ženy s nedávným screeningem nyní do P1/P2 nezařazovat bez ověření. Jejich případný samostatný test má výsledek vztažený k individuálnímu doporučenému termínu, ne tlak na výkon do 180 dní od původní pozvánky.

## 9. Dopad a efektivní využití zdrojů

Vysoké riziko neúčasti neznamená vysoký účinek podpory. Dosavadní data dovolují prioritizaci podle potřeby a objemu, ne odhad individuálního kauzálního přínosu.

**Ilustrace, nikoli prognóza:** skutečný přírůstek +2 p. b. znamená 20 dodatečných účastí na 1 000 žen vystavených nové strategii. Pokud by se stejný efekt přenesl na všech 78 213 historických žen S1, šlo by přibližně o 1 564 dodatečných účastí; u 16 314 žen S4 o 326. Přenositelnost, nábor, vhodnost i kapacity jsou neověřené.

```text
Dodatečné účasti v intervenčním rameni
    = n_B × (účast_B − účast_A)

Cena na dodatečnou účast
    = dodatečné náklady B oproti srovnatelnému standardu
      / odhad dodatečných účastí
```

Do nákladů zahrnout práci ordinací a navigátorek, pokusy o kontakt, IT, objednávání a provozní koordinaci. Při nulovém nebo záporném odhadu přínosu neprezentovat kladnou „cenu za úspěch“. Vyhodnotit nejistotu nákladové efektivity.

Po úspěšném pilotu je vhodné přidělovat intenzivnější podporu podle očekávaného **dodatečného přínosu na jednotku nákladů**, nejen podle nejvyššího rizika. Model takového přínosu lze učit až z vhodných experimentálních dat.

## 10. Podmínky nasazení, limity a navazující kroky

### 10.1 Datové a metodické podmínky

| Oblast | Co ověřit / zajistit | Proč je to podstatné |
|---|---|---|
| Úplnost horizontů | Datum úplnosti vykázaných výkonů, rezerva a podmínka `T0 + horizont ≤ datum úplnosti`. | Podmínka pojištění do konce roku 2025 sama negarantuje úplné vykázání; neúplné okno nesmí být nula. |
| Kumulativní labely | Na každém řádku `3M ≤ 6M ≤ 12M`; ověřit duplicity a přiřazení výkonů k překrývajícím se oknům pozvánek. | Jedna událost může souviset s více pozvánkami; nejde o nezávislé účasti. |
| Časová dostupnost | Použít jen informace skutečně známé při T0, včetně zpoždění vykazování starší péče. | Retrospektivně doplněný údaj může vytvořit únik informace. |
| Historie mamografie | Přesná data, recence 0 i hodnoty nad popsaným pětiletým oknem. | Oddělit recentní péči, imputaci a definici zdroje. |
| Význam nul a chybění | Oddělit nulu, chybění a strukturální absenci posledního screeningu. | `NEVER_SCREENED_BEFORE_FLAG` bezpečně vykládat jako absenci v doloženém okně, ne jako celoživotní historii. |
| Region | VZP územní pracoviště není administrativní okres ani přímá dostupnost péče. | Bez doplnění nelze diagnostikovat dopravní či kapacitní bariéru. |
| Kontakty a rezervace | Oprávněné kontakty, preference, ordinace, rezervovaný termín, kapacita a čekací doba. | Bez nich není plnohodnotné směrování podpory připravené k provozu. |
| Soukromí a férovost | Pseudonymní ID pro analýzu, omezený přístup, právní a klinické schválení, audit podle věku/regionu/kanálu. | Podpora nesmí vést k odhalování historie, stigmatizaci nebo odepření standardní péče. |

### 10.2 Doporučený provozní tok

```text
Nová pozvánka při T0
  → kontrola vhodnosti, posledního výkonu a známé rezervace
  → kontrola kvality a časové dostupnosti vstupů
  → pravidlový segment + validované skóre neúčasti, pokud je k dispozici
  → nabídka podpory / přiřazení do předem schváleného experimentu
  → konkrétní objednací krok a případný jeden navazující kontakt
  → společné zjišťování výsledků a nákladů pro A i B
  → vyhodnocení dodatečné účasti, bezpečnosti a férovosti
```

Při nedostatečné kvalitě dat nebo neověřené vhodnosti předat k prověření, ne vygenerovat zdánlivě přesné riziko. Neexistence digitálního kontaktu nesmí znamenat neexistenci nabídky pomoci.

### 10.3 Co udělat v pořadí

1. **Datový garant:** doložit filtry kohort, úplnost labelů, časovou dostupnost a rozpory recence; reprodukovat součet 256 466 a pravidlové přiřazení na řádkových datech.
2. **Analytický tým:** opravit názvosloví skóre při implementaci, sestavit časově validovanou pipeline s kalibrací, porovnat model s historickým segmentovým skóre a doložit metriky při provozní kapacitě.
3. **Klinický a provozní garant:** potvrdit způsobilost, vhodné intervaly, spolupráci ordinací, skutečné objednací cesty a kapacity.
4. **Garant komunikace a ochrany dat:** schválit texty, kanály, oprávnění, pravidla zastavení a rovnocennou offline cestu.
5. **Garant experimentu:** předregistrovat P1/P2, primární efekt, velikost, analýzu a kritéria škálování; zahájit až po splnění předchozích podmínek.
6. **Po uzavření sledování:** rozhodnout o škálování podle randomizovaného efektu a nákladů, průběžně sledovat drift modelu a dostupnost podpory.

Tyto kroky jsou návrh navazující realizace; tento metareport netvrdí, že již proběhly.

## 11. Kontrola splnění zadání

| Požadavek prezentace | Výstup metareportu | Stav důkazu |
|---|---|---|
| Model / skóre rizika neúčasti | XGBoost/CatBoost, opravený směr skóre a explicitní historické segmentové skóre, oddíl 4. | Uložené průzkumné výsledky existují; kalibrace a nezávislá časová validace zbývají. |
| Hlavní související faktory | Syntéza asociací, SHAP, permutací a dalších vysvětlení, oddíl 5. | Doložené asociace a chování modelu, ne příčiny. |
| Praktické segmenty | Dvě rozhodovací osy, bezpečnostní filtr, čtyři priority a úplný soupis, oddíl 6 a příloha A. | Doložené agregace; reprodukce přiřazení vyžaduje řádková data. |
| Konkrétní intervence | Podpora přes ordinaci, navigace, objednací krok, informační a bezpečně načasovaná komunikace, oddíl 7. | Testovatelné hypotézy; účinek dosud neověřen. |
| Experiment / A/B test | Dva piloty s kontrolou, randomizací žen, ITT, primární metrikou, velikostí a ekonomickým vyhodnocením, oddíly 8–9. | Konkrétní návrh, nikoli již naměřený efekt. |

**Závěr pro porotu:** Řešení nepřestává u předpovědi neúčasti. Překládá riziko do vysvětlitelné nabídky podpory, chrání ženy před nevhodně načasovaným zvaním a navrhuje experiment, který odliší skutečně získanou účast od pouhého kontaktování žen, jež by přišly i bez intervence.

---

## Příloha A — úplný číselný základ segmentů roku 2024

Převzato z [A3]; každá žena je podle podkladu právě v jedné skupině. Sloupce „90/180/365“ jsou **počty s účastí**, nikoli procenta. Z nich lze vypočítat míry účasti i historické skóre neúčasti pro libovolný ze tří horizontů. Výsledky do různých horizontů jsou kumulativní, nikoli součet samostatných událostí.

| Identifikátor segmentu | Počet žen | Do 90 dní | Do 180 dní | Do 365 dní |
|---|---:|---:|---:|---:|
| `repeated_zero__medical_prevention` | 78 213 | 2 431 | 4 253 | 7 186 |
| `repeated_zero__other_prevention` | 31 057 | 378 | 906 | 1 880 |
| `repeated_zero__care_no_prevention` | 20 861 | 178 | 436 | 890 |
| `repeated_zero__low_contact` | 16 314 | 67 | 164 | 410 |
| `unobserved_repeat__medical_prevention` | 24 033 | 3 381 | 5 752 | 8 698 |
| `unobserved_first__medical_prevention` | 32 478 | 8 216 | 13 776 | 19 608 |
| `single_zero__medical_prevention` | 14 890 | 1 332 | 2 233 | 3 506 |
| `recent_screen` | 7 600 | 116 | 166 | 182 |
| `unobserved_repeat__other_prevention` | 5 436 | 317 | 718 | 1 243 |
| `past_response__medical_prevention` | 6 592 | 989 | 1 701 | 2 567 |
| `single_zero__other_prevention` | 4 367 | 167 | 398 | 708 |
| `unobserved_first__other_prevention` | 2 816 | 233 | 479 | 802 |
| `unobserved_repeat__care_no_prevention` | 2 292 | 73 | 152 | 316 |
| `single_zero__care_no_prevention` | 2 093 | 64 | 140 | 263 |
| `single_zero__low_contact` | 1 461 | 18 | 42 | 74 |
| `unobserved_repeat__low_contact` | 1 502 | 25 | 58 | 115 |
| `past_response__other_prevention` | 1 449 | 107 | 247 | 427 |
| `unobserved_first__low_contact` | 1 077 | 15 | 44 | 72 |
| `unobserved_first__care_no_prevention` | 1 089 | 38 | 90 | 158 |
| `past_response__no_prevention` | 846 | 35 | 82 | 160 |
| **Celkem** | **256 466** | **18 180** | **31 837** | **49 265** |

**Kontrola:** 256 466 − 49 265 = **207 201** žen bez zaznamenaného screeningu do 365 dní. Čtyři segmenty `repeated_zero` mají celkem 146 445 žen a 10 366 ročních účastí, tedy 136 079 bez záznamu. Tato kontrola ověřuje aritmetiku dodaných agregací, nikoli úplnost zdrojových zdravotních záznamů ani nezávislé provedení segmentace.
