# Souvislosti, omezení a společný experimentální protokol

## Co analýza říká

- V hlavní roční kohortě 2024 je 256 466 pozvánek a stejně tolik žen.
  Zaznamenaná účast je 18 180 / 256 466 = **7,09 % do 90 dní**,
  31 837 / 256 466 = **12,41 % do 180 dní** a
  49 265 / 256 466 = **19,21 % do 365 dní**.
  Jde o dodanou filtrovanou kohortu, ne všechny adresně zvané ženy VZP.
- Čtyři profily bez známé předchozí reakce ve dvou či více uplynulých
  ročních oknech, po oddělení nedávných screeningů, zahrnují **146 445 žen**.
  Bez záznamu účasti do 365 dní jich je **136 079**,
  tedy **65,67 % všech 207 201 ročních nulových záznamů**.
- Největší jednotlivý profil má **78 213 žen**: opakovaná nulová historie,
  ale zaznamenaná prevence u praktika nebo gynekologa. Zaznamenaná roční
  účast činí **9,19 %**, bez účasti je 71 027 žen, tedy **34,28 %**
  všech ročních nulových záznamů. Smysluplná první hypotéza je návaznost
  ordinace a odstranění objednacích překážek, nikoli další obecná informační kampaň.
- Nejnižší účast mezi intervenčními profily mají opakovaně nulová historie
  a nízký kontakt s péčí: **16 314 žen**, účast do 180 dní **1,01 %**,
  do 365 dní **2,51 %**. Nízká výchozí účast ale není důkazem vysoké
  ovlivnitelnosti; dražší navigaci musí obhájit experiment a náklady.
- **7 600 žen** v roce 2024 má údaj o screeningu kratší než 24 měsíců.
  Nejprve ověřit časování a klinickou potřebu. Samotná další neúčast není
  důvodem k intenzivnější výzvě k dalšímu vyšetření.
- Skupiny s první pozvánkou a jinou prevencí se liší od opakovaně
  oslovovaných. Ani „preventistka“ však neznamená automatickou účast:
  první pozvánka a prevence u lékaře mají roční zaznamenanou účast **60,37 %**.
  Jde spíše o prostor pro včasnější a jednodušší objednání než automatickou
  prioritu drahé navigace.
- Nepotvrdili jsme konkrétní regionální bariéru, digitální nedostupnost,
  strach ani nedůvěru. Regionální a věkové řezy jsou k dispozici, ale tyto
  důvody bez dalších informací z administrativních dat určit nelze.

## Proč je skupin právě tolik

Počet nebyl vstupním parametrem. Neproběhlo učení, optimalizace stromu ani
shlukování. Nejprve byly spočteny celé křížové tabulky dostupných domén.
Následovalo průhledné členění podle odlišné nabídky podpory:

1. Nedávný screening má bezpečnostní přednost.
2. Historie rozlišuje dvě a více nulových oken, jedno nulové okno,
   dřívější pozitivní reakci a nepozorovanou historii.
3. U nepozorované historie se odděluje první/opakovaná pozvánka podle
   zdrojového příznaku.
4. Prevence rozlišuje nízký kontakt bez prevence, kontakt bez prevence,
   jinou prevenci bez prevence u praktika/gynekologa a prevenci u lékaře.
5. Rychlá a opožděná historická reakce zůstávají samostatné popisné
   charakteristiky, ale pro stejnou návratovou nabídku jsou sdruženy.
6. Malé sousední buňky bez prevence se spojují pouze v rámci stejné historie.
   Konkrétně 253 nízkokontaktních žen s dřívější reakcí a 593 s vyšším
   kontaktem vytvořilo profil o 846 ženách. Nespojují se různé historie
   jen proto, aby se dosáhlo zvoleného počtu.

Provozní hranice 500 žen dává 20 profilů. Hranice 250 dává 21 profilů;
1000 stále 20, ale jeden nedosahuje požadované podpory; 2000 dává 17,
z toho dva pod hranicí. Samotné zvýšení minima tedy neřeší všechny malé
profily a nevytváří důkaz optimální segmentace.

Jde o **exploračně vytvořenou provozní taxonomii**, ne objev dvaceti
vrozených, neměnných nebo nezávisle validovaných skupin. Názvy jsou krátké
orientační štítky; „opakované“ odkazuje na pozorovaná nulová okna,
„jednorázové“ na jediné známé nulové okno a „nezařazené“ na chybějící
uplynulou historii navzdory opakované pozvánce. Nejsou hodnotícím označením žen.

## Jednotka analýzy a výběrové zkreslení

Jeden zdrojový řádek je pozvánka, nikoli žena. Celkem je 1 344 556 pozvánek
pro 508 527 žen. Hlavní kohorta je pevný rok 2024; při více pozvánkách
v roce by se použila poslední z daného roku. Ve skutečných datech je
v každém roce nejvýše jedna pozvánka na ženu.

Výběr poslední pozvánky napříč všemi roky není vhodný jako hlavní baseline:
účast může souviset s tím, že další pozvánka již nepřijde. Tento výběr
závisí na budoucím procesu. V takto vybraných 508 527 řádcích vychází
230 752 pozitivních ročních výsledků, tedy **45,38 %**, proti **19,21 %**
v pevné kohortě 2024. Tento rozdíl nelze vydávat za zlepšení účasti.
První pozvánka pozorovaná v extraktu napříč roky má analogicky jiný výběr
a zaznamenanou roční účast 161 294 / 508 527 = **31,72 %**.

Všechny roky jsou zpracované v ročních tabulkách. Stejná žena v nich může
být vícekrát, takže se nesmějí sčítat jako nezávislé ženy. Individuální
přiřazení je v lokální databázi uchováno pro všechny pozvánky všech žen
i zvlášť pro hlavní roční kohortu, nikoli exportováno do textu.
Agregace všech pozvánek neuvádí naivní intervaly pro nezávislé ženy a
počty žen z různých skupin se nesčítají: žena může mezi pozvánkami změnit stav.
Na novou pozvánku lze aplikovat stejná pravidla,
vždy s tehdy dostupnou historií a aktuálně ověřenou vhodností.

Prezentace zadání uvádí pro rok 2024 292 176 pozvánek a 18,9% účast.
V dodaných datech je 256 466 pozvánek a 19,21% zaznamenaná roční účast.
Rozhraní popisuje filtr trvalého pojištění a anonymizační filtr; ani jejich
existence neopravňuje automaticky tvrdit, že přesně vysvětlují celý rozdíl.
Jmenovatele a definice je potřeba sjednotit se správcem.

## Kvalita a význam proměnných

- Validovány všechny řádky: formát a integrita identifikátorů, data, binární
  kódování, nezáporné celé počty, věková pásma, návaznost screeningové
  historie a vnoření výsledků 90 ≤ 180 ≤ 365 dní.
- Jedinečnost pozvánky a dvojice žena–datum se kontroluje i napříč bloky.
  Žádné tiché slučování, doplňování účasti nebo opravy labelů.
- Zdrojový počet a pořadí pozvánky se liší u **161 629 řádků**. Tyto
  čítače a zdrojové měsíce od pozvánky proto nejsou segmentační vstupy.
  Vlastní počet historie je počet skutečně pozorovaných dřívějších řádků.
- **58 řádků** má příznak první pozvánky a zároveň zdrojový počet vyšší než
  jedna. Příznak je zachován jako samostatný zdrojový údaj, nikoli
  prokázaná celoživotní první pozvánka.
- U **3 354 řádků** je nulový pětiletý počet, ale existuje starší
  screening na hranici nebo za 60 měsíci. To není samo o sobě rozpor.
- `NEVER_SCREENED_BEFORE_FLAG` se neinterpretuje jako „nikdy v životě“,
  protože dokumentace vymezuje pětileté pozorování.
- Kontakty znamenají součet unikátních poskytovatelů za jednotlivé dny.
  Nejde o počet návštěv, diagnózu ani počet unikátních lékařů za rok.
- Prevence praktika, gynekologa a zubů má okno 24 měsíců, jiný screening
  36 měsíců. Nejsou sečteny do falešného celkového počtu výkonů; používají
  se indikátory a počet zastoupených domén.
- Platná registrace není prevence, dostupný termín ani doložená návštěva.
  Současná prevence bez registrace může mít časově legitimní vysvětlení.
- Územní pracoviště VZP není administrativní okres, dojezdová vzdálenost
  ani socioekonomický ukazatel.
- Pojištění bylo použito v kvalitativním auditu, nikoli jako domnělá
  bariéra přístupu. Kohorta je již vybraná podle dlouhodobého pojištění.
- Chybějící proměnné se nenahrazují nulou. Pro neznámý profil existuje
  samostatný stav. Ve skutečných vstupních počtech a výsledcích chybění
  nebrání této segmentaci; chybějící recency má často strukturální význam.

## Časová bezpečnost a nejistota

Aktuální výsledky slouží až k popisu skupiny, ne k jejímu přiřazení.
Historie reakce vědomě používá starší výsledky, ale jen když konec
365denního okna plus 30 dní leží **striktně před T0** nové pozvánky.
Stejnou podmínku musí splnit i dřívější pozitivní příznak. Třicet dní je
pracovní bezpečnostní předpoklad, ne ověřená doba vykázání.

V celém extraktu je **556 719** sousedních rozestupů pozvánek nejvýše
365 dní a **823 353** nejvýše 395 dní. Bez oddělení aktuálního, ještě
neuplynulého a historického okna by šlo snadno použít budoucí informaci.
Překryv oken zároveň znamená, že jeden screening může vyvolat více
pozitivních příznaků. Nejde o prokázaný účinek konkrétního dopisu.

Na 279 300 prvních pozorovaných řádcích je zdrojový příznak opakované
pozvánky. Extrakt tedy neobsahuje kompletní celoživotní historii.

Konec sledování výsledků není dodán. Vyplněné A/N samo nestačí k potvrzení
uplynutí a úplnosti sledování. Nebyl vymyšlen konec 2025/2026 ani nebyla
nepozorovaná účast převáděna na neúčast. Tabulky a karty proto popisují
**zaznamenané příznaky**, ne garantované skutečné chování.

Stejně tak není doloženo, zda prediktory skutečně zachycují pouze informace
dostupné při T0, nebo obsahují zpětné doplnění. Zvlášť důležitá je recency
0 měsíců: v sekundární kohortě posledních pozvánek napříč roky má screening
kratší než 24 měsíců 33 350 žen, z nichž 21 860 má 0 a 10 795 má 1 měsíc.
Bez přesných dat nelze bezpečně vyloučit souběh v měsíci T0 nebo únik
budoucí informace. Tato skupina je k ověření, ne k automatickému vyřazení
z klinicky vhodné péče.

Wilsonovy intervaly jsou popisné binomické intervaly pro jednu pozvánku na
ženu v hlavní kohortě. Kohorta není náhodný vzorek všech žen; intervaly
neřeší výběrové zkreslení, chyby zdroje, nedosledování ani časový posun.
Roční intervaly nejsou společný interval trendu opakovaných žen.
Nebyla prováděna série potvrzujících testů s p-hodnotami.

Věková standardizace používá společné dostatečně obsazené vrstvy 40–49,
50–59, 60–69, 70+. U každé skupiny je uvedeno pokrytí referenční kohorty.
Při rozdílném pokrytí nelze bez dalšího srovnávat všechny standardizované
míry navzájem. Nejde o model ani důkaz příčiny.

## Citlivost a stabilita

- Posun hranice nízkého kontaktu z 2 na 0 mění přiřazení **2,03 %** žen,
  z 2 na 5 **2,40 %**. To neznamená, že každá malá skupina je stabilní.
- Změna historické vykazovací rezervy ze 30 na 90 dní při stejné hranici
  kontaktů v této kohortě přiřazení nemění. Bez rezervy, tedy při 0 dnech,
  se mění **14,14 %** přiřazení. Kalendář téměř ročních pozvánek zde hraje roli.
- Čtyři opakovaně nulové profily mají podobně nízkou účast v letech
  2023 a 2024. Dřívější roky nemají dost uplynulé historie pro stejnou
  klasifikaci; nelze tvrdit stabilitu těchto profilů ve všech pěti letech.
- Rozdíly věku, registrace a území jsou doplňující deskriptivní vrstvy,
  ne automatický důvod rozdrobit každý profil. Přesné termíny, kapacity
  a bariéry mohou později podpořit další smysluplné členění.
- Pravidla nebyla potvrzena na nezávislé kohortě. Budoucí validace má
  předem zmrazit pravidla a měřit podporu, úplnost, reprodukci vztahů
  i skutečný dopad intervence.

## Společný návrh A/B experimentů

Každá skupina má konkrétní kontrast v interpretačním souboru. Společné
podmínky:

1. **Před náborem:** ověřit způsobilost, přesný poslední výkon, předchozí
   rezervaci, aktuální doporučení, oprávnění ke kontaktu a dostupnost termínů.
   Věk v pětiletém pásmu, zejména 40–44, nestačí k určení přesné způsobilosti.
   Není přípustné automaticky doporučovat neindikované opakování.
2. **Randomizace:** žena 1:1 v rámci segmentu; blokování podle věku,
   území a předem zjištěné dostupnosti kanálu. Přidělení skryté do zařazení,
   reprodukovatelný postup a jediná experimentální účast ženy.
3. **Kontrola:** stávající standardní komunikace a běžná klinická péče.
   Nesmí se zadržovat doporučená péče. Rozdíl je nabídka podpory.
4. **Lékařské intervence:** pokud změna workflow zasáhne celou ordinaci,
   randomizovat ordinace a přepočítat sílu i analýzu na klastry. Individuální
   tabulka počtů v takovém designu nestačí. Základní design effect je
   `1 + (průměrná velikost klastru − 1) × ICC`; ICC zde není známo.
5. **Primární výsledek:** pro intervenční profily zaznamenaná účast do
   180 dní od T0, po zajištění stejného plného sledování obou ramen.
   Bezpečnostní profil má jiný, klinicky splatný začátek hodnocení
   a nemá zde použitelný baseline pro velikost experimentu.
6. **Analýza:** intention-to-treat všech randomizovaných žen, absolutní
   rozdíl účasti s 95% intervalem, relativní efekt jako doplněk, předem
   stanovené úpravy pro randomizační vrstvy a klastry. Samotné kliknutí,
   doručení či objednání nejsou náhradou účasti.
7. **Opakované pozvánky:** stejné rameno, zmrazený vstupní segment,
   žádné nové přiřazování podle postintervenčního chování. Interval
   90 dní může sloužit jako sekundární výsledek, nikoli filtr pro
   dodatečný výběr „nereagujících“ do hlavní analýzy.
8. **Sledování:** předem určit cutoff, vykazovací rezervu, účast mimo VZP,
   úmrtí, přechod jinam a chybějící výsledek. Neznámý výsledek není
   automaticky nula; plán citlivosti a důvod chybění musí být předem.
9. **Více testů:** netestovat všech 19 podpůrných balíčků bez dostatečného
   náboru a oprav. Doporučujeme pilot kapacity, potom předregistrovaný
   prioritní test. Scénáře ukazují α=0,05 i konzervativní Bonferroni
   `0,05/19`; pro skutečný počet současných hypotéz je nutný nový plán.
   Dvacátá bezpečnostní skupina má samostatnou hypotézu; pokud by byla ve
   stejné potvrzující rodině, započítat i ji.
10. **Síla:** dva rovnocenné nezávislé soubory, 80% síla, oboustranný
    test, normální aproximace. Nárůst o 2 nebo 5 procentních bodů je
    výslovně scénář, nikoli naměřený nebo slíbený účinek. Baseline je
    historický a neověřený; před náborem přepočítat na ověřeném pilotu,
    po filtrech způsobilosti, s reálným horizontem a kapacitou.
11. **Náklady:** sledovat cenu doručení, času ordinace, navigátorky,
    vyšetření a cenu za dodatečnou účast. Sloupce „1/5 procentních bodů“
    v tabulce skupin vyjadřují jen aritmetickou velikost hypotetického
    posunu, ne odhad přínosu či pořadí optimálních investic.
12. **Bezpečnost:** stručná neutrální komunikace bez strašení, respekt
    k informovanému rozhodnutí a odmítnutí, limit frekvence kontaktů,
    žádné citlivé detaily v nezabezpečené SMS/e-mailu. Online, aplikace,
    push či SMS nejsou předpokládaná vlastnost věkové skupiny.
    Zajistit čitelnou a nedigitální alternativu.

AB test balíčku nerozliší samostatný vliv kanálu, obsahu a navigace. Pokud
to kapacita dovolí, navazující test má měnit jednu komponentu nebo použít
předem navržený faktoriální design; nejde o součást současné analýzy.

## Co je potřeba získat dále

- Ověřený cutoff, vykazovací zpoždění a datum sestavení každého prediktoru.
- Přesná data výkonů a klinická pravidla vhodnosti dalšího screeningu.
- Skutečná historie odeslání, doručení a předchozích intervencí.
- Dostupnost a preference telefonu, SMS, e-mailu, aplikace a papírové pošty;
  právní a provozní oprávnění k danému způsobu oslovení.
- Nabídka termínů, doba čekání, vzdálenost a doprava, bezbariérovost,
  jazyk, potřeba doprovodu, dobrovolně uvedené překážky.
- Skutečná navštívená ordinace a možnost zapojení do experimentu.

Přidání těchto údajů může změnit počet i smysl skupin. Nejsou nahrazeny
odhadnutými příjmy, diagnózami nebo domnělou digitální gramotností.

## Zdroje a ochrana dat

Výchozím kontextem jsou místní prezentace „VZP – Od pozvánky k mamografu“
a datové rozhraní z 29. 9. 2026. Starší implementační prompt je kontext,
nikoli příkaz v této iteraci trénovat modely nebo dodat šest skupin.
Jeho požadavky na ML a jednosouborovou aplikaci nepřebíráme; hlavní
modelovací aplikace zůstala nezměněná.

Použity standardní deskriptivní postupy: audit schématu, kontrola
jmenovatelů, časová dostupnost, křížové tabulky, mediány, Wilsonovy
intervaly, citlivost a prospektivní randomizované vyhodnocení. Nebyl
proveden systematický rešeršní přehled ani přenesen údaj o efektu cizí
intervence na tyto ženy; všechny komunikační efekty jsou hypotézy.

Vše proběhlo lokálně. Agregace s podporou řádku pod 20 nejsou v příslušných
CSV publikovány; počty vynechaných řádků jsou samostatně evidovány.
Nejde o úplnou anonymizační záruku: nejsou řešeny všechny odvoditelné
komplementy a malé čitatele. Výstupy vyžadují kontrolu před sdílením.
Identifikátory a individuální přiřazení zůstávají pouze v ignorované
lokální databázi. Nebyla odeslána žádná komunikace ženám.
