"""Render Czech group cards, trial proposals, plots and reproducibility metadata."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import BASE, code_hash, parser, read_json, write_json

HISTORY_TEXT = {
    "repeated_zero": "alespoň dvě dřívější pozvánky s uplynulým oknem 365 + 30 dní, v žádném z těchto známých oken není zaznamenána účast",
    "single_zero": "právě jedna dřívější pozvánka se známým výsledkem po uplynutí 365 + 30 dní, bez zaznamenané účasti",
    "past_response": "alespoň jedna zaznamenaná účast po dřívější pozvánce s uplynulým oknem 365 + 30 dní",
    "unobserved_first": "žádné dřívější známé okno 365 + 30 dní a zdrojový příznak první pozvánky",
    "unobserved_repeat": "žádné dřívější známé okno 365 + 30 dní, ale zdrojový příznak opakované pozvánky",
}
CARE_TEXT = {
    "low_contact": "nula ve všech čtyřech doménách prevence a nejvýše dva kontakty typu poskytovatel/den za 12 měsíců",
    "care_no_prevention": "nula ve všech čtyřech doménách prevence, ale více než dva kontakty typu poskytovatel/den za 12 měsíců",
    "other_prevention": "žádná prevence u praktika ani gynekologa za 24 měsíců, ale alespoň jedna stomatologická prevence za 24 měsíců nebo jiný screening za 36 měsíců",
    "medical_prevention": "alespoň jedna preventivní prohlídka u praktika nebo gynekologa za 24 měsíců",
    "no_prevention": "nula ve všech čtyřech doménách prevence; nízký a vyšší kontakt byly sloučeny kvůli malé četnosti",
}
PREFIX = {
    "repeated_zero": "Opakované", "single_zero": "Jednorázové",
    "unobserved_first": "První", "unobserved_repeat": "Nezařazené", "past_response": "Navracející",
}
SUFFIX = {
    "low_contact": "nízkokontaktní", "care_no_prevention": "nepreventivní",
    "other_prevention": "selektivní", "medical_prevention": "preventistky",
    "no_prevention": "nepreventivní",
}
CARE_INTERVENTION = {
    "low_contact": (
        "Nabídnout jednoduchý vstup do péče, ne další obecné varování. Nízký kontakt neznamená odmítání ani nefunkční telefon.",
        "srozumitelný dopis s bezplatnou linkou; při oprávnění telefon navigátorky, případně SMS s volbou zpětného zavolání",
        "krátké vysvětlení přínosů i omezení screeningu bez příznaků a jeden konkrétní krok: zavolat nebo požádat o společné objednání",
        "dosažitelnost, jazyk, zdravotní gramotnost, preference kanálu, důvody absence kontaktu, potřeba doprovodu",
        "pořádání péče s navigátorkou přes nabídku zpětného volání; v nedigitální větvi stejně dostupná bezplatná linka v dopise",
    ),
    "care_no_prevention": (
        "Využít existující kontakt s péčí. Z počtu kontaktů nelze poznat diagnózu, zátěž ani konkrétní dostupnou ordinaci.",
        "podnět pro registrovaného praktika nebo gynekologa, osobní rozhovor při kontaktu, podpůrný dopis; SMS či portál jen podle oprávnění",
        "nabídka krátké konzultace a pomoci s výběrem termínu místo předpokladu, že další samostatný dopis stačí",
        "skutečná navštívená ordinace, termíny kontaktů, klinická vhodnost, konkurenční zdravotní potřeby, souhlas ordinace",
        "doporučení a nabídka objednání prostřednictvím vlastní ordinace; není-li dostupná, nabídka stejné konzultace navigátorkou",
    ),
    "other_prevention": (
        "Navázat na doloženou účast v jiné prevenci; ta však neprokazuje důvěru v mamografii ani znalost jejího průběhu.",
        "personalizovaný dopis s jednoduchým objednáním; volitelný e-mail, SMS nebo zabezpečený portál po ověření kontaktu a preference",
        "vysvětlení, v čem se mamografie liší od jiné prevence, co vyšetření obnáší a jak získat odpověď na obavy",
        "který preventivní program žena využila, zkušenost s mamografií, obavy, pracovní časy, preferované pracoviště",
        "dopis s krátkým rozhodovacím průvodcem a přímým kontaktem na objednání namísto obecné výzvy",
    ),
    "medical_prevention": (
        "Využít preventivní návaznost praktika či gynekologa. Nedovozovat, že žena odmítá screening nebo nepotřebuje podporu.",
        "doporučení konkrétní registrované ordinace; SMS s odkazem či telefonem, e-mail nebo push v aplikaci pouze při doložené dostupnosti",
        "jednoznačný objednací krok a možnost vyřešit otázky s lékařem, nikoli opakování obecných zdravotních hrozeb",
        "kontakt na ordinaci, digitální preference, důvod prodlení, již rezervovaný termín, čekací doba, přesná poslední mamografie",
        "pozvánka podpořená doporučením ordinace s jedním objednacím krokem a volbou telefonního nebo online objednání",
    ),
    "no_prevention": (
        "Obnovit přístup k prevenci po dřívější doložené účasti; spojená skupina zahrnuje různé úrovně kontaktu se zdravotnictvím.",
        "dopis a nabídka konzultace u praktika; telefon navigátorky při oprávnění, SMS nebo portál pouze po ověření preference",
        "neutrální připomenutí dalšího vhodného screeningu s možností probrat dosavadní zkušenost a pomoc s termínem",
        "důvod přerušení prevence, změny registrace a kontaktu, zkušenost s minulým vyšetřením, aktuální klinická vhodnost",
        "nabídka osobního navigačního hovoru a společného objednání s offline možností požádat o zavolání",
    ),
}
HISTORY_INTERVENTION = {
    "repeated_zero": (
        "Opakované nulové záznamy odůvodňují vyzkoušet jinou podporu než stejný dopis; neprokazují nedoručení ani vědomé odmítnutí.",
        "Nabídnout podporu již při T0 a nejvýše jeden dohodnutý navazující kontakt kolem 30. dne.",
    ),
    "single_zero": (
        "Jedno dřívější nulové okno není dlouhodobé odmítání. Nejprve ověřit praktickou překážku a porozumění.",
        "Při T0 nabídnout jednoduchý objednací krok; kolem 30. dne jeden navazující kontakt podle předem daného postupu.",
    ),
    "unobserved_first": (
        "Jde o první pozvánku podle zdrojového příznaku, nikoli důkaz první celoživotní zkušenosti se screeningem.",
        "Při T0 poskytnout stručnou orientaci; po 30 dnech nabídnout pomoc pouze podle předem stanovené politiky ramene B.",
    ),
    "unobserved_repeat": (
        "Historie je nedostatečně pozorovaná nebo ještě neuplynula. Neoznačovat tuto ženu za opakovaně nereagující.",
        "Při T0 ověřit, zda již není objednaná, a nabídnout podporu; jeden další kontakt kolem 30. dne.",
    ),
    "past_response": (
        "Dřívější účast ukazuje možnost návratu, ne jistotu budoucí účasti. Pozdní a rychlá historická reakce zůstávají popisnými příznaky.",
        "Při T0 navázat na dřívější zkušenost bez sdělování citlivé historie v nezabezpečené zprávě; jeden podpůrný kontakt kolem 30. dne.",
    ),
}


def percent(value) -> str:
    return f"{100 * value:.1f} %".replace(".", ",")


def integer(value) -> str:
    return f"{int(value):,}".replace(",", " ")


def number(value) -> str:
    return "neuvedeno" if pd.isna(value) else f"{value:g}".replace(".", ",")


def group_name(group_id: str) -> str:
    if group_id == "recent_screen":
        return "Nedávný záznam"
    history, care = group_id.split("__")
    return f"{PREFIX[history]} {SUFFIX[care]}"


def group_card(row, chars, standardized, annual, power, population):
    gid = row["group_id"]
    title = group_name(gid)
    lines = [
        f"## {title}",
        f"- **Identifikátor:** `{gid}`.",
        f"- **Základ:** {integer(row['n'])} žen / pozvánek v roce {population['index_year']} "
        f"({percent(row['share'])} roční kohorty); každá žena právě jednou.",
    ]
    if gid == "recent_screen":
        lines += [
            "- **Pravidlo:** údaj o posledním screeningu je kratší než 24 měsíců; přednost před všemi ostatními pravidly. "
            "Jde o bezpečnostní filtr k ověření, nikoli automatickou klinickou výjimku nebo diagnózu.",
        ]
    else:
        history, care = gid.split("__")
        lines.append(f"- **Pravidlo:** {HISTORY_TEXT[history]}; současně {CARE_TEXT[care]}; "
                     "není uveden screening v posledních 24 měsících.")
    lines += [
        f"- **Zaznamenaná účast:** do 90 dní {integer(row['attended_90d'])}/{integer(row['known_90d'])} "
        f"= {percent(row['rate_90d'])}; do 180 dní {integer(row['attended_180d'])}/{integer(row['known_180d'])} "
        f"= {percent(row['rate_180d'])}; do 365 dní {integer(row['attended_365d'])}/{integer(row['known_365d'])} "
        f"= {percent(row['rate_365d'])}.",
        f"- **Nejistota a prodlení:** popisný 95% Wilsonův interval účasti do 180 dní "
        f"{percent(row['ci_low_180d'])}–{percent(row['ci_high_180d'])}, do 365 dní "
        f"{percent(row['ci_low_365d'])}–{percent(row['ci_high_365d'])}; "
        f"v intervalu 91–180 dní je {integer(row['recorded_90_180_interval'])} "
        f"a v intervalu 181–365 dní {integer(row['recorded_180_365_interval'])} záznamů. "
        "Neověřená úplnost sledování není těmito intervaly pokryta.",
        f"- **Rozsah nezaznamenané účasti:** do 365 dní bez záznamu {integer(row['no_recorded_screen_365d'])} "
        f"žen ({percent(1-row['rate_365d'])}); {percent(row['share_of_nonattendance'])} všech ročních nulových záznamů. "
        "Nejde o odhad preventabilních případů.",
        f"- **Charakteristiky:** medián kontaktů poskytovatel/den za 12 měsíců "
        f"{number(chars['healthcare_contact_12m_cnt_median'])}; mediány prevence "
        f"praktik/gynekologie/zuby za 24 měsíců a jiný screening za 36 měsíců "
        f"{number(chars['gp_prevention_24m_cnt_median'])}/"
        f"{number(chars['gyn_prevention_24m_cnt_median'])}/"
        f"{number(chars['dental_prevention_24m_cnt_median'])}/"
        f"{number(chars['other_screening_36m_cnt_median'])}. "
        f"Bez screeningu ve zdrojovém pětiletém zpětném okně {percent(chars['no_screen_lookback_fraction'])}; "
        f"bez registrace u praktika {percent(chars['no_gp_registration_fraction'])}, "
        f"u gynekologa {percent(chars['no_gyn_registration_fraction'])}.",
        f"- **Další vlastnosti:** nejčastější věkové pásmo {chars['modal_age_band']} "
        f"({percent(chars['modal_age_fraction'])}); 65+ {percent(chars['age_65_plus_fraction'])}; "
        f"první pozvánka podle zdroje {percent(chars['source_first_fraction'])}; "
        f"medián dřívějších pozvánek pozorovaných v extraktu {number(chars['observed_prior_median'])}.",
    ]
    past = annual.loc[annual["group_id"].eq(gid) & annual["n"].ge(500)]
    if len(past):
        trend = "; ".join(f"{int(r.year)}: {percent(r.rate_180d)} (n={integer(r.n)})"
                          for r in past.itertuples())
        lines.append(f"- **Časový kontext:** zaznamenaná účast do 180 dní v podporovaných ročních buňkách: {trend}. "
                     "Ženy se mezi roky opakují; změna není důkazem efektu komunikace.")
    lines.append(
        f"- **Věkový kontext:** popisně standardizovaná účast do 180 dní "
        f"{percent(standardized['age_standardized_180d'])} proti {percent(standardized['same_strata_reference_180d'])} "
        f"ve stejných podporovaných věkových vrstvách; ty pokrývají "
        f"{percent(standardized['reference_population_coverage'])} referenční kohorty. Nejde o kauzální očištění."
    )
    if gid == "recent_screen":
        lines += [
            "- **Interpretace a priorita:** nízká další účast sama neznamená selhání. U této skupiny je medián recency "
            f"{number(chars['months_since_last_screen_median'])} měsíců. Časové zaokrouhlení v měsíci T0, souběh procesů "
            "nebo zpětné doplnění mohou narušit interpretaci; nejprve ověřit přesné datum výkonu, indikaci a datovou dostupnost.",
            "- **Řešení a komunikace:** po klinickém ověření aktualizovat termín další vhodné kontroly, nabídnout "
            "zabezpečený portál/aplikaci, e-mail či SMS podle oprávnění a preferencí, vždy i papírovou nebo telefonní alternativu. "
            "Nepobízet k neindikovanému opakování vyšetření.",
            "- **A/B test:** pouze po ověření vhodnosti a nejbližšího skutečně splatného screeningu randomizovat ženu 1:1. "
            "A = běžná správně načasovaná pozvánka; B = stejná klinická péče plus potvrzený osobní plán a připomenutí "
            "před doporučeným termínem. Primární výsledek je účast do 90 dní po individuálním doporučeném termínu; "
            "sekundárně neindikované duplicity, porozumění a odmítnutí. Obě ramena musí dodržet klinický interval.",
            "- **Proveditelnost a potřebná data:** přesné datum a typ výkonu, klinické doporučení dalšího termínu, dostupnost "
            "kontaktu a preference. Historická 180denní míra zde není vhodný baseline pro tento test; velikost testu "
            "nelze z dodaných proměnných poctivě spočítat.",
        ]
        return "\n".join(lines)
    history, care = gid.split("__")
    mechanism, channels, message, missing, variant = CARE_INTERVENTION[care]
    rationale, timing = HISTORY_INTERVENTION[history]
    priority = ("nižší zaznamenaná účast do 180 dní než v celé roční kohortě"
                if row["difference_180d_pp"] < 0 else "vyšší zaznamenaná účast do 180 dní než v celé roční kohortě")
    lines += [
        f"- **Interpretace a priorita:** {priority}; rozdíl "
        f"{row['difference_180d_pp']:+.1f} procentního bodu. {rationale} {mechanism}",
        f"- **Navržené řešení:** {message}. {timing} Před každou zprávou zkontrolovat aktuální vhodnost, "
        "již známý výkon a sjednaný termín; neodesílat další výzvy po vyřešení situace nebo odhlášení.",
        f"- **Komunikační možnosti:** {channels}. Dostupnost těchto kanálů data nedokládají. "
        "Citlivé zdravotní podrobnosti nepatří do nezabezpečených notifikací; offline možnost musí zůstat rovnocenná.",
        f"- **A/B test:** po ověření způsobilosti náhodně přiřadit ženu 1:1 při T0. "
        f"A = stávající adresná pozvánka a běžná péče; B = stejná péče plus {variant}. "
        f"Konkrétní variantu kanálu a záložní cestu určit podle předem zjištěného oprávnění a preference, nikoli po výsledku. "
        "Test hodnotí celý balíček, ne izolovaný účinek každé jeho části.",
        "- **Měření testu:** primárně doložená účast do 180 dní od T0 v intention-to-treat populaci; "
        "sekundárně 90/365 dní, objednání, čas do výkonu pouze po doplnění termínů, cena na dodatečnou účast, "
        "nedoručení, odhlášení, stížnosti, nerovnosti mezi kanály a neindikované duplicity. "
        "Jedna žena zůstává ve stejném rameni i při další pozvánce; segment se pro analýzu zmrazí při randomizaci.",
        f"- **Co doplnit:** {missing}; dále úplnost výsledků a skutečnou doručitelnost. "
        "Hypotézu o bariéře ověřit krátkým dobrovolným dotazem; z této analýzy nelze vyvozovat motivaci.",
    ]
    scenario = power.loc[power["group_id"].eq(gid) & power["scenario_absolute_increase"].eq(.05)
                         & power["alpha_two_sided"].lt(.05)].iloc[0]
    size = int(scenario["per_arm"])
    fits = "vejde" if scenario["fits_entire_historical_group"] else "nevejde"
    lines.append(
        f"- **Orientační velikost A/B:** při hypotetickém navýšení o 5 procentních bodů, síle 80 % a "
        f"oboustranném α={scenario['alpha_two_sided']:.5f} vychází přibližně {integer(size)} žen na rameno "
        f"({integer(2*size)} celkem); to se {fits} do celé historické velikosti skupiny. "
        "Jde o scénář, ne očekávaný efekt ani ověřený nábor; po klinických a kontaktních filtrech počet klesne. "
        "Pro skupinové změny v ordinaci je nutné přepočítat design effect."
    )
    return "\n".join(lines)


def plots(results, groups):
    table = groups.sort_values("rate_180d").copy()
    labels = [group_name(g) for g in table["group_id"]]
    fig, axis = plt.subplots(figsize=(11, 10))
    axis.barh(labels, 100 * table["rate_180d"], color="#287c8e")
    axis.errorbar(100 * table["rate_180d"], range(len(table)),
                  xerr=[100 * (table["rate_180d"] - table["ci_low_180d"]),
                        100 * (table["ci_high_180d"] - table["rate_180d"])],
                  fmt="none", ecolor="black", capsize=2)
    axis.set_xlabel("Zaznamenaná účast do 180 dní (%) — úplnost sledování neověřena")
    axis.set_title("Skupiny v pevné roční kohortě 2024; popisné 95% intervaly")
    fig.tight_layout()
    fig.savefig(results / "groups_180d.png", dpi=150)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(11, 10))
    axis.barh(labels, table["n"], color="#735a98")
    axis.set_xlabel("Ženy v roční kohortě 2024")
    axis.set_title("Velikost skupin; nejde o počet žen ovlivnitelných intervencí")
    fig.tight_layout()
    fig.savefig(results / "group_sizes.png", dpi=150)
    plt.close(fig)


def render(results, documentation):
    started = time.monotonic()
    audit = read_json(results / "audit.json")
    metadata = read_json(results / "groups.json")
    groups = pd.read_csv(results / "groups.csv")
    chars = pd.read_csv(results / "group_characteristics.csv").set_index("group_id")
    standardized = pd.read_csv(results / "age_standardization.csv").set_index("group_id")
    annual = pd.read_csv(results / "annual_groups.csv")
    power = pd.read_csv(results / "trial_power_scenarios.csv")
    order = groups.sort_values(["no_recorded_screen_365d"], ascending=False)
    cards = [group_card(row, chars.loc[row["group_id"]], standardized.loc[row["group_id"]],
                        annual, power, metadata) for row in order.to_dict("records")]
    interpretation = documentation / "interpretation"
    interpretation.mkdir(parents=True, exist_ok=True)
    (interpretation / "groups.md").write_text("\n\n".join(cards) + "\n", encoding="utf-8")
    plots(results, groups)
    manifest = {
        "data_sha256": audit["data_sha256"], "final_code_sha256": code_hash(),
        "source_invitations": audit["invitations"], "source_people": audit["people"],
        "index_year": metadata["index_year"], "main_people": metadata["people"],
        "groups": len(groups), "total_group_support": int(groups["n"].sum()),
        "all_people_assigned_once": metadata["all_people_assigned_once"],
        "outcome_cutoff_verified": False, "models_fitted": 0,
        "python": platform.python_version(),
        "utc_completed": datetime.now(timezone.utc).isoformat(),
        "report_elapsed_seconds": round(time.monotonic() - started, 2),
        "aggregate_files": sorted(
            {p.name for p in results.iterdir() if p.is_file() and p.name not in
             {"run_manifest.json", "validation.json"}} | {"run_manifest.json"}),
    }
    write_json(results / "run_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    cli = parser(__doc__)
    cli.add_argument("--documentation", type=Path, default=BASE)
    args = cli.parse_args()
    render(args.results, args.documentation)
