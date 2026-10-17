# Rakathon 2026 – automatický analytický report

> Jde o průzkumnou analýzu asociací. Výsledky nejsou důkazem kauzality a samy o sobě neprokazují, že změna faktoru změní účast.

## 1. Velikost a časové pokrytí

- Řádky po vyřazení neplatných dat pozvánky: **1 344 556**
- Unikátní klientky podle dostupného ID: **508527**
- Unikátní ID pozvánek: **1344556**
- Rozsah dat pozvánek: **2020-01-14 00:00:00 – 2024-12-06 00:00:00**
- Klientky s více řádky: **317557**
- Podíl řádků patřících klientkám s více pozvánkami: **85.80%**

## 2. Základní prevalence cílových labelů

| target             |   horizon_days |   rows_labeled |   participants |   nonparticipants |   participation_rate_pct |
|:-------------------|---------------:|---------------:|---------------:|------------------:|-------------------------:|
| SCREENING_3M_FLAG  |             90 |        1344556 |          95769 |           1248787 |                    7.123 |
| SCREENING_6M_FLAG  |            180 |        1344556 |         168621 |           1175935 |                   12.541 |
| SCREENING_12M_FLAG |            365 |        1344556 |         259536 |           1085020 |                   19.303 |

Interpretace: srovnávejte labely s ohledem na délku časového okna. Účast do 12 měsíců přirozeně může být vyšší než do 3 měsíců.

## 3. Nejvýraznější kategoriální asociace

### SCREENING_3M_FLAG
| feature                          |   cramers_v |   range_between_group_rates_pp |   min_group_rate_pct |   max_group_rate_pct |
|:---------------------------------|------------:|-------------------------------:|---------------------:|---------------------:|
| FIRST_INVITATION                 |      0.2481 |                        16.9704 |               4.2295 |              21.1999 |
| CARE_AND_SCREENING_PROFILE       |      0.2314 |                        18.5507 |               0.3682 |              18.9189 |
| PREVENTION_BREADTH_BAND          |      0.2172 |                        13.094  |               1.1667 |              14.2607 |
| PREVENTION_TOTAL_BAND            |      0.2158 |                        16.2925 |               1.1667 |              17.4593 |
| AGE_BAND                         |      0.1897 |                        19.7837 |               3.7298 |              23.5135 |
| PRIOR_INVITATION_COUNT_BAND      |      0.172  |                        10.5063 |               2.0415 |              12.5478 |
| DAYS_SINCE_PRIOR_INVITATION_BAND |      0.1671 |                         8.9369 |               3.6109 |              12.5478 |
| HEALTHCARE_CONTACT_BAND          |      0.1113 |                         8.6116 |               0.709  |               9.3206 |
| GYN_REG_DURATION_BAND            |      0.1086 |                        10.3248 |               2.0902 |              12.415  |
| REGISTRATION_PROFILE             |      0.1029 |                         7.5692 |               0.9567 |               8.526  |
| GYN_REGISTERED_FLAG              |      0.1023 |                         6.4091 |               2.0902 |               8.4993 |
| LAST_SCREEN_RECENCY_BAND         |      0.099  |                        18.5464 |               0      |              18.5464 |

### SCREENING_6M_FLAG
| feature                          |   cramers_v |   range_between_group_rates_pp |   min_group_rate_pct |   max_group_rate_pct |
|:---------------------------------|------------:|-------------------------------:|---------------------:|---------------------:|
| FIRST_INVITATION                 |      0.3161 |                        27.8382 |               7.795  |              35.6332 |
| CARE_AND_SCREENING_PROFILE       |      0.2851 |                        28.889  |               0.9382 |              29.8272 |
| PREVENTION_BREADTH_BAND          |      0.2682 |                        21.1133 |               2.7101 |              23.8234 |
| PREVENTION_TOTAL_BAND            |      0.2606 |                        25.1295 |               2.7101 |              27.8396 |
| AGE_BAND                         |      0.2474 |                        33.9933 |               6.4891 |              40.4823 |
| PRIOR_INVITATION_COUNT_BAND      |      0.2239 |                        17.6437 |               3.963  |              21.6067 |
| DAYS_SINCE_PRIOR_INVITATION_BAND |      0.2173 |                        14.9443 |               6.6624 |              21.6067 |
| GYN_REG_DURATION_BAND            |      0.1406 |                        16.3638 |               4.0345 |              20.3983 |
| HEALTHCARE_CONTACT_BAND          |      0.1391 |                        14.2348 |               1.7763 |              16.0111 |
| LAST_SCREEN_RECENCY_BAND         |      0.136  |                        31.5507 |               0      |              31.5507 |
| REGISTRATION_PROFILE             |      0.1351 |                        13.035  |               1.877  |              14.912  |
| GYN_REGISTERED_FLAG              |      0.1343 |                        10.8333 |               4.0345 |              14.8678 |

### SCREENING_12M_FLAG
| feature                          |   cramers_v |   range_between_group_rates_pp |   min_group_rate_pct |   max_group_rate_pct |
|:---------------------------------|------------:|-------------------------------:|---------------------:|---------------------:|
| FIRST_INVITATION                 |      0.3598 |                        37.7649 |              12.8644 |              50.6293 |
| CARE_AND_SCREENING_PROFILE       |      0.3307 |                        43.6374 |               1.8171 |              45.4545 |
| PREVENTION_BREADTH_BAND          |      0.3086 |                        29.3362 |               5.3149 |              34.6511 |
| PREVENTION_TOTAL_BAND            |      0.2964 |                        33.9266 |               5.3149 |              39.2415 |
| AGE_BAND                         |      0.2873 |                        46.8139 |              10.0086 |              56.8226 |
| PRIOR_INVITATION_COUNT_BAND      |      0.2593 |                        24.5772 |               7.1407 |              31.7179 |
| DAYS_SINCE_PRIOR_INVITATION_BAND |      0.2505 |                        21.2839 |              11.2147 |              32.4986 |
| LAST_SCREEN_RECENCY_BAND         |      0.175  |                        54.7938 |               2.349  |              57.1429 |
| GYN_REG_DURATION_BAND            |      0.171  |                        22.8674 |               6.9052 |              29.7726 |
| REGISTRATION_PROFILE             |      0.1652 |                        19.3129 |               3.4442 |              22.7571 |
| GYN_REGISTERED_FLAG              |      0.1643 |                        15.7886 |               6.9052 |              22.6938 |
| HEALTHCARE_CONTACT_BAND          |      0.1634 |                        20.3891 |               3.6964 |              24.0855 |

## 4. Numerické atributy s nejsilnější monotónní asociací

### SCREENING_3M_FLAG
| feature                          |   spearman_rho_with_participation |   mean_among_participants |   mean_among_nonparticipants |
|:---------------------------------|----------------------------------:|--------------------------:|-----------------------------:|
| GYN_PREVENTION_24M_CNT           |                            0.2434 |                    1.2483 |                       0.4977 |
| ANY_GYN_PREVENTION               |                            0.2319 |                    0.7722 |                       0.3367 |
| PREVENTION_DOMAINS_WITH_ACTIVITY |                            0.2179 |                    2.8996 |                       1.7878 |
| PREVENTION_TOTAL_CNT             |                            0.2147 |                    5.3278 |                       3.0562 |
| OTHER_SCREENING_36M_CNT          |                            0.1845 |                    2.2222 |                       1.239  |
| ANY_OTHER_SCREENING              |                            0.1826 |                    0.8488 |                       0.4972 |
| HIST_PRIOR_INVITATION_COUNT      |                           -0.1678 |                    0.5622 |                       1.3824 |
| HIST_PRIOR_INVITATION_DATES      |                           -0.1678 |                    0.5622 |                       1.3824 |
| DENTAL_PREVENTION_24M_CNT        |                            0.1224 |                    1.3124 |                       0.8815 |
| ANY_DENTAL_PREVENTION            |                            0.1187 |                    0.7639 |                       0.5386 |
| HEALTHCARE_CONTACT_COUNT_LOG1P   |                            0.1047 |                    2.7946 |                       2.3564 |
| HEALTHCARE_CONTACT_12M_CNT       |                            0.1047 |                   20.1071 |                      15.6281 |

### SCREENING_6M_FLAG
| feature                          |   spearman_rho_with_participation |   mean_among_participants |   mean_among_nonparticipants |
|:---------------------------------|----------------------------------:|--------------------------:|-----------------------------:|
| GYN_PREVENTION_24M_CNT           |                            0.2925 |                    1.1601 |                       0.4638 |
| ANY_GYN_PREVENTION               |                            0.2814 |                    0.7279 |                       0.3161 |
| PREVENTION_DOMAINS_WITH_ACTIVITY |                            0.2698 |                    2.8012 |                       1.7331 |
| PREVENTION_TOTAL_CNT             |                            0.2631 |                    5.0786 |                       2.9512 |
| ANY_OTHER_SCREENING              |                            0.2281 |                    0.8206 |                       0.4794 |
| OTHER_SCREENING_36M_CNT          |                            0.2244 |                    2.1025 |                       1.1953 |
| HIST_PRIOR_INVITATION_COUNT      |                           -0.2179 |                    0.5954 |                       1.4285 |
| HIST_PRIOR_INVITATION_DATES      |                           -0.2179 |                    0.5954 |                       1.4285 |
| DENTAL_PREVENTION_24M_CNT        |                            0.1577 |                    1.2887 |                       0.8582 |
| ANY_DENTAL_PREVENTION            |                            0.1541 |                    0.7544 |                       0.526  |
| HEALTHCARE_CONTACT_COUNT_LOG1P   |                            0.1323 |                    2.7574 |                       2.3346 |
| HEALTHCARE_CONTACT_12M_CNT       |                            0.1323 |                   19.6096 |                      15.4219 |

### SCREENING_12M_FLAG
| feature                          |   spearman_rho_with_participation |   mean_among_participants |   mean_among_nonparticipants |
|:---------------------------------|----------------------------------:|--------------------------:|-----------------------------:|
| GYN_PREVENTION_24M_CNT           |                            0.3308 |                    1.0832 |                       0.4239 |
| ANY_GYN_PREVENTION               |                            0.3206 |                    0.6854 |                       0.2918 |
| PREVENTION_DOMAINS_WITH_ACTIVITY |                            0.3098 |                    2.7009 |                       1.6676 |
| PREVENTION_TOTAL_CNT             |                            0.2999 |                    4.852  |                       2.8271 |
| ANY_OTHER_SCREENING              |                            0.2644 |                    0.7914 |                       0.4578 |
| OTHER_SCREENING_36M_CNT          |                            0.2552 |                    2.0025 |                       1.1432 |
| HIST_PRIOR_INVITATION_COUNT      |                           -0.2537 |                    0.6635 |                       1.482  |
| HIST_PRIOR_INVITATION_DATES      |                           -0.2537 |                    0.6635 |                       1.482  |
| DENTAL_PREVENTION_24M_CNT        |                            0.1834 |                    1.2523 |                       0.8308 |
| ANY_DENTAL_PREVENTION            |                            0.1818 |                    0.7375 |                       0.5109 |
| HEALTHCARE_CONTACT_COUNT_LOG1P   |                            0.1519 |                    2.7233 |                       2.3074 |
| HEALTHCARE_CONTACT_12M_CNT       |                            0.1519 |                   19.233  |                      15.1611 |

## 5. Víceproměnná logistická regrese – nejsilnější faktory

Koeficienty jsou odhadnuty na reprodukovatelném vzorku označených řádků. Číselné proměnné jsou škálované; u kategorií jde o kontrast proti referenční úrovni. Jde o asociace po zohlednění ostatních featur v modelu, nikoli o kauzální účinky.

### SCREENING_3M_FLAG
| source_feature                                    |   max_abs_coefficient |   rms_coefficient |   number_of_encoded_terms |
|:--------------------------------------------------|----------------------:|------------------:|--------------------------:|
| ANY_GYN_PREVENTION                                |                0.2147 |            0.2147 |                         1 |
| MONTHS_SINCE_LAST_SCREEN                          |                0.1783 |            0.1783 |                         1 |
| GYN_PREVENTION_24M_CNT                            |                0.1744 |            0.1744 |                         1 |
| FIRST_INVITATION                                  |                0.1645 |            0.1645 |                         2 |
| PREVENTION_DOMAINS_WITH_ACTIVITY                  |                0.1332 |            0.1332 |                         1 |
| ANY_OTHER_SCREENING                               |                0.1069 |            0.1069 |                         1 |
| HIST_PRIOR_INVITATION_DATES                       |                0.106  |            0.106  |                         1 |
| HIST_PRIOR_INVITATION_COUNT                       |                0.106  |            0.106  |                         1 |
| HEALTHCARE_CONTACT_COUNT_LOG1P                    |                0.1021 |            0.1021 |                         1 |
| AGE_BAND                                          |                0.1003 |            0.0507 |                         7 |
| PREVENTION_TOTAL_CNT                              |                0.0979 |            0.0979 |                         1 |
| missingindicator_HIST_DAYS_SINCE_FIRST_INVITATION |                0.0904 |            0.0904 |                         1 |
| missingindicator_HIST_DAYS_SINCE_PRIOR_INVITATION |                0.0904 |            0.0904 |                         1 |
| HIST_DAYS_SINCE_PRIOR_INVITATION                  |                0.0662 |            0.0662 |                         1 |
| DENTAL_PREVENTION_24M_CNT                         |                0.0647 |            0.0647 |                         1 |

### SCREENING_6M_FLAG
| source_feature                                    |   max_abs_coefficient |   rms_coefficient |   number_of_encoded_terms |
|:--------------------------------------------------|----------------------:|------------------:|--------------------------:|
| MONTHS_SINCE_LAST_SCREEN                          |                0.2619 |            0.2619 |                         1 |
| FIRST_INVITATION                                  |                0.2295 |            0.2294 |                         2 |
| ANY_GYN_PREVENTION                                |                0.1868 |            0.1868 |                         1 |
| GYN_PREVENTION_24M_CNT                            |                0.1563 |            0.1563 |                         1 |
| AGE_BAND                                          |                0.1426 |            0.0756 |                         7 |
| PREVENTION_DOMAINS_WITH_ACTIVITY                  |                0.1298 |            0.1298 |                         1 |
| ANY_OTHER_SCREENING                               |                0.1153 |            0.1153 |                         1 |
| HIST_PRIOR_INVITATION_COUNT                       |                0.1114 |            0.1114 |                         1 |
| HIST_PRIOR_INVITATION_DATES                       |                0.1114 |            0.1114 |                         1 |
| HEALTHCARE_CONTACT_COUNT_LOG1P                    |                0.1101 |            0.1101 |                         1 |
| missingindicator_HIST_DAYS_SINCE_FIRST_INVITATION |                0.0804 |            0.0804 |                         1 |
| missingindicator_HIST_DAYS_SINCE_PRIOR_INVITATION |                0.0804 |            0.0804 |                         1 |
| HIST_DAYS_SINCE_PRIOR_INVITATION                  |                0.0775 |            0.0775 |                         1 |
| PREVENTION_TOTAL_CNT                              |                0.0709 |            0.0709 |                         1 |
| ANY_DENTAL_PREVENTION                             |                0.0645 |            0.0645 |                         1 |

### SCREENING_12M_FLAG
| source_feature                                    |   max_abs_coefficient |   rms_coefficient |   number_of_encoded_terms |
|:--------------------------------------------------|----------------------:|------------------:|--------------------------:|
| MONTHS_SINCE_LAST_SCREEN                          |                0.3326 |            0.3326 |                         1 |
| FIRST_INVITATION                                  |                0.2811 |            0.2809 |                         2 |
| AGE_BAND                                          |                0.2147 |            0.1049 |                         7 |
| ANY_GYN_PREVENTION                                |                0.1778 |            0.1778 |                         1 |
| GYN_PREVENTION_24M_CNT                            |                0.1534 |            0.1534 |                         1 |
| HEALTHCARE_CONTACT_COUNT_LOG1P                    |                0.1221 |            0.1221 |                         1 |
| PREVENTION_DOMAINS_WITH_ACTIVITY                  |                0.1203 |            0.1203 |                         1 |
| HIST_PRIOR_INVITATION_COUNT                       |                0.1033 |            0.1033 |                         1 |
| HIST_PRIOR_INVITATION_DATES                       |                0.1033 |            0.1033 |                         1 |
| ANY_OTHER_SCREENING                               |                0.1024 |            0.1024 |                         1 |
| REGISTRATION_PROFILE                              |                0.0834 |            0.0564 |                         4 |
| GYN_REGISTERED_FLAG                               |                0.0806 |            0.0804 |                         2 |
| HIST_DAYS_SINCE_PRIOR_INVITATION                  |                0.078  |            0.078  |                         1 |
| missingindicator_HIST_DAYS_SINCE_FIRST_INVITATION |                0.0751 |            0.0751 |                         1 |
| missingindicator_HIST_DAYS_SINCE_PRIOR_INVITATION |                0.0751 |            0.0751 |                         1 |

## 6. Nelineární model (Random Forest) – důležitosti faktorů

Random Forest může zachytit nelinearity a interakce, které logistická regrese s lineárními členy nezachytí. Zde uváděné impurity importance jsou orientační a mohou zvýhodňovat atributy s více kategoriemi; ověřujte je proti univariačním tabulkám, logistické regresi a stabilitě v čase.

### SCREENING_3M_FLAG
| source_feature             |   total_impurity_importance |   max_encoded_importance |   number_of_encoded_terms |
|:---------------------------|----------------------------:|-------------------------:|--------------------------:|
| VZP_TERRITORIAL_AREA       |                     0.15024 |                  0.00774 |                        75 |
| HEALTHCARE_CONTACT_12M_CNT |                     0.0668  |                  0.0668  |                         1 |
| INSURANCE_DURATION_YEARS   |                     0.06426 |                  0.06426 |                         1 |
| PRIOR_SCREEN_COUNT_5Y      |                     0.0616  |                  0.0616  |                         1 |
| MONTHS_SINCE_LAST_SCREEN   |                     0.06157 |                  0.06157 |                         1 |
| AGE_BAND                   |                     0.05879 |                  0.00891 |                         7 |
| GP_PREVENTION_24M_CNT      |                     0.05742 |                  0.05742 |                         1 |
| GYN_PREVENTION_24M_CNT     |                     0.04931 |                  0.04931 |                         1 |
| OTHER_SCREENING_36M_CNT    |                     0.04363 |                  0.04363 |                         1 |
| DENTAL_PREVENTION_24M_CNT  |                     0.03998 |                  0.03998 |                         1 |
| INVITATION_YEAR            |                     0.03184 |                  0.03184 |                         1 |
| INVITATION_MONTH           |                     0.02515 |                  0.02515 |                         1 |
| INVITATION_QUARTER         |                     0.02126 |                  0.02126 |                         1 |
| INVITATION_DAY_OF_WEEK     |                     0.01964 |                  0.01964 |                         1 |
| INVITATION_MONTH_SIN       |                     0.0184  |                  0.0184  |                         1 |

### SCREENING_6M_FLAG
| source_feature             |   total_impurity_importance |   max_encoded_importance |   number_of_encoded_terms |
|:---------------------------|----------------------------:|-------------------------:|--------------------------:|
| VZP_TERRITORIAL_AREA       |                     0.13717 |                  0.00707 |                        75 |
| HEALTHCARE_CONTACT_12M_CNT |                     0.08459 |                  0.08459 |                         1 |
| INSURANCE_DURATION_YEARS   |                     0.08423 |                  0.08423 |                         1 |
| PRIOR_SCREEN_COUNT_5Y      |                     0.07185 |                  0.07185 |                         1 |
| AGE_BAND                   |                     0.06138 |                  0.00945 |                         7 |
| MONTHS_SINCE_LAST_SCREEN   |                     0.05719 |                  0.05719 |                         1 |
| GP_PREVENTION_24M_CNT      |                     0.04543 |                  0.04543 |                         1 |
| GYN_PREVENTION_24M_CNT     |                     0.0445  |                  0.0445  |                         1 |
| OTHER_SCREENING_36M_CNT    |                     0.03534 |                  0.03534 |                         1 |
| DENTAL_PREVENTION_24M_CNT  |                     0.02863 |                  0.02863 |                         1 |
| INVITATION_YEAR            |                     0.02769 |                  0.02769 |                         1 |
| INVITATION_MONTH           |                     0.02392 |                  0.02392 |                         1 |
| INVITATION_QUARTER         |                     0.02217 |                  0.02217 |                         1 |
| INVITATION_DAY_OF_WEEK     |                     0.02161 |                  0.02161 |                         1 |
| INVITATION_MONTH_SIN       |                     0.02064 |                  0.02064 |                         1 |

### SCREENING_12M_FLAG
| source_feature             |   total_impurity_importance |   max_encoded_importance |   number_of_encoded_terms |
|:---------------------------|----------------------------:|-------------------------:|--------------------------:|
| VZP_TERRITORIAL_AREA       |                     0.13143 |                  0.00731 |                        75 |
| HEALTHCARE_CONTACT_12M_CNT |                     0.08399 |                  0.08399 |                         1 |
| INSURANCE_DURATION_YEARS   |                     0.08271 |                  0.08271 |                         1 |
| AGE_BAND                   |                     0.06283 |                  0.01022 |                         7 |
| PRIOR_SCREEN_COUNT_5Y      |                     0.0586  |                  0.0586  |                         1 |
| MONTHS_SINCE_LAST_SCREEN   |                     0.04697 |                  0.04697 |                         1 |
| GP_PREVENTION_24M_CNT      |                     0.04365 |                  0.04365 |                         1 |
| GYN_PREVENTION_24M_CNT     |                     0.04224 |                  0.04224 |                         1 |
| OTHER_SCREENING_36M_CNT    |                     0.03542 |                  0.03542 |                         1 |
| DENTAL_PREVENTION_24M_CNT  |                     0.02644 |                  0.02644 |                         1 |
| INVITATION_YEAR            |                     0.0261  |                  0.0261  |                         1 |
| INVITATION_MONTH           |                     0.02553 |                  0.02553 |                         1 |
| INVITATION_QUARTER         |                     0.0255  |                  0.0255  |                         1 |
| INVITATION_DAY_OF_WEEK     |                     0.02438 |                  0.02438 |                         1 |
| INVITATION_MONTH_SIN       |                     0.02321 |                  0.02321 |                         1 |

## 7. Interpretovatelné segmenty

Soubor `rule_segments__*.csv` obsahuje kombinace věkové skupiny, historie screeningu v posledních 5 letech, celkového prevenčního chování a registrace u praktika/gynekologa. Drobné skupiny jsou filtrovány; pořadí zohledňuje velikost segmentu i odchylku účasti od průměru.

## 8. Segmentace pomocí clusterů

|   cluster |   n_rows |   share_rows_pct | top_screening_history_group   | top_prevention_total_band   | top_registration_profile   |   SCREENING_12M_FLAG_participation_rate_pct |
|----------:|---------:|-----------------:|:------------------------------|:----------------------------|:---------------------------|--------------------------------------------:|
|         0 |   242293 |           18.02  | 0 in last 5y                  | 0                           | GP+GYN registered          |                                       6.09  |
|         1 |   269162 |           20.019 | 0 in last 5y                  | 4-6                         | GP+GYN registered          |                                      47.963 |
|         2 |   273584 |           20.348 | 0 in last 5y                  | 4-6                         | GP+GYN registered          |                                      20.926 |
|         3 |   239373 |           17.803 | 0 in last 5y                  | 2-3                         | GP+GYN registered          |                                      13.45  |
|         4 |   320144 |           23.81  | 0 in last 5y                  | 0                           | GP+GYN registered          |                                       8.195 |

Clustering používá pouze prediktorové featury a nesleduje labely při tvorbě skupin. Labely se připojí až pro profilaci clusterů. Klastry jsou statistické segmenty, nikoli automaticky klinicky smysluplné typy klientek.

## 9. Nově vytvořené featury

- Cyklické časové proměnné měsíce / dne v týdnu a roční období.
- Předchozí pozvánky a počet předchozích dat pozvánek, odstup od poslední a od první pozvánky; řádky stejného dne se navzájem nepočítají do historie.
- Prevenční součet, počet oblastí prevence s aktivitou a explicitní počet chybějících prevenčních atributů.
- Binární indikátory prevence, skupiny intenzity předchozího screeningu, recence screeningu, kontaktů se zdravotnictvím.
- Kombinovaný profil registrace u praktika/gynekologa a čitelný behaviorální profil pro segmentaci.

## 10. Důležité metodické limity

1. Opakované pozvánky jedné klientky znamenají, že řádky nejsou nezávislé. Zde se vztahy zkoumají na úrovni pozvánky; pro intervaly spolehlivosti a testy hypotéz by bylo třeba zohlednit shlukování podle osoby.
2. U velmi velkého N mohou být p-hodnoty miniaturní i při slabém vztahu. Sledujte velikost efektu (Cramérovo V, rozdíly v procentních bodech, lift), stabilitu v čase a velikost skupiny.
3. Missing neznamená automaticky nulu. Součty prevence používají nulu pouze jako pomocný výpočet, zároveň je vytvořena featura `PREVENTION_DATA_MISSING_COUNT`.
4. `MONTHS_SINCE_LAST_SCREEN` a `NEVER_SCREENED_BEFORE_FLAG` mohou popisovat pouze vymezené pětileté pozorovací okno; neinterpretujte je bez ověření definice jako celoživotní historii.
5. Pozvánky v novějších obdobích musí mít dokončené celé 90/180/365denní okno. Neúplné labely mohou vychýlit sazby; ověřte datum úplnosti vykázaných výkonů.
6. Některé featury vznikají přímo z původních indikátorů, proto jejich jednotlivé efekty nejsou nezávislé. Logistická regrese je regularizovaná, ale multikolinearita interpretaci stále ovlivňuje.

## 11. Kde hledat podrobné výstupy

- `missingness.csv`, `numeric_summary.csv`: kvalita dat a distribuce.
- `categorical_rates__*.csv`, `numeric_bin_rates__*.csv`: konkrétní skupiny / intervaly s mírou účasti, počty a odchylkou od průměru.
- `categorical_associations__*.csv`, `numeric_associations__*.csv`: souhrn síly univariačních asociací.
- `logistic_terms__*.csv`, `logistic_factor_ranking__*.csv`: víceproměnné koeficienty a pořadí vstupních faktorů.
- `forest_encoded_importance__*.csv`, `forest_factor_ranking__*.csv`: nelineární stromové důležitosti (orientační).
- `rule_segments__*.csv`: srozumitelné kombinace faktorů.
- `cluster_profile.csv`, `cluster_assignments.csv`: popis clusterů a přiřazení každého řádku.
- `monthly_participation_trend.csv` a `plot_*.png`: časový trend a grafy.
