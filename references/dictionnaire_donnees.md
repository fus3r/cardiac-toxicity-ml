# Dictionnaire des données — `RT_Thorax_v1.csv`

> **Projet** : Pôle Projet P15 — Développement de cartes d'activation des risques pour prédire les toxicités cardiaques après radiothérapie thoracique  
> **Cohorte** : Sous-ensemble de la **FCCSS** (*French Childhood Cancer Survivors Study*)  
> **Date** : Mars 2026

---

## Vue d'ensemble du fichier

| Propriété | Valeur |
|---|---|
| Nombre de lignes (patients) | **1 375** |
| Nombre de colonnes | **23** |
| Séparateur | Virgule (`,`) |
| Valeurs manquantes | **Aucune** (toutes les colonnes ont 1 375 valeurs non vides) |
| Population | Survivants de cancers pédiatriques traités par **radiothérapie thoracique**, issus de la cohorte FCCSS |
| Période de diagnostic | **1950 – 2000** |

### Évolutions par rapport à la version 0

| Changement | Détail |
|---|---|
| **Colonnes supprimées** | `radiotherapie` et `radiotherapie_1K` (variables non informatives, constantes = 1 pour tous) |
| **Colonnes ajoutées** | `mean`, `V5`, `V10`, `V15`, `V20` (indicateurs dose-volume cardiaques) |
| **Patients exclus** | 3 patients sans matrice de dose cardiaque → **1 375 patients** (vs 1 378) |

### Contexte scientifique

Ce fichier est un extrait de la cohorte **FCCSS** (French Childhood Cancer Survivors Study), une cohorte multicentrique française de **7 670 survivants** de cancers de l'enfant diagnostiqués avant l'âge de 21 ans et traités entre 1945 et 2001 dans 5 centres anticancéreux français (Chounta *et al.* 2023 ; Bentriou *et al.* 2024). Parmi eux, environ **3 900 ont été traités par radiothérapie** et bénéficient d'une reconstruction dosimétrique voxélisée corps entier.

Le sous-ensemble `RT_Thorax_v1.csv` contient **1 375 patients** ayant reçu une **radiothérapie thoracique**. L'objectif clinique est de **prédire la survenue de pathologies cardiaques** (`Pathologie_cardiaque`) en fonction de caractéristiques cliniques, démographiques, de traitement et **dosimétriques**.

### Références bibliographiques (dossier `bib/`)

| Abréviation | Référence complète | DOI |
|---|---|---|
| **Chounta 2023** | Chounta S, Allodji R, *et al.* « Dosiomics-Based Prediction of Radiation-Induced Valvulopathy after Childhood Cancer ». *Cancers* 2023;15:3107 | `10.3390/cancers15123107` |
| **Bentriou 2024** | Bentriou M, Letort V, *et al.* « Combining dosiomics and machine learning methods for predicting severe cardiac diseases in childhood cancer survivors: the FCCSS ». *Front. Oncol.* 2024;14:1241221 | `10.3389/fonc.2024.1241221` |
| **Sarrade 2023** | Sarrade T, Allodji R, *et al.* « CANTO-RT: One of the Largest Prospective Multicenter Cohort of Early Breast Cancer Patients Treated with Radiotherapy ». *Cancers* 2023;15:751 | `10.3390/cancers15030751` |
| **Veres 2014** | Veres C, Allodji R, *et al.* « Retrospective Reconstructions of Active Bone Marrow Dose-Volume Histograms ». *Int J Radiat Oncol Biol Phys* 2014;90:1217–1224 | `10.1016/j.ijrobp.2014.08.021` |
| **Sujet projet** | « Développement de cartes d'activation des risques… » — Description du projet pôle P15 (2025/2026) | — |
| **Lancement S6** | Présentation de lancement — Pôle P15 — 6 février 2026 | — |

---

## Dictionnaire colonne par colonne

---

### 1. `ctr` — Centre de traitement

| Propriété | Détail |
|---|---|
| **Type** | Entier catégoriel |
| **Description** | Identifiant anonymisé du **centre anticancéreux** où le patient a été diagnostiqué et traité |
| **Valeurs observées** | `1` (n=1069, 77,7 %), `3` (n=213), `12` (n=55), `10` (n=27), `8` (n=11) |
| **Valeurs manquantes** | 0 |

**Explication clinique :**  
La cohorte FCCSS regroupe des patients de **5 centres français de cancérologie** (Bentriou 2024, Section 2.1 : *"a large multi-centric cohort of 7670 patients diagnosed with cancer between 1946 and 2000, among five centers"*). Les centres collaborateurs incluent Gustave Roussy (Villejuif), l'Institut Curie (Paris), le Centre Antoine-Lacassagne (Nice), l'Institut Claudius Regaud (Toulouse) et le CHU de Reims (Bentriou 2024, Acknowledgments). Les identifiants numériques sont anonymisés. Le centre `1` domine nettement (~78 % des patients), probablement Gustave Roussy, centre coordinateur de l'étude.

---

### 2. `numcent` — Numéro d'identification du patient

| Propriété | Détail |
|---|---|
| **Type** | Entier (identifiant unique) |
| **Description** | Numéro d'identification unique du patient au sein de son centre |
| **Valeurs uniques** | 1 375 (unique pour chaque patient) |
| **Valeurs manquantes** | 0 |

**Explication clinique :**  
Il s'agit du **numéro de dossier patient** pseudonymisé, propre à chaque centre. Combiné avec `ctr`, il forme une clé unique. Ce numéro est utilisé pour la jointure avec les données dosimétriques 3D (matrices de dose voxélisées).

>  **Ne jamais publier ni diffuser ce champ** : même pseudonymisé, il pourrait permettre une ré-identification combiné avec d'autres variables, conformément aux exigences de la CNIL (accords n° 902287 et n° 12038829 mentionnés dans Chounta 2023).

---

### 3. `deces` — Décès

| Propriété | Détail |
|---|---|
| **Type** | Binaire (0/1) |
| **Description** | Indicateur de **statut vital** à la date de dernière mise à jour du suivi |
| **Distribution** | `0` (vivant) : 848 (61,7 %) · `1` (décédé) : 527 (38,3 %) |
| **Valeurs manquantes** | 0 |

**Explication clinique :**  
Le statut vital est obtenu pour tous les patients via le **CépiDC** (Centre d'épidémiologie sur les causes médicales de décès) et le registre français des décès (Chounta 2023, Section 2.1). Les causes de décès sont codées selon la CIM-9 et CIM-10 (*International Classification of Diseases*). Le taux de mortalité de 38,3 % reflète le très long suivi de cette cohorte (médiane ~32 ans) et la gravité des cancers pédiatriques historiques.

**Rôle en analyse de survie :**  
Dans un modèle de survie (Cox PH, Random Survival Forest), `deces` participe à la définition de l'**événement de censure**. Si un patient est décédé mais pas d'une pathologie cardiaque, son temps de suivi est censuré à la date de décès pour l'événement « pathologie cardiaque ».

---

### 4. `chimiotherapie_1K` — Chimiothérapie au 1er cancer

| Propriété | Détail |
|---|---|
| **Type** | Binaire (0/1) |
| **Description** | Indique si le patient a reçu de la **chimiothérapie** lors du traitement du premier cancer |
| **Distribution** | `1` (oui) : 1 139 (82,8 %) · `0` (non) : 236 (17,2 %) |
| **Valeurs manquantes** | 0 |

**Explication clinique :**  
La chimiothérapie est un **facteur de risque connu** des pathologies cardiaques chez les survivants de cancers pédiatriques, indépendamment de la radiothérapie (Bentriou 2024 : *"chemotherapy (4) are known long-term risk factors for CDs"*). Dans les modèles de Bentriou 2024, deux variables binaires de chimiothérapie sont incluses comme ajustement : traitement par **anthracyclines** et par **agents alkylants**. La chimiothérapie est un **confondant potentiel** qu'il est essentiel d'ajuster dans tout modèle prédictif de cardiotoxicité radio-induite.

---

### 5. `Pathologie_cardiaque` — Événement cardiaque (VARIABLE CIBLE)

| Propriété | Détail |
|---|---|
| **Type** | Binaire (0/1) |
| **Description** | Indique si le patient a développé une **pathologie cardiaque sévère** (grade ≥ 3 CTCAE) |
| **Distribution** | `0` (non) : 1 139 (82,8 %) · `1` (oui) : 236 (17,2 %) |
| **Valeurs manquantes** | 0 |

**Explication clinique :**  
C'est la **variable cible** (*outcome*) du projet. Les pathologies cardiaques incluent :

- **Valvulopathies** (VHD — *Valvular Heart Disease*) étudiées spécifiquement dans Chounta 2023
- **Insuffisance cardiaque** (*heart failure*)
- **Cardiopathies ischémiques**
- **Péricardites**
- **Troubles du rythme**

Les événements sont identifiés, validés et gradés selon les **CTCAE v4.03** (*Common Terminology Criteria for Adverse Events*). Seuls les cas **sévères (grade ≥ 3)** sont retenus pour éviter le biais de déclaration des événements de faible grade (Chounta 2023 : *"We considered only severe VHD cases (grade ≥ 3), since there are concerns that non-severe cardiovascular disease is often self-declared and could cause a reporting bias"*).

**Prévalence** : 17,2 % dans ce sous-ensemble thoracique, nettement plus élevée que les ~5 % dans la cohorte FCCSS complète.

>  **C'est la variable à prédire** dans les modèles de machine learning et d'analyse de survie du projet.

---

### 6. `age_diag` — Âge au diagnostic

| Propriété | Détail |
|---|---|
| **Type** | Entier (années) |
| **Description** | Âge du patient au moment du **diagnostic du premier cancer** |
| **Plage** | 0 – 20 ans |
| **Moyenne** | 7,94 ans |
| **Valeurs manquantes** | 0 |

**Explication clinique :**  
L'âge au diagnostic est une **variable clinique fondamentale** dans les études sur les survivants de cancers pédiatriques. Les enfants diagnostiqués très jeunes (< 5 ans) ont des tissus en croissance plus vulnérables aux effets tardifs des radiations. Le critère d'inclusion de la FCCSS est un diagnostic **avant l'âge de 21 ans**.

---

### 7. `anthra_1K` — Anthracyclines au 1er cancer

| Propriété | Détail |
|---|---|
| **Type** | Binaire (0/1) |
| **Description** | Indique si le patient a été traité par **anthracyclines** lors du traitement du premier cancer |
| **Distribution** | `0` (non) : 856 (62,3 %) · `1` (oui) : 519 (37,7 %) |
| **Valeurs manquantes** | 0 |

**Explication clinique :**  
Les **anthracyclines** (doxorubicine, daunorubicine, épirubicine…) sont une classe majeure de chimiothérapie antitumorale connue pour leur **cardiotoxicité dose-dépendante**. Elles provoquent des lésions directes aux cardiomyocytes par génération de radicaux libres et stress oxydatif.

L'effet cardiotoxique des anthracyclines est **additif voire synergique** avec celui de la radiothérapie thoracique. C'est un **facteur de confusion majeur** : ne pas l'ajuster biaiserait l'estimation de l'effet propre de la dose de radiation.

---

### 8. `do_anthra_1K` — Dose cumulée d'anthracyclines au 1er cancer

| Propriété | Détail |
|---|---|
| **Type** | Numérique continu (mg/m²) |
| **Description** | **Dose cumulée** d'anthracyclines reçue lors du traitement du premier cancer, en mg/m² de surface corporelle |
| **Plage** | 0 – 797,31 mg/m² |
| **Moyenne** | 91,27 mg/m² |
| **Valeurs manquantes** | 0 |

**Explication clinique :**  
La cardiotoxicité des anthracyclines est **dose-dépendante**. Les seuils classiques de la littérature sont :

| Dose cumulée | Risque |
|---|---|
| < 250 mg/m² | Risque modéré |
| 250 – 400 mg/m² | Risque augmenté |
| > 400 mg/m² | Risque élevé d'insuffisance cardiaque |

>  Cette variable apporte une information **plus fine** que `anthra_1K` (binaire). Toutefois, les deux sont fortement corrélées puisque `do_anthra_1K = 0` ⟺ `anthra_1K = 0`.

---

### 9. `Year_date_diag` — Année de diagnostic

| Propriété | Détail |
|---|---|
| **Type** | Entier (année calendaire) |
| **Description** | **Année** du diagnostic du premier cancer |
| **Plage** | 1950 – 2000 |
| **Moyenne** | 1981,46 |
| **Valeurs manquantes** | 0 |

**Explication clinique :**  
L'année de diagnostic est un **proxy puissant** pour les pratiques thérapeutiques :

- **Avant 1970** : radiothérapie ancienne (cobalt-60, rayons X conventionnels), doses élevées, champs larges, peu de protection cardiaque
- **1970-1990** : amélioration progressive, introduction des accélérateurs linéaires
- **Après 1990** : techniques modernes (3D conformationnelle), réduction des doses aux organes sains

---

### 10. `Year_date_diag_cut` — Année de diagnostic catégorisée

| Propriété | Détail |
|---|---|
| **Type** | Catégoriel ordonné (chaîne de caractères) |
| **Description** | **Catégorie** de la période de diagnostic |
| **Valeurs** | |

| Valeur | Signification | Effectif |
|---|---|---|
| `1_< 1970` | Diagnostic **avant 1970** | 192 (14,0 %) |
| `2_1970-1980` | Diagnostic entre **1970 et 1980** | 432 (31,4 %) |
| `3_1970-1980` | Diagnostic entre **1980 et 1990** (probable erreur d'étiquetage : devrait être `3_1980-1990`) | 375 (27,3 %) |
| `4_1990-2000` | Diagnostic entre **1990 et 2000** | 376 (27,3 %) |

**Valeurs manquantes** | 0

>  **Attention** : la valeur `3_1970-1980` est probablement une **erreur d'étiquetage** dans le CSV. Compte tenu de l'ordinal `3_` et de la distribution des années réelles, il devrait s'agir de `3_1980-1990`. À vérifier avec les fournisseurs de données.

---

### 11. `age_diag_cut` — Âge au diagnostic catégorisé

| Propriété | Détail |
|---|---|
| **Type** | Catégoriel ordonné |
| **Description** | Catégorie d'**âge au diagnostic** |
| **Valeurs** | |

| Valeur | Signification | Effectif |
|---|---|---|
| `1_< 5` | 0 à 4 ans | 420 (30,5 %) |
| `2_5-10` | 5 à 9 ans | 391 (28,4 %) |
| `3_10-15` | 10 à 14 ans | 425 (30,9 %) |
| `4_15-20` | 15 à 20 ans | 139 (10,1 %) |

**Valeurs manquantes** | 0

---

### 12. `CT_sans_anthra` — Chimiothérapie sans anthracyclines

| Propriété | Détail |
|---|---|
| **Type** | Binaire (0/1) |
| **Description** | Indique si le patient a reçu une **chimiothérapie ne contenant pas d'anthracyclines** (par ex. agents alkylants, antimétabolites, etc.) |
| **Distribution** | `0` (non) : 770 (56,0 %) · `1` (oui) : 605 (44,0 %) |
| **Valeurs manquantes** | 0 |

**Explication clinique :**  
Cette variable capture le traitement par des **agents chimiothérapeutiques non-anthracycliniques**, notamment les **agents alkylants** (cyclophosphamide, ifosfamide, etc.), qui ont également des effets cardiotoxiques potentiels.

---

### 13. `iccc_type` — Type de cancer (classification ICCC)

| Propriété | Détail |
|---|---|
| **Type** | Catégoriel (chaîne de caractères) |
| **Description** | **Type histologique du premier cancer** selon la classification ICCC-3 (*International Classification of Childhood Cancer*, 3ème édition) |
| **Valeurs** | |

| Valeur | Cancer | Effectif |
|---|---|---|
| `02 -Lymphomas` | Lymphomes (Hodgkin et non-Hodgkin) | 429 (31,2 %) |
| `03 -CNS tumor` | Tumeurs du système nerveux central | 311 (22,6 %) |
| `06 -Renal tumors` | Tumeurs rénales (néphroblastome / tumeur de Wilms) | 160 (11,6 %) |
| `04 -Peripheral nervouus tumors` | Tumeurs du système nerveux périphérique (neuroblastome principalement) | 140 (10,2 %) |
| `Others` | Autres types regroupés (leucémies, rétinoblastomes, hépatocarcinomes, etc.) | 127 (9,2 %) |
| `08 -Bone sarcomas` | Sarcomes osseux (ostéosarcome, sarcome d'Ewing) | 113 (8,2 %) |
| `09 -Soft-tissue sarcomas` | Sarcomes des tissus mous (rhabdomyosarcome, etc.) | 95 (6,9 %) |

**Valeurs manquantes** | 0

>  Note : le libellé `Peripheral nervouus tumors` contient une faute de frappe (*nervouus* au lieu de *nervous*).

---

### 14. `time` — Temps de suivi (années)

| Propriété | Détail |
|---|---|
| **Type** | Numérique continu (années) |
| **Description** | **Durée de suivi** depuis le diagnostic du premier cancer jusqu'à l'événement (pathologie cardiaque) ou la **censure** (décès d'autre cause, perdu de vue, date de point) |
| **Plage** | 0,12 – 69,13 ans |
| **Moyenne** | 32,24 ans |
| **Valeurs manquantes** | 0 |

**Explication clinique :**  
C'est la **variable temps** de l'analyse de survie (*time-to-event*). Dans un modèle de Cox ou un Random Survival Forest, le couple (`time`, `Pathologie_cardiaque`) définit le statut de chaque patient.

---

### 15. `age_exit` — Âge à la sortie d'étude

| Propriété | Détail |
|---|---|
| **Type** | Numérique continu (années) |
| **Description** | **Âge du patient** au moment de l'événement cardiaque ou de la censure |
| **Plage** | 1,69 – 82,52 ans |
| **Moyenne** | 40,72 ans |
| **Valeurs manquantes** | 0 |

**Explication clinique :**  
Relation avec les autres variables : `age_exit ≈ age_diag + time`. Cette variable est utile pour caractériser le **profil d'âge** des patients au moment où ils développent (ou non) une pathologie cardiaque.

---

### 16. `time_cut` — Temps de suivi catégorisé

| Propriété | Détail |
|---|---|
| **Type** | Catégoriel ordonné |
| **Description** | Catégorie de **durée de suivi** |
| **Valeurs** | |

| Valeur | Signification | Effectif |
|---|---|---|
| `1_< 10` | Moins de 10 ans | 134 (9,7 %) |
| `2_10-20` | 10 à 20 ans | 125 (9,1 %) |
| `3_20-30` | 20 à 30 ans | 343 (24,9 %) |
| `4_30-40` | 30 à 40 ans | 343 (24,9 %) |
| `5_>40` | Plus de 40 ans | 430 (31,3 %) |

**Valeurs manquantes** | 0

---

### 17. `age_exit_cut` — Âge à la sortie catégorisé

| Propriété | Détail |
|---|---|
| **Type** | Catégoriel ordonné |
| **Description** | Catégorie d'**âge** au moment de l'événement ou de la censure |
| **Valeurs** | |

| Valeur | Signification | Effectif |
|---|---|---|
| `1_< 15` | Moins de 15 ans | 63 (4,6 %) |
| `2_15-25` | 15 à 25 ans | 161 (11,7 %) |
| `3_25-35` | 25 à 35 ans | 191 (13,9 %) |
| `4_35-45` | 35 à 45 ans | 415 (30,2 %) |
| `5_>45` | Plus de 45 ans | 545 (39,6 %) |

**Valeurs manquantes** | 0

---

### 18. `gender` — Sexe

| Propriété | Détail |
|---|---|
| **Type** | Catégoriel (chaîne de caractères) |
| **Description** | **Sexe biologique** du patient |
| **Distribution** | `Male` : 753 (54,8 %) · `Female` : 622 (45,2 %) |
| **Valeurs manquantes** | 0 |

**Explication clinique :**  
Le sexe est un **facteur d'ajustement systématique** dans toutes les études de la FCCSS. Il influence la distribution des types de cancer, la radiosensibilité cardiaque et l'espérance de vie. Le sexe est aussi utilisé dans la **reconstruction dosimétrique** pour le choix du fantôme anthropomorphe.

---

## Variables dosimétriques (NOUVELLES dans v1)

Les cinq colonnes suivantes sont des **indicateurs dose-volume (DVH)** extraits des matrices de dose cardiaque 3D voxélisées. Ces données ont été reconstruites pour chaque patient à partir des dossiers de radiothérapie historiques (Bentriou 2024 : *"whole-body voxelized dosimetric data were reconstructed"*).

---

### 19. `mean` — Dose moyenne au cœur

| Propriété | Détail |
|---|---|
| **Type** | Numérique continu (Gray, Gy) |
| **Description** | **Dose moyenne** de radiation reçue par l'ensemble du volume cardiaque |
| **Plage** | 0 – 61,20 Gy |
| **Moyenne** | 15,42 Gy |
| **Valeurs manquantes** | 0 |

**Explication clinique :**  
La dose moyenne au cœur (*mean heart dose*, MHD) est l'indicateur dosimétrique le plus couramment utilisé pour évaluer l'exposition cardiaque en radiothérapie. Elle représente la moyenne arithmétique des doses reçues par tous les voxels du volume cardiaque segmenté.

**Interprétation** :
- **< 5 Gy** : Risque faible
- **5-15 Gy** : Risque modéré
- **> 25 Gy** : Risque élevé de complications cardiaques tardives

La moyenne observée de 15,42 Gy reflète l'hétérogénéité des techniques de radiothérapie utilisées entre 1950 et 2000. Les patients traités avant 1970 présentent typiquement des doses plus élevées.

---

### 20. `V5` — Volume recevant ≥ 5 Gy

| Propriété | Détail |
|---|---|
| **Type** | Numérique continu (proportion, 0–1) |
| **Description** | **Fraction du volume cardiaque** recevant une dose ≥ 5 Gy |
| **Plage** | 0 – 1 |
| **Moyenne** | 0,67 (67 %) |
| **Valeurs manquantes** | 0 |

**Explication clinique :**  
V5 représente la proportion du cœur exposée à au moins 5 Gy. Une valeur de 0,67 signifie qu'en moyenne, 67 % du volume cardiaque des patients de la cohorte a reçu au moins 5 Gy.

Les métriques Vx (V5, V10, etc.) sont issues de l'**histogramme dose-volume (DVH)** et capturent la distribution spatiale de la dose, contrairement à la dose moyenne qui perd cette information.

---

### 21. `V10` — Volume recevant ≥ 10 Gy

| Propriété | Détail |
|---|---|
| **Type** | Numérique continu (proportion, 0–1) |
| **Description** | **Fraction du volume cardiaque** recevant une dose ≥ 10 Gy |
| **Plage** | 0 – 1 |
| **Moyenne** | 0,57 (57 %) |
| **Valeurs manquantes** | 0 |

**Explication clinique :**  
V10 est parfois utilisé comme seuil de cardiotoxicité faible dans la littérature. Dans cette cohorte, plus de la moitié du cœur a reçu en moyenne ≥ 10 Gy, témoignant de techniques d'irradiation historiques moins conformationnelles.

---

### 22. `V15` — Volume recevant ≥ 15 Gy

| Propriété | Détail |
|---|---|
| **Type** | Numérique continu (proportion, 0–1) |
| **Description** | **Fraction du volume cardiaque** recevant une dose ≥ 15 Gy |
| **Plage** | 0 – 1 |
| **Moyenne** | 0,49 (49 %) |
| **Valeurs manquantes** | 0 |

**Explication clinique :**  
V15 marque un seuil intermédiaire de dose. Environ la moitié du volume cardiaque a reçu au moins 15 Gy en moyenne dans cette cohorte.

---

### 23. `V20` — Volume recevant ≥ 20 Gy

| Propriété | Détail |
|---|---|
| **Type** | Numérique continu (proportion, 0–1) |
| **Description** | **Fraction du volume cardiaque** recevant une dose ≥ 20 Gy |
| **Plage** | 0 – 1 |
| **Moyenne** | 0,34 (34 %) |
| **Valeurs manquantes** | 0 |

**Explication clinique :**  
V20 est un indicateur de **dose élevée**. Une V20 > 30 % est généralement considérée comme un facteur de risque significatif de complications cardiaques. Dans cette cohorte, environ un tiers du cœur a reçu ≥ 20 Gy en moyenne.

**Contexte de reconstruction :**  
Ces indicateurs DVH ont été calculés à partir des **matrices de dose cardiaque 3D** reconstruites via simulation Monte Carlo ou algorithmes analytiques, avec une résolution de 2 mm (Bentriou 2024). L'algorithme a produit une anatomie ajustée correspondant au mieux à l'anatomie de chaque patient, en tenant compte du sexe, de l'âge et de la position adoptée pendant la radiothérapie.

---

## Résumé des types de variables

### Variables d'identification
| Colonne | Type | Usage |
|---|---|---|
| `ctr` | Catégoriel | Centre de traitement |
| `numcent` | Identifiant | Numéro patient (clé de jointure) |

### Variable cible
| Colonne | Type | Usage |
|---|---|---|
| `Pathologie_cardiaque` | Binaire | **Événement à prédire** |

### Variables de suivi / survie
| Colonne | Type | Usage |
|---|---|---|
| `deces` | Binaire | Statut vital (censure) |
| `time` | Continu | Temps jusqu'à événement/censure |
| `age_exit` | Continu | Âge à l'événement/censure |
| `time_cut` | Catégoriel | Temps catégorisé |
| `age_exit_cut` | Catégoriel | Âge sortie catégorisé |

### Variables cliniques (prédicteurs)
| Colonne | Type | Usage |
|---|---|---|
| `age_diag` | Continu | Âge au diagnostic |
| `age_diag_cut` | Catégoriel | Âge au diagnostic catégorisé |
| `gender` | Catégoriel | Sexe biologique |
| `iccc_type` | Catégoriel | Type de cancer (ICCC-3) |
| `Year_date_diag` | Continu | Année de diagnostic |
| `Year_date_diag_cut` | Catégoriel | Période de diagnostic |

### Variables de traitement chimiothérapeutique (prédicteurs)
| Colonne | Type | Usage |
|---|---|---|
| `chimiotherapie_1K` | Binaire | Chimio au 1er cancer |
| `anthra_1K` | Binaire | Anthracyclines au 1er cancer |
| `do_anthra_1K` | Continu | Dose cumulée d'anthracyclines (mg/m²) |
| `CT_sans_anthra` | Binaire | Chimio sans anthracyclines |

### Variables dosimétriques cardiaques (prédicteurs)
| Colonne | Type | Usage |
|---|---|---|
| `mean` | Continu | Dose moyenne au cœur (Gy) |
| `V5` | Continu | Fraction du cœur recevant ≥ 5 Gy |
| `V10` | Continu | Fraction du cœur recevant ≥ 10 Gy |
| `V15` | Continu | Fraction du cœur recevant ≥ 15 Gy |
| `V20` | Continu | Fraction du cœur recevant ≥ 20 Gy |

---

## Notes importantes pour la modélisation

### 1. Nouvelles variables dosimétriques
Les 5 indicateurs DVH (`mean`, `V5`, `V10`, `V15`, `V20`) sont les **principales variables prédictives** pour la cardiotoxicité radio-induite. Ils capturent l'exposition cardiaque aux radiations de manière plus précise que les variables catégorielles de traitement.

### 2. Colinéarité des variables dosimétriques
Les variables `mean`, `V5`, `V10`, `V15`, `V20` sont **fortement corrélées** entre elles. Il peut être nécessaire de :
- Sélectionner un sous-ensemble (ex. : `mean` seule, ou `V20` seule)
- Utiliser des techniques de réduction de dimension (ACP)
- Appliquer une régularisation (LASSO, Ridge) dans les modèles

### 3. Redondance entre variables continues et catégorielles
Les paires (`age_diag`, `age_diag_cut`), (`Year_date_diag`, `Year_date_diag_cut`), (`time`, `time_cut`), (`age_exit`, `age_exit_cut`) sont redondantes. Choisir l'une ou l'autre selon le modèle.

### 4. Colinéarité `anthra_1K` ↔ `do_anthra_1K`
`anthra_1K = 0` ⟺ `do_anthra_1K = 0`. Ne pas inclure les deux simultanément dans un modèle linéaire.

### 5. Erreur d'étiquetage probable
La valeur `3_1970-1980` dans `Year_date_diag_cut` devrait vraisemblablement être `3_1980-1990`. À confirmer.

### 6. Faute de frappe
`iccc_type = "04 -Peripheral nervouus tumors"` → *nervouus* devrait être *nervous*.

---

## Schéma conceptuel du jeu de données

```
┌─────────────────────────────────────────────────────────────┐
│                    RT_Thorax_v1.csv                          │
│                   (1 375 patients)                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  IDENTIFICATION          CIBLE              SUIVI           │
│  ┌─────────┐         ┌──────────────┐    ┌──────────┐       │
│  │ ctr     │         │ Pathologie_  │    │ deces    │       │
│  │ numcent │         │ cardiaque    │    │ time     │       │
│  └─────────┘         │ (0/1)        │    │ age_exit │       │
│                      └──────┬───────┘    └──────────┘       │
│                             │                               │
│          ┌──────────────────┼──────────────────┐            │
│          ▼                  ▼                  ▼            │
│  CLINIQUE            TRAITEMENT CT      DOSIMÉTRIE          │
│  ┌────────────┐     ┌───────────────┐   ┌───────────────┐   │
│  │ age_diag   │     │ chimiotherapie│   │ mean (Gy)     │   │
│  │ gender     │     │ _1K           │   │ V5, V10, V15  │   │
│  │ iccc_type  │     │ anthra_1K     │   │ V20           │   │
│  │ Year_diag  │     │ do_anthra_1K  │   │ (DVH cœur)    │   │
│  └────────────┘     │ CT_sans_anthra│   └───────────────┘   │
│                     └───────────────┘                       │
│                                                             │
│  + Données dosimétriques 3D (fichiers séparés)              │
│    → Jointure via ctr + numcent                             │
│    → Matrices de dose voxélisées (2mm)                      │
└─────────────────────────────────────────────────────────────┘
```

---

*Document de référence — Pôle Projet P15 — CentraleSupélec 2025/2026*  
*Encadrant : Rodrigue Allodji (INSERM / Gustave Roussy)*
