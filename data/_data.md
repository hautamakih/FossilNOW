# Data exploration

## 1. File groupping

### 1.1. Occurrence group

This group includes files showing the occurence of genera given sites.

#### a. AllSites_SiteOccurrences_AllGenera_26.1.24.csv

The occurence matrix converted directly from raw database from NOW. Leftmost column contains list of sites whereas header row contains species's name.

#### b. AllNOW_genera_26.1.24

The occurence matrix converted directly from raw database from NOW. Leftmost column contains list of sites whereas header row contains species's name. Probably the difference between a/ and b/ is that the file in b/ is processed.

#### c. data_occ

Occurence of species/genus/family/order (not yet known).

#### d. predictions_occ_train

Just a prediction of occurence from R code.

#### e. LargeSites_SiteOccurrences_LargeGenera_26.1.24

Occurence matrix; like a/ but for large sites and major genera only.

### 1.2. Raw data

This group includes piece or entire raw NOW dataset.

#### a. now_export_locsp_public_2024-02-08T19#42#46+0000

Full dataset from NOW database website. Contain metadata about sites and species. Each line contains data about one single species, its metadata and information about the site which the species was found.

### 1.3. Dental trait

This group includes files containing information about charactertistics of species such as (taxonomy, diet, body...). These information is side data and can be used to build the species's profile.

#### a.DentalTraits_Species_PPPA

Contains side information about species such as diet characteristis, body mass...

#### b.DentalTraits_Genus_PPPA

Contains side information about diet, body mass but at genus level (genus is higher than species).

#### c. FossilGenera_MammalMassDiet_Jan24

Contains taxonomy, mass, size and diet type information on genus level.

## 2. Profile building

### 2.1. Species profile building

This section describes the fields which are from given data files and can be used to to build species profile.

| Field                    | Column name             | Field type | Data type          | From file           |
| ------------------------ | ----------------------- | ---------- | ------------------ | ------------------- |
| order                    | order                   | taxonomy   | categorical (text) | 1.2.a, 1.3.a, 1.3.c |
| family                   | family                  | taxonomy   | categorical (text) | 1.2.a, 1.3.a, 1.3.c |
| genus                    | genus                   | taxonomy   | categorical (text) | 1.2.a, 1.3.a, 1.3.c |
| body mass                | BODYMASS, Mass.g, Massg | size       | numerical          | 1.2.a, 1.3.a, 1.3.c |
| Diet.Plant               | Diet.Plant              | diet       | numerical          | 1.3.a               |
| Diet.Vertebrate          | Diet.Vertebrate         | diet       | numerical          | 1.3.a               |
| Diet.Invertebrate        | Diet.Invertebrate       | diet       | numerical          | 1.3.a               |
| ?                        | SizeClass               | ?          | categorical (text) | 1.3.c               |
| Diet type (genera level) | Diet                    | diet       | categorical (text) | 1.3.c               |

### 2.2. Site profile building

This section describes the fields which are from given data files and can be used to to build site profile.

| Field   | Column name | Field type | Data type          | From file |
| ------- | ----------- | ---------- | ------------------ | --------- |
| lat     | LAT         | locality   | numerical          | 1.2.a     |
| long    | LONG        | locality   | numerical          | 1.2.a     |
| country | COUNTRY     | locality   | categorical (text) | 1.2.a     |
| state   | STATE       | locality   | categorical (text) | 1.2.a     |
| county  | COUNTY      | locality   | categorical (text) | 1.2.a     |
| min age | MIN_AGE     | age        | numerical          | 1.2.a     |
| max age | MAX_AGE     | age        | numerical          | 1.2.a     |
