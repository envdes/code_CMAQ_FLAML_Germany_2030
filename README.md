# JialingQin_CMAQ-FLAML

# Introduction

This repository is a supplementary to the manuscript "Future Projections of Ozone and Public Health Risks in Germany and Neighboring Regions using Automated Machine Learning (AutoML)"

The objectives of this project are:

* To develop develop an improved ozone projection framework by integrating Machine Learning (ML) with CMAQ simulations

* Project future ozone concentrations for Germany and its adjacent region under different 2030 Shared Socioeconomic Pathways (SSPs) and evaluate their compliance with the European Union's 2030 air quality targets

* Quantify health impacts of ozone pollution under climate scenarios

# Scripts And Data

## Prerequisite

* If you do not have the "[conda](https://docs.conda.io/en/latest/)" system

```shell
# Download and install conda
$ wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
$ chmod +x Miniconda3-latest-Linux-x86_64.sh
$ ./Miniconda3-latest-Linux-x86_64.sh
# Edit .bash_profile or .bashrc
PATH=$PATH:$HOME/.local/bin:$HOME/bin:$HOME/miniconda3/bin
# Activate the conda system
$source .bash_profile
# OR source .bashrc
```

* Create and activate your own conda environment

```shell
# Create an environment "partmc" and install the necessary packages
conda env create -f environment.yml
# Activate the "partmc" environment
conda activate uhws
```

## Scripts

| Tasks                          | Folders                | Fig or Tab in paper                                                                              |
| ------------------------------ | ---------------------- | ------------------------------------------------------------------------------------------------ |
| Data preparation               | 1\_data\_prep          |                                                                                                  |
| Model development & validation | 2\_model\_train\&valid | Fig 3(b)                                                                                         |
| Model application              | 3\_model\_application  |                                                                                                  |
| Data analysis                  | 4\_result\_analysis    | Fig 3 (model performace), Fig 4 (future projection), Fig 5, Fig S1 (other model), Fig S2 (metro) |
| Appendix (SI)                  | 5\_appendix            | Fig S1 (other model), Fig S2 (metro), Fig S3 (error metrics & exceedance), Fig S4 (2021 validation), Fig S5 (EMEP inventory), Tab S2–S4 (health effect) |

## Data

* 1\_data\_prep

| Num  | Folder                                                        | Comments                        | How to get it?                                            |
| ---- | ------------------------------------------------------------- | ------------------------------- | --------------------------------------------------------- |
| 1.1  | CMAQ\_MDA8\_{year}.csv                                        | CMAQ output ozone concentration | Raw CMAQ output and script 'CMAQ\_raw\_nc\_to\_csv.ipynb' |
| 1.2  | ERA5\_Daily\_{date}.nc                                        | ERA5 daily data                 | Download from website                                     |
| 1.3  | {species}\_{year}.csv                                         | ERA5 data csv                   | Data 1.2 & script 'ERA5\_nc\_to\_csv.ipynb'               |
| 1.4  | multiple-states\_input4MIPs\_landState\_ScenarioMIP\_UofMD.nc | Landuse data                    | Download from website                                     |
| 1.5  | population\_{year}.nc                                         | Population data                 | Download from website                                     |
| 1.6  | landuse/population\_{year}.csv                                | landuse\&population data        | Data 1.4 & 1.5 & script 'pop\&lu\_nc\_to\_csv.ipynb'      |
| 1.7  | obs\_MDA8.csv                                                 | Observation data                | Download from website                                     |
| 1.8  | CMIP6\_metro.nc                                               | CMIP6 metro data                | Download from website                                     |
| 1.9  | CMIP6\_metro.csv                                              | CMIP6 metro data                | Data 1.8 & script 'CMIP6\_nc\_to\_csv.ipynb'              |
| 1.10 | processed\_metro.csv                                          | Future metro data(processed)    | Data 1.3 & 1.8 & script 'metro\_ratio.ipynb'              |

* 2\_model\_train\&valid

| Num | Folder                                      | Comments                 | How to get it?                                                                |
| --- | ------------------------------------------- | ------------------------ | ----------------------------------------------------------------------------- |
| 2.1 | preML\_{year}.csv                           | Merged data for training | Data1.1&1.3&1.6&1.7&1.9&1.10 & script 'merge\_multisource.ipynb'              |
| 2.2 | automl1.pkl                                 | Trained model            | Data 2.1 & script 'model\_training.ipynb'                                     |
| 2.3 | Model-validation (available at data folder) | Model validation result  | Data 2.1 & scripts 'model\_training.ipynb'&'fig\_3\_model\_performance.ipynb' |

* 3\_model\_application

| Num | Folder                                | Comments               | How to get it?                              |
| --- | ------------------------------------- | ---------------------- | ------------------------------------------- |
| 3.1 | 2030\_pred (available at data folder) | 2030 ozone predictions | Data2.1 & script 'model\_application.ipynb' |

* 4\_result\_analysis

| Num | Folder                          | Comments                    | How to get it?                                                                                                                                  |
| --- | ------------------------------- | --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| 4.1 | Model explaination & prediction | Data shown in Fig 3         | Data 3.1 & 2.2 & scripts 'fig\_3\_4\_model\_performance.ipynb' & 'model\_explaination.py'                                                        |
| 4.2 | Future projections & analysis   | Data shown in Fig 5 & Fig 6 | Data 3.1 & script 'fig\_5\_6\_future\_projections.ipynb' (Fig 5 & Fig 6) & Origin projects 'fig\_6\_a.opju' (Fig 6a) & 'fig\_6\_b.opju' (Fig 6b) |
| 4.3 | Health impact analysis          | Data shown in Fig 7         | Data 3.1 & 1.1 & 1.5 & scripts 'fig\_7\_health\_impact\_data.py' & 'fig\_7\_health\_impact\_plot.ipynb' & Origin project 'fig\_7\_a.opju' (Fig 7a) |

* 5\_appendix

| Num | Fig or Tab in SI | Scripts                                                                                                                    |
| --- | ---------------- | -------------------------------------------------------------------------------------------------------------------------- |
| 5.1 | Figure S1        | 'fig\_s1\_othermodel.ipynb'                                                                                                |
| 5.2 | Figure S2        | 'fig\_s2\_metro.ipynb'                                                                                                    |
| 5.3 | Figure S3        | 'fig\_s3\_error\_metrics\_exceedance.ipynb'                                                                                |
| 5.4 | Figure S4        | 'fig\_s4\_2021\_validation.ipynb'                                                                                          |
| 5.5 | Figure S5        | 'fig\_s5\_emep/train\_flaml\_emep\_36km.py', 'fig\_s5\_emep/train\_flaml\_2018.py', 'fig\_s5\_emep/apply\_model\_2019.py', 'fig\_s5\_emep/fig\_s5\_plot.ipynb' |
| 5.6 | Table S2–S4      | 'tab\_s2\_s3\_health\_effect.py'                                                                                           |

# Acknowledgement