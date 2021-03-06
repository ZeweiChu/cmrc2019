## The Third Evaluation Workshop on Chinese Machine Reading Comprehension (CMRC 2019)

This is the official repository for CMRC 2019.

More details:

- CMRC 2019 Official Website (中文)：[https://cmrc2019.hfl-rc.com/](https://hfl-rc.github.io/cmrc2019/)

- CMRC 2019 Official Website (English)：[https://cmrc2019.hfl-rc.com/english/](https://hfl-rc.github.io/cmrc2019/english/)

## Updates
August 1, 2019	**Qualifying data** has been released, check *data* directory. (**Note that, json key 'answers' no longer contains answers.**)  
June 10, 2019	**Development data** has been released, check *data* directory.  
May 24, 2019  **Baseline system** has been released, check *baseline* directory.  
May 23, 2019  **Train/Trial data** have been released, check *data* directory.  
May 17, 2019	We are expected to release task introduction and call for participation on late May or early June. Stay tuned!  

## Directory Guide

- baseline: a Chinese BERT-based simple baseline system

- eval: contains official evaluation script

- data: contains offical evaluation data

- sample_submission: sample submission for codalab competition platform (`trial_rand_submission.zip` is a randomly generated prediction file, `trial_submission.zip` is the BERT baseline prediction file)


## Baseline System

We provide a BERT-based baseline system for participants (check *baseline* directory for more info).

Results on other sets will be annouced later.

> QAC: Question-Level Accuracy

> PAC: Passage-Level Accuracy

| Data | Passage # | Query # | QAC | PAC | Fake Candidates | Availability |
| :------ | :-----: | :-----: | :-----: | :-----: | :-----: | :----- | 
| Trial Data | 139 | 1,504 | 71.941% | 28.776% | No | Public |
| Train Data | 9,638 | 100,009 | N/A | N/A | No | Public |
| Development Data | 300 | 3,053 | 70.586% | 13.333% | **Yes** | Public |
| Qualifying Data | 500 | 5,081 | 70.01% | 8.20% | **Yes** | Semi-Hidden |
| Test Data | - | - |  - | - | **Yes** | Hidden |


## Organization Committee
Host: Chinese Information Processing Society of China (CIPS)  
Organizer: Joint Laboratory of HIT and iFLYTEK (HFL)  
Sponsor: iFLYTEK Co., Ltd. and iFLYTEK Research (Hebei)  

## Evaluation Co-Chairs
Ting Liu, Harbin Institute of Technology  
Yiming Cui, iFLYTEK Research

## Official Discussion QQ Group
You can join our QQ group for discussion purpose.

![qqgroup.png](https://github.com/ymcui/cmrc2019/raw/master/qqgroup.png) 


## Official HFL WeChat Account
Follow Joint Laboratory of HIT and iFLYTEK Research (HFL) on WeChat.

![qrcode.png](https://github.com/ymcui/cmrc2019/raw/master/qrcode.jpg)


## Contact us
Any problems? Feel free to concat us.  
Email: **[cmrc2019 [aT] 126 [DoT] com](mailto:cmrc2019@126.com)**  
Forum: [CodaLab Competition Forum](https://competitions.codalab.org/forums/19781/)
