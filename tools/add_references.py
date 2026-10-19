"""Add structured references to the model files (ids match metrology-atlas datasets/vitaldb/references.yaml)."""
import json
import os

MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'vitaldb_resolution', 'data', 'models')
REF = {
 'rietveld2002rounding': {'citation': 'Rietveld P. Rounding of arrival and departure times in travel surveys: an interpretation in terms of scheduled activities. J Transp Stat. 2002;5(1):71-81.',
                          'url': 'https://www.bts.gov/archive/publications/journal_of_transportation_and_statistics/volume_05_number_01/paper_05/index',
                          'used_for': 'rounding probabilities fitted by maximum likelihood; Bayes rule for the true value given the reported one'},
 'healey2011diurnal': {'citation': 'Healey EL, Burden AM, McEwan IM, Fowler NE. Diurnal variation in stature: do those with chronic low-back pain differ from asymptomatic controls? Clin Biomech. 2011;26(4):331-336.',
                       'doi': '10.1016/j.clinbiomech.2010.11.017', 'used_for': 'diurnal loss of stature (measured-height SD)'},
 'leatt1986spinal': {'citation': 'Leatt P, Reilly T, Troup JG. Spinal loading during circuit weight-training and running. Br J Sports Med. 1986;20(3):119-124.',
                     'doi': '10.1136/bjsm.20.3.119', 'used_for': 'diurnal loss of stature (measured-height SD)'},
 'voss1997diurnal': {'citation': 'Voss LD, Bailey BJR. Diurnal variation in stature: is stretching the answer? Arch Dis Child. 1997;77(4):319-322.',
                     'doi': '10.1136/adc.77.4.319', 'used_for': 'diurnal variation and stadiometer reproducibility (measured-height SD)'},
 'ko2022selfreported': {'citation': 'Ko Y, Choi S, Won J, Lee YK, Kim DH, Lee SK. Differences in accuracy of height, weight, and body mass index between self-reported and measured using the 2018 Korea Community Health Survey data. Epidemiol Health. 2022;44:e2022024.',
                        'doi': '10.4178/epih.e2022024', 'used_for': 'self-reported minus measured height and weight by sex and age (written-in bias and SD)'},
 'khosla1964bodyweight': {'citation': 'Khosla T, Billewicz WZ. Measurement of change in body-weight. Br J Nutr. 1964;18(1):227-239.',
                          'doi': '10.1079/BJN19640022', 'used_for': 'day-to-day variation of body weight (measured-weight SD)'},
 'vivanti2013weight': {'citation': 'Vivanti A, Yu L, Palmer M, Dakin L, Sun J, Campbell K. Short-term body weight fluctuations in older well-hydrated hospitalised patients. J Hum Nutr Diet. 2013;26(5):429-435.',
                       'doi': '10.1111/jhn.12034', 'used_for': 'short-term weight fluctuation in hospital (measured-weight SD)'},
 'lee2022vitaldb': {'citation': 'Lee HC, Park Y, Yoon SB, Yang SM, Park D, Jung CW. VitalDB, a high-fidelity multi-parameter vital signs database in surgical patients. Sci Data. 2022;9:279.',
                    'doi': '10.1038/s41597-022-01411-5', 'used_for': 'data source'},
}
USE = {'height': ['lee2022vitaldb', 'rietveld2002rounding', 'leatt1986spinal', 'healey2011diurnal', 'voss1997diurnal', 'ko2022selfreported'],
       'weight': ['lee2022vitaldb', 'rietveld2002rounding', 'khosla1964bodyweight', 'vivanti2013weight', 'ko2022selfreported']}
HEAPED = ['intraop_uo', 'intraop_ebl', 'intraop_crystalloid', 'intraop_ppf', 'anestart', 'aneend', 'opstart', 'opend']
for f in sorted(os.listdir(MODELS)):
    v = f.replace('.model.json', ''); p = os.path.join(MODELS, f); m = json.load(open(p))
    ids = USE.get(v) or (['lee2022vitaldb', 'rietveld2002rounding'] if v in HEAPED else ['lee2022vitaldb'])
    m['references'] = [{'id': i, **REF[i]} for i in ids]
    mv = m.get('measurement_variability')
    if mv is not None:
        mv['references'] = [REF[i]['citation'] + (f" doi:{REF[i]['doi']}" if 'doi' in REF[i] else '') for i in ids if i not in ('lee2022vitaldb', 'rietveld2002rounding')]
    json.dump(m, open(p, 'w'), indent=1)
    print(v, ids)
