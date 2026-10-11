# Data dictionary

**Dataset:** UCI Student Performance (Cortez & Silva, 2008)
**Source:** https://archive.ics.uci.edu/dataset/320/student+performance
**License:** Creative Commons Attribution 4.0 International (CC BY 4.0); DOI: 10.24432/C5TG7T. Attribute the dataset to Paulo Cortez and the UCI Machine Learning Repository.
**Files:** `student-mat.csv` (Math), `student-por.csv` (Portuguese), merged with `subject`.
**PII:** none in the public files. Do not add names, emails, or student IDs.

Scale notes: many ordinal fields are 1â€“5 or 1â€“4 **bands**, not minutes or hours. Treat them as ordered categories encoded as numbers.

| Feature | Type | Unit / coding | Source | Why it may matter | Limitation |
| --- | --- | --- | --- | --- | --- |
| school | categorical | GP / MS | UCI | School context and resources | Only two schools in one region |
| sex | categorical | F / M | UCI | **Not used in models** (fairness) | Binary encoding is outdated |
| age | numeric | years 15â€“22 | UCI | Age can correlate with repeats | Narrow range |
| address | categorical | U urban / R rural | UCI | Travel and resource access | Coarse |
| famsize | categorical | LE3 / GT3 | UCI | Household constraints | Not household income |
| Pstatus | categorical | T together / A apart | UCI | Home stability proxy | Sensitive; not causal |
| Medu / Fedu | ordinal | 0â€“4 education band | UCI | Home academic capital | Self-reported band |
| Mjob / Fjob | categorical | job group | UCI | Time and academic support | Very coarse jobs |
| reason | categorical | home/reputation/course/other | UCI | Motivation for school choice | Not a skill measure |
| guardian | categorical | mother/father/other | UCI | Who is academically present | Sensitive |
| traveltime | ordinal | 1â€“4 commute band | UCI | Time left for study | Not minutes |
| studytime | ordinal | 1â€“4 weekly study band | UCI | Effort / time on task | Self-reported |
| failures | numeric | 0â€“3, else 4 | UCI | Strong academic-risk history | Past labels can freeze a student in the model |
| schoolsup / famsup / paid | binary | yes/no | UCI | Existing support | "yes" can mean already struggling |
| activities | binary | yes/no | UCI | Engagement | Not quality of activity |
| nursery | binary | yes/no | UCI | Early education | Weak, distant signal |
| higher | binary | yes/no | UCI | Educational goal | Circular with performance |
| internet | binary | yes/no | UCI | Access to digital practice | Dated (2008) |
| romantic | binary | yes/no | UCI | Time / attention proxy | Sensitive and stereotyped |
| famrel | ordinal | 1â€“5 | UCI | Home climate | Subjective |
| freetime / goout | ordinal | 1-5 | UCI | Social load | Not "laziness" |
| Dalc / Walc | ordinal | 1â€“5 alcohol | UCI | Health / wellbeing | Sensitive |
| health | ordinal | 1â€“5 | UCI | Health load on study | Self-reported |
| absences | numeric | count 0â€“93 | UCI | Attendance | No official attendance rate |
| G1 / G2 | numeric | 0â€“20 period grades | UCI | **Excluded (leakage)** | Would dominate G3 |
| G3 | numeric | 0â€“20 final grade | UCI | Regression target | One course, one year |
| subject | categorical | math / portuguese | derived | Course difficulty differs | Only two subjects |

## Engineered features

| Feature | Formula (simplified) | Intent |
| --- | --- | --- |
| parent_edu_avg | (Medu+Fedu)/2 | Combined home education background |
| attendance_proxy | 1 - clip(absences,0,30)/30 | Attendance-style signal |
| alcohol_index | (Dalc+Walc)/2 | Combined wellbeing flag |
| study_vs_travel | studytime / traveltime | Effort net of commute |
| support_count | schoolsup+famsup+paid as 0/1 | How much support is already in place |
| engagement_score | studytime + activities + internet + higher | Composite engagement |
| social_load | mix of goout, freetime, romantic | Time competition |

## Support label

`high_support` if G3 < 10; `medium_support` if 10 â‰¤ G3 < 14; `low_support` if G3 â‰¥ 14.
