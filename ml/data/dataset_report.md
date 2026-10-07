# TrustHire — EMSCAD Dataset Inspection & EDA Report

## 1. Overview
- **Dataset Filename**: `fake_job_postings.csv`
- **Dataset Relative Path**: `data\raw\fake_job_postings.csv`
- **Total Rows**: 17,880
- **Total Columns**: 18
- **Schema Validation Status**: **WARN**

## 2. Target Variable Analysis (`fraudulent`)
- **Data Type**: `object`
- **Unique Values**: `['f', 't']`
- **Value Counts & Distribution**:
  - `0` (Legitimate): 0 (0%)
  - `1` (Potentially Fraudulent): 0 (0%)
- **Class Imbalance Identified**: Yes (Imbalanced Dataset)

## 3. Duplicate Analysis
- **Exact Duplicate Rows**: 235 (1.31%)
- **Duplicate Job IDs**: 0 (0.0%)

## 4. Missing Values Analysis
| Column | Data Type | Missing Count | Missing % |
| --- | --- | --- | --- |
| `salary_range` | `object` | 15,012 | 83.96% |
| `department` | `object` | 11,547 | 64.58% |
| `required_education` | `object` | 8,105 | 45.33% |
| `benefits` | `object` | 7,196 | 40.25% |
| `required_experience` | `object` | 7,050 | 39.43% |
| `function` | `object` | 6,455 | 36.1% |
| `industry` | `object` | 4,903 | 27.42% |
| `employment_type` | `object` | 3,471 | 19.41% |
| `company_profile` | `object` | 3,308 | 18.5% |
| `requirements` | `object` | 2,689 | 15.04% |
| `location` | `object` | 346 | 1.94% |
| `title` | `object` | 0 | 0.0% |
| `description` | `object` | 0 | 0.0% |
| `telecommuting` | `object` | 0 | 0.0% |
| `has_company_logo` | `object` | 0 | 0.0% |
| `has_questions` | `object` | 0 | 0.0% |
| `fraudulent` | `object` | 0 | 0.0% |
| `in_balanced_dataset` | `object` | 0 | 0.0% |

## 5. Text Features Summary
| Column | Non-Empty Count | Empty Count | Char Length (Mean) | Word Count (Mean) |
| --- | --- | --- | --- | --- |
| `title` | 17,880 | 0 | 28.53 | 3.76 |
| `company_profile` | 14,572 | 3,308 | 851.3 | 117.73 |
| `description` | 17,880 | 0 | 1438.56 | 185.97 |
| `requirements` | 15,191 | 2,689 | 860.85 | 105.6 |
| `benefits` | 10,684 | 7,196 | 448.19 | 56.91 |

## 6. Categorical Features Summary
| Column | Unique Count | Missing Count | Top Frequent Value |
| --- | --- | --- | --- |
| `location` | 3,105 | 346 | `GB, LND, London` (718) |
| `department` | 1,337 | 11,547 | `Sales` (551) |
| `employment_type` | 5 | 3,471 | `Full-time` (11,620) |
| `required_experience` | 7 | 7,050 | `Mid-Senior level` (3,809) |
| `required_education` | 13 | 8,105 | `Bachelor's Degree` (5,145) |
| `industry` | 131 | 4,903 | `Information Technology and Services` (1,734) |
| `function` | 37 | 6,455 | `Information Technology` (1,749) |

## 7. Salary Analysis (`salary_range`)
- **Missing Count**: 15,012 (83.96%)
- **Unique Values Count**: 874
- **Sample Values**: `['20000-28000', '100000-120000', '120000-150000', '50000-65000', '40000-50000']`

---
*Report generated programmatically during Step 2 EMSCAD Dataset Inspection.*