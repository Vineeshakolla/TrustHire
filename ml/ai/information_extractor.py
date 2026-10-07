"""
TrustHire - Layered Local AI/NLP Job Information Extraction Core Module
Applies layered NLP, Regex patterns, Section Parsing, and Skill Dictionaries to structure job postings.
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from ml.ai.schemas import ExtractedJobInformation, SalaryInformation
from ml.ai.patterns import (
    parse_salary_text, EXPERIENCE_PATTERNS, DURATION_PATTERNS,
    START_DATE_PATTERNS, WORK_MODE_PATTERNS, EMPLOYMENT_TYPE_MAP,
    EDUCATION_KEYWORDS, TITLE_PREFIX_PATTERNS
)
from ml.ai.skill_dictionary import default_skill_dictionary


def extract_sections(text: str) -> Dict[str, List[str]]:
    """
    Parses job posting text into key structural sections (Responsibilities, Requirements, Benefits).
    Extracts bullet points and distinct text items.
    """
    sections: Dict[str, List[str]] = {
        "responsibilities": [],
        "requirements": [],
        "benefits": []
    }

    if not text:
        return sections

    lines = text.splitlines()
    current_section = None

    for line in lines:
        clean_line = line.strip()
        if not clean_line:
            continue

        lower_line = clean_line.lower()

        # Section Header Detection
        if any(h in lower_line for h in ["responsibilities", "what you'll do", "key duties", "role description"]):
            current_section = "responsibilities"
            continue
        elif any(h in lower_line for h in ["requirements", "qualifications", "what we're looking for", "nice to have", "skills required"]):
            current_section = "requirements"
            continue
        elif any(h in lower_line for h in ["benefits", "perks", "what we offer", "compensation & perks"]):
            current_section = "benefits"
            continue

        # Extract item content if under active section
        if current_section:
            item = re.sub(r'^[•\-\*\d\.\)\s]+', '', clean_line).strip()
            if item and len(item) > 3:
                sections[current_section].append(item)

    return sections


def extract_job_title(title_input: Optional[str], text: str) -> Tuple[Optional[str], str]:
    """Extracts job title from structured input or text patterns."""
    if title_input and title_input.strip() and title_input.lower() not in ["unknown", "none", "n/a"]:
        return title_input.strip(), "high"

    if not text:
        return None, "low"

    # Search for explicit standard role titles
    exact_title_match = re.search(r'\b((?:Software|Data|Backend|Frontend|Full Stack|DevOps|Systems)\s+(?:Engineering|Engineer|Developer|Analyst|Scientist)\s+(?:Intern|Internship))\b', text, re.IGNORECASE)
    if not exact_title_match:
        exact_title_match = re.search(r'\b((?:Software|Data|Backend|Frontend|Full Stack|DevOps|Systems)\s+(?:Engineering|Engineer|Developer|Analyst|Scientist))\b', text, re.IGNORECASE)
    if not exact_title_match:
        exact_title_match = re.search(r'\b((?:Software|Data|Backend|Frontend|Full Stack|DevOps|Systems)\s+(?:Intern|Internship))\b', text, re.IGNORECASE)

    if exact_title_match:
        cand = exact_title_match.group(1).strip()
        if len(cand) > 3:
            return cand, "high"


    for pattern in TITLE_PREFIX_PATTERNS:
        match = pattern.search(text)
        if match:
            extracted = match.group(1).strip()
            if len(extracted) < 80 and "client" not in extracted.lower():
                return extracted, "high"

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in lines[:5]:
        if "client" in line.lower():
            continue
        if any(keyword in line.lower() for keyword in ["engineer", "developer", "intern", "analyst", "manager", "specialist"]):
            return line, "medium"

    return None, "low"



def extract_company_info(company_input: Optional[str], text: str) -> Tuple[Optional[str], Optional[str], str]:
    """
    Extracts company name and company profile.
    If posting states 'This role is for one of our clients', sets company_name to None and profile accordingly.
    """
    company_name = None
    company_profile = None
    confidence = "low"

    if company_input and company_input.strip() and company_input.lower() not in ["unknown", "none", "n/a"]:
        company_profile = company_input.strip()
        if "client" in company_input.lower():
            company_name = None
            confidence = "high"
        else:
            company_name = company_input.strip()
            confidence = "high"
        return company_name, company_profile, confidence

    if text:
        if "this role is for one of our clients" in text.lower() or "client of" in text.lower():
            company_name = None
            company_profile = "This role is for one of our clients"
            confidence = "high"
            return company_name, company_profile, confidence

        comp_match = re.search(r'(?:company|about)\s*:?\s*([A-Z][A-Za-z0-9\s]+(?:Inc|LLC|Labs|Corp|Technologies|Solutions|Ltd)?)', text)
        if comp_match:
            candidate = comp_match.group(1).strip()
            if len(candidate) < 50:
                company_name = candidate
                company_profile = f"About {candidate}"
                confidence = "medium"

    return company_name, company_profile, confidence


def extract_location_and_mode(location_input: Optional[str], text: str) -> Tuple[Optional[str], Optional[str], str]:
    """
    Extracts location (city/country) and work_mode (Remote, On-site, Hybrid).
    Separates location from work mode cleanly (e.g. 'Bengaluru (Remote)' -> location: 'Bengaluru', work_mode: 'Remote').
    """
    combined_str = f"{location_input or ''} {text or ''}"
    location = None
    work_mode = None
    confidence = "low"

    # Work Mode Extraction
    for pattern, mode in WORK_MODE_PATTERNS:
        if pattern.search(combined_str):
            work_mode = mode
            break

    # Location Extraction
    raw_loc = location_input
    if not raw_loc or raw_loc.lower() in ["unknown", "none", "n/a"]:
        loc_match = re.search(r'location\s*:?\s*([^\n\r,]+)', combined_str, re.IGNORECASE)
        if loc_match:
            raw_loc = loc_match.group(1).strip()

    if raw_loc:
        clean_loc = re.sub(r'\(?\b(remote|hybrid|on-site|onsite)\b\)?', '', raw_loc, flags=re.IGNORECASE).strip()
        clean_loc = clean_loc.strip("-,/.:; ")
        if clean_loc and len(clean_loc) >= 2:
            location = clean_loc
            confidence = "high"

    if not work_mode and location and "remote" in location.lower():
        work_mode = "Remote"

    return location, work_mode, confidence


def extract_employment_type(emp_type_input: Optional[str], text: str, title: Optional[str]) -> Tuple[Optional[str], str]:
    """Extracts and normalizes employment type."""
    combined = f"{emp_type_input or ''} {title or ''} {text or ''}".lower()

    if "jobtype:" in combined or "job type:" in combined or "employment type:" in combined:
        match = re.search(r'(?:job\s*type|employment\s*type)\s*:?\s*([^\n\r,]+)', combined)
        if match:
            val = match.group(1).strip().lower()
            if val in EMPLOYMENT_TYPE_MAP:
                return EMPLOYMENT_TYPE_MAP[val], "high"

    for raw_key, norm_val in EMPLOYMENT_TYPE_MAP.items():
        if re.search(r'\b' + re.escape(raw_key) + r'\b', combined):
            return norm_val, "high"

    return None, "low"


def extract_experience(exp_input: Optional[str], text: str) -> Tuple[Optional[str], str]:
    """Extracts explicit experience requirements."""
    if exp_input and exp_input.strip() and exp_input.lower() not in ["unknown", "none", "n/a"]:
        return exp_input.strip(), "high"

    if text:
        for pattern in EXPERIENCE_PATTERNS:
            match = pattern.search(text)
            if match:
                return match.group(1).strip(), "high"

    return None, "low"


def extract_education(edu_input: Optional[str], text: str) -> Tuple[Optional[str], str]:
    """Extracts education requirements."""
    if edu_input and edu_input.strip() and edu_input.lower() not in ["unknown", "none", "n/a"]:
        return edu_input.strip(), "high"

    if text:
        found_edu = []
        for kw in EDUCATION_KEYWORDS:
            if re.search(r'\b' + re.escape(kw) + r'\b', text, re.IGNORECASE):
                found_edu.append(kw)
        if found_edu:
            return ", ".join(found_edu[:3]), "high"

    return None, "low"


def extract_skills(text: str) -> List[str]:
    """Extracts unique matching technical skills from context using SkillDictionary."""
    if not text:
        return []

    found_skills = set()
    all_skills = default_skill_dictionary.get_all_skills()

    for skill in all_skills:
        if skill == "C++":
            pattern = r'C\+\+'
        elif skill == "C#":
            pattern = r'C\#'
        elif skill == ".NET":
            pattern = r'\.NET'
        elif len(skill) <= 2:
            pattern = r'\b' + re.escape(skill) + r'\b'
        else:
            pattern = r'\b' + re.escape(skill) + r'\b'

        if re.search(pattern, text, re.IGNORECASE):
            found_skills.add(skill)

    return sorted(list(found_skills))



def extract_duration_and_start_date(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extracts internship/contract duration and start date."""
    duration = None
    start_date = None

    if text:
        for pattern in DURATION_PATTERNS:
            match = pattern.search(text)
            if match:
                dur_raw = match.group(1).strip().rstrip(".,")
                duration = dur_raw
                break

        for pattern in START_DATE_PATTERNS:
            match = pattern.search(text)
            if match:
                start_raw = match.group(1).strip().rstrip(".,")
                start_date = start_raw
                break

    return duration, start_date


def analyze_missing_information(extracted: ExtractedJobInformation) -> List[str]:
    """Identifies key posting fields that are missing or incomplete."""
    missing = []
    if not extracted.company_name and not extracted.company_profile:
        missing.append("Company Information")
    if not extracted.salary or not extracted.salary.raw:
        missing.append("Salary / Stipend")
    if not extracted.location:
        missing.append("Location")
    if not extracted.employment_type:
        missing.append("Employment Type")
    if not extracted.experience:
        missing.append("Experience Requirements")
    if not extracted.education:
        missing.append("Education Requirements")
    if not extracted.requirements:
        missing.append("Requirements List")
    if not extracted.responsibilities:
        missing.append("Responsibilities List")
    if not extracted.benefits:
        missing.append("Benefits Information")

    return missing


def extract_job_information(
    title: Optional[str] = None,
    description: Optional[str] = None,
    company_profile: Optional[str] = None,
    requirements: Optional[str] = None,
    benefits: Optional[str] = None,
    salary_range: Optional[str] = None,
    location: Optional[str] = None,
    department: Optional[str] = None,
    employment_type: Optional[str] = None,
    required_experience: Optional[str] = None,
    required_education: Optional[str] = None,
    industry: Optional[str] = None,
    function: Optional[str] = None
) -> ExtractedJobInformation:
    """
    Main entry point for local AI/NLP job information extraction.
    Combines text blocks, executes pattern extraction, and returns ExtractedJobInformation schema.
    """
    full_text_blocks = [
        title or '', description or '', company_profile or '',
        requirements or '', benefits or '', salary_range or '', location or ''
    ]
    full_text = "\n".join([b for b in full_text_blocks if b.strip()])

    confidence_scores = {}

    # 1. Job Title
    job_title, conf_title = extract_job_title(title, full_text)
    confidence_scores["job_title"] = conf_title

    # 2. Company Info
    comp_name, comp_prof, conf_comp = extract_company_info(company_profile, full_text)
    confidence_scores["company_name"] = conf_comp

    # 3. Location & Work Mode
    loc, work_mode, conf_loc = extract_location_and_mode(location, full_text)
    confidence_scores["location"] = conf_loc
    confidence_scores["work_mode"] = "high" if work_mode else "low"

    # 4. Employment Type
    emp_type, conf_emp = extract_employment_type(employment_type, full_text, job_title)
    confidence_scores["employment_type"] = conf_emp

    # 5. Salary / Stipend
    salary_raw_source = salary_range or full_text
    parsed_sal = parse_salary_text(salary_raw_source)
    salary_obj = SalaryInformation(**parsed_sal) if parsed_sal else None
    confidence_scores["salary"] = "high" if salary_obj else "low"

    # 6. Experience
    exp, conf_exp = extract_experience(required_experience, full_text)
    confidence_scores["experience"] = conf_exp

    # 7. Education
    edu, conf_edu = extract_education(required_education, full_text)
    confidence_scores["education"] = conf_edu

    # 8. Skills
    skills = extract_skills(full_text)
    confidence_scores["skills"] = "high" if skills else "low"

    # 9. Duration & Start Date
    duration, start_date = extract_duration_and_start_date(full_text)
    confidence_scores["duration"] = "high" if duration else "low"

    # 10. Internship Detection
    is_internship = False
    combined_intern_text = f"{job_title or ''} {emp_type or ''} {full_text}".lower()
    if any(k in combined_intern_text for k in ["intern", "internship"]):
        is_internship = True

    # 11. Section Extraction (Responsibilities, Requirements, Benefits)
    parsed_sections = extract_sections(full_text)
    
    req_list = parsed_sections["requirements"]
    if requirements and requirements.strip() and not req_list:
        req_list = [r.strip() for r in requirements.splitlines() if r.strip()]

    ben_list = parsed_sections["benefits"]
    if benefits and benefits.strip() and not ben_list:
        ben_list = [b.strip() for b in benefits.splitlines() if b.strip()]

    resp_list = parsed_sections["responsibilities"]

    # Construct schema
    result = ExtractedJobInformation(
        job_title=job_title,
        company_name=comp_name,
        company_profile=comp_prof,
        location=loc,
        work_mode=work_mode,
        employment_type=emp_type,
        salary=salary_obj,
        experience=exp,
        education=edu,
        skills=skills,
        duration=duration,
        start_date=start_date,
        industry=industry if (industry and industry.lower() != "unknown") else None,
        department=department if (department and department.lower() != "unknown") else None,
        responsibilities=resp_list,
        requirements=req_list,
        benefits=ben_list,
        internship=is_internship,
        confidence_scores=confidence_scores
    )

    result.important_missing_information = analyze_missing_information(result)

    return result
