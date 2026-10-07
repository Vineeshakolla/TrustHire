"""
TrustHire - Pattern Matching & Regular Expressions for NLP Extraction
Contains compiled regex patterns for salary, experience, location, duration, education, and role attributes.
"""

import re
from typing import Dict, Any, Optional, List, Tuple

# Salary & Stipend Regex Patterns
SALARY_PATTERNS = [
    # ₹30,000 – ₹50,000 per month / $60,000-$80,000 annually / 20,000-30,000/month
    re.compile(r'(?:stipend|salary|pay|remuneration)?\s*:?\s*([₹$€£]|INR|USD|EUR|GBP|Rs\.?|Rupees)?\s*([\d,]+(?:\.\d+)?)\s*(?:-|–|to)\s*([₹$€£]|INR|USD|EUR|GBP|Rs\.?|Rupees)?\s*([\d,]+(?:\.\d+)?)\s*(?:per|\/|a)?\s*(month|monthly|year|yearly|annually|hour|hourly|hr)?', re.IGNORECASE),
    # Fixed salary e.g. ₹50,000 per month / $70,000 / year
    re.compile(r'(?:stipend|salary|pay)?\s*:?\s*([₹$€£]|INR|USD|EUR|GBP|Rs\.?|Rupees)\s*([\d,]+(?:\.\d+)?)\s*(?:per|\/|a)?\s*(month|monthly|year|yearly|annually|hour|hourly|hr)?', re.IGNORECASE),
    # Plain range without currency e.g. 30000-50000 per month
    re.compile(r'([\d,]{4,})\s*(?:-|–|to)\s*([\d,]{4,})\s*(?:per|\/|a)?\s*(month|monthly|year|yearly|annually|hour|hourly|hr)?', re.IGNORECASE)
]

# Experience Patterns
EXPERIENCE_PATTERNS = [
    re.compile(r'(?:min(?:imum)?\s*experience|experience)\s*:?\s*(\d+\+?\s*(?:-|–|to)?\s*\d*\s*years?)', re.IGNORECASE),
    re.compile(r'(\d+\+?\s*(?:-|–|to)?\s*\d*\s*years?\s*(?:of)?\s*experience)', re.IGNORECASE),
    re.compile(r'(junior\s*-\s*entry\s*level|entry\s*level|senior|mid-senior|executive|associate)', re.IGNORECASE)
]

# Duration Patterns
DURATION_PATTERNS = [
    re.compile(r'duration\s*:?\s*([^\n\r]+)', re.IGNORECASE),
    re.compile(r'((?:minimum\s*)?\d+\s*(?:-|–|to)?\s*\d*\s*(?:months?|weeks?)(?:\s*,\s*extendable\s*up\s*to\s*\d+\s*months?)?)', re.IGNORECASE)
]

# Start Date Patterns
START_DATE_PATTERNS = [
    re.compile(r'start\s*date\s*:?\s*([^\n\r]+)', re.IGNORECASE),
    re.compile(r'(immediate(?:ly)?(?:\s*\/\s*as\s*soon\s*as\s*possible)?)', re.IGNORECASE)
]

# Work Mode Patterns
WORK_MODE_PATTERNS = [
    (re.compile(r'\b(remote|work\s*from\s*home|wfh)\b', re.IGNORECASE), "Remote"),
    (re.compile(r'\b(hybrid)\b', re.IGNORECASE), "Hybrid"),
    (re.compile(r'\b(on-site|onsite|in-office)\b', re.IGNORECASE), "On-site")
]

# Employment Type Normalization Map
EMPLOYMENT_TYPE_MAP = {
    "full-time": "Full-time",
    "full time": "Full-time",
    "fulltime": "Full-time",
    "part-time": "Part-time",
    "part time": "Part-time",
    "contract": "Contract",
    "temporary": "Temporary",
    "internship": "Internship",
    "intern": "Internship",
    "freelance": "Freelance",
    "volunteer": "Volunteer"
}

# Education Keywords
EDUCATION_KEYWORDS = [
    "Bachelor's degree", "B.Tech", "B.E.", "B.S.", "BS", "Master's degree",
    "M.Tech", "M.S.", "MS", "Ph.D.", "Computer Science", "Engineering", "STEM"
]

# Title Explicit Prefix Patterns
TITLE_PREFIX_PATTERNS = [
    re.compile(r'(?:job\s*title|role|position)\s*:?\s*([^\n\r]+)', re.IGNORECASE),
    re.compile(r'(?:we\s*are\s*looking\s*for\s*a|hiring\s*a)\s+([^\n\r,\.]+)', re.IGNORECASE)
]


def parse_salary_text(text: str) -> Optional[Dict[str, Any]]:
    """Parses raw text for salary/stipend expressions and returns structured numerical output."""
    if not text:
        return None

    for pattern in SALARY_PATTERNS:
        match = pattern.search(text)
        if match:
            groups = match.groups()
            raw_str = match.group(0).strip()
            
            # Extract currency
            currency = None
            for g in groups:
                if g and g.upper() in ["₹", "INR", "RS", "RUPEES"]:
                    currency = "INR"
                    break
                elif g and g.upper() in ["$", "USD"]:
                    currency = "USD"
                    break
                elif g and g.upper() in ["€", "EUR"]:
                    currency = "EUR"
                    break
                elif g and g.upper() in ["£", "GBP"]:
                    currency = "GBP"
                    break

            # Extract numbers
            numbers = []
            for g in groups:
                if g:
                    clean_g = g.replace(',', '')
                    try:
                        val = float(clean_g)
                        numbers.append(val)
                    except ValueError:
                        pass

            min_sal = numbers[0] if len(numbers) > 0 else None
            max_sal = numbers[1] if len(numbers) > 1 else min_sal

            # Extract period
            period = None
            raw_lower = raw_str.lower()
            if "month" in raw_lower or "stipend" in raw_lower:
                period = "month"
            elif "year" in raw_lower or "annual" in raw_lower:
                period = "year"
            elif "hour" in raw_lower or "hr" in raw_lower:
                period = "hour"

            return {
                "raw": raw_str,
                "minimum": min_sal,
                "maximum": max_sal,
                "currency": currency or ("INR" if "₹" in text or "Rs" in text else None),
                "period": period
            }

    return None
