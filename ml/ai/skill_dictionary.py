"""
TrustHire - Configurable Skill Dictionary for Job Extraction
Provides a comprehensive set of technical and domain skill keywords for pattern matching.
"""

from typing import List, Set, Dict

# Core technical skills organized by domain
DEFAULT_SKILL_DICTIONARY: Dict[str, List[str]] = {
    "programming_languages": [
        "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "C", "Go", "Golang",
        "Rust", "Ruby", "PHP", "Swift", "Kotlin", "R", "Scala", "SQL", "HTML", "CSS", "Bash", "Shell"
    ],
    "web_frameworks": [
        "React", "React.js", "React Native", "Angular", "Vue", "Vue.js", "Node.js", "Express",
        "FastAPI", "Django", "Flask", "Spring", "Spring Boot", "ASP.NET", "Next.js", "Tailwind"
    ],
    "cloud_and_devops": [
        "AWS", "Amazon Web Services", "Azure", "GCP", "Google Cloud", "Docker", "Kubernetes",
        "Git", "GitHub", "GitLab", "CI/CD", "Jenkins", "Terraform", "Ansible", "Linux"
    ],
    "data_and_ai": [
        "Machine Learning", "Deep Learning", "NLP", "Natural Language Processing",
        "Computer Vision", "TensorFlow", "PyTorch", "Scikit-learn", "Pandas", "NumPy",
        "Matplotlib", "Seaborn", "Spark", "Hadoop", "Tableau", "Power BI",
        "Data Structures", "Algorithms"
    ],
    "databases": [
        "PostgreSQL", "MySQL", "MongoDB", "SQLite", "Redis", "Elasticsearch", "DynamoDB", "Oracle"
    ]
}


class SkillDictionary:
    """Configurable skill lookup engine."""

    def __init__(self, custom_skills: List[str] = None):
        self.skills: Set[str] = set()
        for category, skill_list in DEFAULT_SKILL_DICTIONARY.items():
            for skill in skill_list:
                self.skills.add(skill)

        if custom_skills:
            for skill in custom_skills:
                self.skills.add(skill)

    def get_all_skills(self) -> List[str]:
        """Returns sorted list of all registered skills."""
        return sorted(list(self.skills), key=lambda x: len(x), reverse=True)

    def add_skills(self, new_skills: List[str]):
        """Dynamically adds new skills to the dictionary."""
        for skill in new_skills:
            self.skills.add(skill)


# Global default instance
default_skill_dictionary = SkillDictionary()
