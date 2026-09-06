"""Curated knowledge that must always be retrievable.

These facts used to be pasted into the system prompt, which meant paying for
them on *every* request whether or not anyone asked. They are now ordinary
corpus documents: retrieved when relevant, invisible when not. Keeping them in
code (rather than in ``data/``) guarantees they exist even if the data directory
is misconfigured.
"""

from __future__ import annotations

from typing import List, Tuple

_ABOUT_SINA = """
# About SINA and the team behind it

## Who built this assistant

SINA is the AI assistant for R.N.G. Patel Institute of Technology (RNGPIT).
It was built by **Team InnoCrew**, a group of students from RNGPIT.

| Member | Role | Programme |
| --- | --- | --- |
| Shis Tushar Maheta | Lead AI Engineer | B.Tech Computer Science, Class of 2025 |
| Zuveriya Meman | Developer | B.Voc Software Development, Class of 2025 |
| Karan Chaudhary | Developer | B.Voc Software Development, Class of 2023 |
| Sem Surti | Developer | B.Voc Software Development, Class of 2023 |
| Shreyansh Vasava | Developer | B.Voc Software Development, Class of 2023 |

Team InnoCrew developed this assistant to help students and visitors learn about
RNG Patel Institute of Technology.

## What SINA can do

SINA answers questions about RNGPIT: courses and programmes, admissions and
eligibility, fees, departments and faculty, laboratories and facilities,
placements and recruiters, committees, events, hostel and campus life, and
contact details. It has a text chat mode and a voice mode with a 3D avatar.

## How to reach the institute

- Email: info@rngpit.ac.in
- Phone: +91-9512900457
- Website: https://rngpit.ac.in
- Address: Bardoli-Navsari Road, Isroli-Afwa, Taluka Palsana, District Surat,
  Gujarat - 394620
""".strip()


CURATED_DOCUMENTS: List[Tuple[str, str]] = [
    ("curated/about-sina.md", _ABOUT_SINA),
]
