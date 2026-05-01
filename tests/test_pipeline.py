"""
Tests for the data pipeline.
Run with: pytest tests/test_pipeline.py -v
"""

import pytest
import os
import csv


MEMBERS_CSV      = os.path.join(os.path.dirname(__file__), "../data/family_members.csv")
RELATIONSHIPS_CSV = os.path.join(os.path.dirname(__file__), "../data/family_relationships.csv")


def load_members():
    with open(MEMBERS_CSV, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def load_relationships():
    with open(RELATIONSHIPS_CSV, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


# --- CSV Data Tests ---

def test_members_file_exists():
    assert os.path.exists(MEMBERS_CSV), "family_members.csv not found"

def test_relationships_file_exists():
    assert os.path.exists(RELATIONSHIPS_CSV), "family_relationships.csv not found"

def test_minimum_members():
    members = load_members()
    assert len(members) >= 50, f"Expected 50+ members, got {len(members)}"

def test_minimum_relationships():
    rels = load_relationships()
    assert len(rels) >= 75, f"Expected 75+ relationships, got {len(rels)}"

def test_required_member_columns():
    members = load_members()
    required = ["Full Name", "First Name", "Last Name", "Gender", "Born"]
    for col in required:
        assert col in members[0], f"Missing column: {col}"

def test_required_relationship_columns():
    rels = load_relationships()
    assert "Person A" in rels[0]
    assert "Related to" in rels[0]
    assert "Person B" in rels[0]

def test_no_empty_full_names():
    members = load_members()
    for m in members:
        assert m["Full Name"].strip() != "", f"Empty Full Name found: {m}"

def test_relationship_names_match_members():
    members = {r["Full Name"].strip() for r in load_members()}
    rels     = load_relationships()
    mismatches = []
    for row in rels:
        a = row["Person A"].strip()
        b = row["Person B"].strip()
        if a not in members:
            mismatches.append(f"Person A not found: '{a}'")
        if b not in members:
            mismatches.append(f"Person B not found: '{b}'")
    assert not mismatches, "\n".join(mismatches)

def test_valid_relationship_types():
    rels  = load_relationships()
    valid = {"FATHER", "MOTHER", "SPOUSE", "CHILD", "SIBLING"}
    for row in rels:
        rel = row["Related to"].strip().upper()
        assert rel in valid, f"Invalid relationship type: '{rel}'"

def test_gender_values():
    members = load_members()
    valid   = {"Male", "Female", ""}
    for m in members:
        assert m["Gender"].strip() in valid, f"Unexpected gender: '{m['Gender']}' for {m['Full Name']}"