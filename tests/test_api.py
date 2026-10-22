"""
Automated Tests for Article NER Service API
Tests cover: normal input, empty/invalid input, punctuation, repeated entities,
non-ASCII/dates, and text length limits.
"""
import json
import sys
import os
import pytest

# Add project root and src to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from fastapi.testclient import TestClient
from src.app import app, MAX_COMBINED_LENGTH

client = TestClient(app)


class TestNormalInput:
    """Test 1: Normal article input with valid entities."""
    
    def test_normal_article_returns_200(self):
        """Normal article should return 200 with valid entity structure."""
        response = client.post("/extract-entities", json={
            "title": "Maybank Reports Strong Q3 Results in Kuala Lumpur",
            "body": "Maybank CEO Dato Khairussaleh announced record profits on September 15, 2026. "
                    "The bank headquartered in Kuala Lumpur exceeded analyst expectations."
        })
        assert response.status_code == 200
        data = response.json()
        
        # Check JSON schema
        assert "entities" in data
        assert "meta" in data
        assert isinstance(data["entities"], list)
        assert "entity_count" in data["meta"]
        assert "model" in data["meta"]
        assert data["meta"]["entity_count"] == len(data["entities"])
    
    def test_entity_offsets_are_accurate(self):
        """Verify that start/end offsets accurately match entity text."""
        response = client.post("/extract-entities", json={
            "title": "Elon Musk visits Tokyo",
            "body": "Tesla CEO Elon Musk arrived in Tokyo on Monday for business meetings with Japanese officials."
        })
        assert response.status_code == 200
        data = response.json()
        
        # Reconstruct combined text
        combined = "Elon Musk visits Tokyo\n\nTesla CEO Elon Musk arrived in Tokyo on Monday for business meetings with Japanese officials."
        
        for entity in data["entities"]:
            assert "text" in entity
            assert "label" in entity
            assert "start" in entity
            assert "end" in entity
            assert isinstance(entity["start"], int)
            assert isinstance(entity["end"], int)
            assert entity["start"] >= 0
            assert entity["end"] > entity["start"]
            # Verify offset accuracy
            extracted = combined[entity["start"]:entity["end"]]
            assert extracted == entity["text"], (
                f"Offset mismatch: combined[{entity['start']}:{entity['end']}] = "
                f"'{extracted}' != '{entity['text']}'"
            )
    
    def test_entity_labels_are_valid(self):
        """All entity labels should be from supported set."""
        response = client.post("/extract-entities", json={
            "title": "Bank Negara Malaysia Meeting",
            "body": "Bank Negara Malaysia held its quarterly policy meeting in Kuala Lumpur."
        })
        assert response.status_code == 200
        valid_labels = {"PERSON", "ORG", "GPE", "DATE", "EVENT"}
        for entity in response.json()["entities"]:
            assert entity["label"] in valid_labels, f"Unexpected label: {entity['label']}"


class TestInvalidInput:
    """Test 2: Empty, wrong-type, and malformed JSON inputs."""
    
    def test_empty_title_returns_400(self):
        """Empty title should return HTTP 400."""
        response = client.post("/extract-entities", json={
            "title": "",
            "body": "Some article body text here."
        })
        assert response.status_code == 400
    
    def test_empty_body_returns_400(self):
        """Empty body should return HTTP 400."""
        response = client.post("/extract-entities", json={
            "title": "Some Title",
            "body": ""
        })
        assert response.status_code == 400
    
    def test_whitespace_only_title_returns_400(self):
        """Whitespace-only title should return HTTP 400."""
        response = client.post("/extract-entities", json={
            "title": "   \t\n  ",
            "body": "Some body text."
        })
        assert response.status_code == 400
    
    def test_whitespace_only_body_returns_400(self):
        """Whitespace-only body should return HTTP 400."""
        response = client.post("/extract-entities", json={
            "title": "Some Title",
            "body": "   \n\t  "
        })
        assert response.status_code == 400
    
    def test_missing_title_returns_400(self):
        """Missing title field should return HTTP 400."""
        response = client.post("/extract-entities", json={
            "body": "Some body text."
        })
        assert response.status_code == 400
    
    def test_missing_body_returns_400(self):
        """Missing body field should return HTTP 400."""
        response = client.post("/extract-entities", json={
            "title": "Some Title"
        })
        assert response.status_code == 400
    
    def test_non_string_title_returns_400(self):
        """Non-string title should return HTTP 400."""
        response = client.post("/extract-entities", json={
            "title": 12345,
            "body": "Some body text."
        })
        assert response.status_code == 400
    
    def test_non_string_body_returns_400(self):
        """Non-string body should return HTTP 400."""
        response = client.post("/extract-entities", json={
            "title": "Some Title",
            "body": ["not", "a", "string"]
        })
        assert response.status_code == 400
    
    def test_malformed_json_returns_400(self):
        """Malformed JSON should return HTTP 400."""
        response = client.post(
            "/extract-entities",
            content=b'{invalid json content',
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code in (400, 422)


class TestPunctuationAndAmbiguousNames:
    """Test 3: Articles with punctuation and ambiguous names."""
    
    def test_entity_with_punctuation(self):
        """Entities containing punctuation should be handled correctly."""
        response = client.post("/extract-entities", json={
            "title": "Dr. Ahmad's Visit to St. Petersburg",
            "body": "Dr. Ahmad bin Abdullah met with officials from the U.N. in St. Petersburg. "
                    "The meeting was organized by UNESCO's regional office."
        })
        assert response.status_code == 200
        data = response.json()
        
        # Reconstruct combined text
        combined = "Dr. Ahmad's Visit to St. Petersburg\n\n" + \
                   "Dr. Ahmad bin Abdullah met with officials from the U.N. in St. Petersburg. " + \
                   "The meeting was organized by UNESCO's regional office."
        
        for entity in data["entities"]:
            extracted = combined[entity["start"]:entity["end"]]
            assert extracted == entity["text"], f"Offset mismatch for '{entity['text']}'"
    
    def test_ambiguous_name(self):
        """Ambiguous names (could be person or place) should not crash the model."""
        response = client.post("/extract-entities", json={
            "title": "Washington Signs New Trade Deal",
            "body": "Washington announced a new trade agreement today. George Washington would be proud."
        })
        assert response.status_code == 200
        assert isinstance(response.json()["entities"], list)


class TestRepeatedEntities:
    """Test 4: Articles with repeated entities at different positions."""
    
    def test_repeated_entity_different_offsets(self):
        """Same entity appearing multiple times should have different start/end offsets."""
        response = client.post("/extract-entities", json={
            "title": "Maybank Announces Partnership",
            "body": "Maybank Group announced today that Maybank will expand its operations. "
                    "Maybank CEO confirmed the plans during a press conference at Maybank Tower."
        })
        assert response.status_code == 200
        data = response.json()
        
        # Find entities with text "Maybank" or containing "Maybank"
        maybank_entities = [e for e in data["entities"] if "Maybank" in e["text"]]
        
        if len(maybank_entities) >= 2:
            # Check that offsets are different for repeated mentions
            offsets = [(e["start"], e["end"]) for e in maybank_entities]
            # At least some should have unique offsets
            assert len(set(offsets)) >= 2, (
                "Repeated entity 'Maybank' should have different offsets for different occurrences"
            )
        
        # Verify all offsets are correct
        combined = "Maybank Announces Partnership\n\n" + \
                   "Maybank Group announced today that Maybank will expand its operations. " + \
                   "Maybank CEO confirmed the plans during a press conference at Maybank Tower."
        
        for entity in data["entities"]:
            extracted = combined[entity["start"]:entity["end"]]
            assert extracted == entity["text"]


class TestNumbersDatesUnicode:
    """Test 5: Articles with numbers, dates, and non-ASCII characters."""
    
    def test_date_entities(self):
        """Articles with date formats should be handled correctly."""
        response = client.post("/extract-entities", json={
            "title": "Q3 2026 Financial Report Released",
            "body": "The report for Q3 2026 was released on October 1, 2026. "
                    "Revenue grew 15% compared to the same period in 2025."
        })
        assert response.status_code == 200
        data = response.json()
        
        combined = "Q3 2026 Financial Report Released\n\n" + \
                   "The report for Q3 2026 was released on October 1, 2026. " + \
                   "Revenue grew 15% compared to the same period in 2025."
        
        for entity in data["entities"]:
            extracted = combined[entity["start"]:entity["end"]]
            assert extracted == entity["text"]
    
    def test_non_ascii_characters(self):
        """Non-ASCII/Unicode characters should not cause offset errors."""
        response = client.post("/extract-entities", json={
            "title": "Bénédicte Müller Joins Deutsche Bank",
            "body": "Bénédicte Müller was appointed as the new head of Deutsche Bank's "
                    "Asian operations. She will be based in São Paulo and Tokyo. "
                    "The appointment is effective from März 2027."
        })
        assert response.status_code == 200
        data = response.json()
        
        combined = "Bénédicte Müller Joins Deutsche Bank\n\n" + \
                   "Bénédicte Müller was appointed as the new head of Deutsche Bank's " + \
                   "Asian operations. She will be based in São Paulo and Tokyo. " + \
                   "The appointment is effective from März 2027."
        
        for entity in data["entities"]:
            extracted = combined[entity["start"]:entity["end"]]
            assert extracted == entity["text"], (
                f"Unicode offset mismatch: combined[{entity['start']}:{entity['end']}] = "
                f"'{extracted}' != '{entity['text']}'"
            )
    
    def test_currency_symbols(self):
        """Currency symbols and special characters should not break processing."""
        response = client.post("/extract-entities", json={
            "title": "Maybank Posts RM2.5B Profit",
            "body": "Maybank reported a profit of RM2.5 billion (€500M) for the fiscal year. "
                    "The Malaysian bank's shares rose 3% on Bursa Malaysia."
        })
        assert response.status_code == 200
        assert isinstance(response.json()["entities"], list)


class TestLongText:
    """Test 6: Text length boundary tests."""
    
    def test_long_text_within_limit(self):
        """Long text within the maximum limit should be processed successfully."""
        # Create text just under the limit
        body = "Maybank is a major bank in Malaysia. " * 1000  # ~36000 chars
        response = client.post("/extract-entities", json={
            "title": "Maybank Long Article",
            "body": body
        })
        assert response.status_code == 200
        data = response.json()
        assert "entities" in data
    
    def test_text_exceeding_limit_returns_400(self):
        """Text exceeding the maximum limit should return HTTP 400."""
        # Create text exceeding the limit
        body = "A" * (MAX_COMBINED_LENGTH + 1000)
        response = client.post("/extract-entities", json={
            "title": "Test Title",
            "body": body
        })
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data


class TestDeterminism:
    """Test that the model produces deterministic results."""
    
    def test_same_input_same_output(self):
        """Same input should always produce the same output."""
        payload = {
            "title": "Bank Negara Malaysia Policy Decision",
            "body": "Bank Negara Malaysia announced its latest interest rate decision today in Kuala Lumpur."
        }
        
        response1 = client.post("/extract-entities", json=payload)
        response2 = client.post("/extract-entities", json=payload)
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        assert response1.json() == response2.json(), "Model output is not deterministic"


class TestUIEndpoint:
    """Test that the UI endpoint serves HTML."""
    
    def test_ui_serves_html(self):
        """GET / should serve an HTML page."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        assert "NER" in response.text or "ner" in response.text.lower()


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main(["-v", __file__]))

