"""Tests for RamDiary (memory/diary.py)."""

import os
import tempfile

import pytest

from thermodynamic_agency.memory.diary import RamDiary, DiaryEntry


@pytest.fixture
def diary(tmp_path):
    db_path = str(tmp_path / "test_diary.db")
    d = RamDiary(path=db_path)
    yield d
    d.close()


class TestRamDiary:
    def test_append_and_recent(self, diary):
        diary.append(DiaryEntry(tick=1, role="thought", content="hello"))
        diary.append(DiaryEntry(tick=2, role="action", content="world"))
        entries = diary.recent(10)
        assert len(entries) == 2
        assert entries[0].content == "hello"
        assert entries[1].content == "world"

    def test_entry_count(self, diary):
        assert diary.entry_count() == 0
        diary.append(DiaryEntry(tick=1, role="thought", content="test"))
        assert diary.entry_count() == 1

    def test_all_entries(self, diary):
        for i in range(5):
            diary.append(DiaryEntry(tick=i, role="thought", content=f"entry {i}"))
        entries = diary.all_entries()
        assert len(entries) == 5

    def test_add_and_get_insights(self, diary):
        diary.add_insight("key insight here", source_from=1, source_to=5)
        insights = diary.insights()
        assert len(insights) == 1
        assert insights[0]["content"] == "key insight here"

    def test_wipe_removes_all(self, diary):
        diary.append(DiaryEntry(tick=1, role="thought", content="to be wiped"))
        diary.add_insight("insight", source_from=1, source_to=1)
        diary.wipe()
        assert diary.entry_count() == 0
        assert diary.insights() == []

    def test_metadata_roundtrip(self, diary):
        meta = {"mask": "Healer", "efe": 3.14}
        diary.append(DiaryEntry(tick=1, role="thought", content="meta test", metadata=meta))
        entries = diary.recent(1)
        assert entries[0].metadata == meta

    def test_recent_respects_limit(self, diary):
        for i in range(20):
            diary.append(DiaryEntry(tick=i, role="thought", content=f"e{i}"))
        recent = diary.recent(5)
        assert len(recent) == 5
        # Should be the last 5
        assert recent[-1].content == "e19"
