from unittest.mock import MagicMock
import json
import pytest
from loom.providers.google import GoogleBatchProvider

def test_google_batch_download_inlined_with_custom_id():
    provider = GoogleBatchProvider(api_key="fake")
    provider.client = MagicMock()
    
    mock_job = MagicMock()
    mock_entry_1 = MagicMock()
    mock_entry_1.metadata = {"custom_id": "cid-1"}
    mock_entry_1.response.candidates = [MagicMock()]
    mock_entry_1.response.candidates[0].content.parts = [MagicMock(text="Ans 1")]
    
    mock_entry_2 = MagicMock()
    mock_entry_2.metadata = {"custom_id": "cid-2"}
    mock_entry_2.response.candidates = [MagicMock()]
    mock_entry_2.response.candidates[0].content.parts = [MagicMock(text="Ans 2")]
    
    mock_job.dest.inlined_responses = [mock_entry_1, mock_entry_2]
    provider._retrieve = MagicMock(return_value=mock_job)
    
    id_map = {
        "cid-1": "0",
        "cid-2": "1",
    }
    
    results = provider.download_results("batch-123", id_map=id_map)
    assert results == (
        {
            "cid-1": "Ans 1",
            "cid-2": "Ans 2",
        },
        {},
    )

def test_google_batch_download_inlined_out_of_order():
    """Responses must be matched by metadata, not list position."""
    provider = GoogleBatchProvider(api_key="fake")
    provider.client = MagicMock()

    mock_job = MagicMock()
    mock_entry_1 = MagicMock()
    mock_entry_1.metadata = {"custom_id": "cid-2"}
    mock_entry_1.response.candidates = [MagicMock()]
    mock_entry_1.response.candidates[0].content.parts = [MagicMock(text="Ans 2")]

    mock_entry_2 = MagicMock()
    mock_entry_2.metadata = {"custom_id": "cid-1"}
    mock_entry_2.response.candidates = [MagicMock()]
    mock_entry_2.response.candidates[0].content.parts = [MagicMock(text="Ans 1")]

    mock_job.dest.inlined_responses = [mock_entry_1, mock_entry_2]
    provider._retrieve = MagicMock(return_value=mock_job)

    results = provider.download_results("batch-123")
    assert results == (
        {
            "cid-1": "Ans 1",
            "cid-2": "Ans 2",
        },
        {},
    )


def test_google_batch_download_inlined_raises_without_metadata():
    provider = GoogleBatchProvider(api_key="fake")
    provider.client = MagicMock()

    mock_job = MagicMock()
    mock_entry = MagicMock()
    mock_entry.metadata = None
    mock_entry.response.candidates = [MagicMock()]
    mock_entry.response.candidates[0].content.parts = [MagicMock(text="Ans 1")]

    mock_job.dest.inlined_responses = [mock_entry]
    provider._retrieve = MagicMock(return_value=mock_job)

    with pytest.raises(ValueError, match="without metadata"):
        provider.download_results("batch-123")

def test_google_batch_download_file_requires_metadata():
    provider = GoogleBatchProvider(api_key="fake")
    provider.client = MagicMock()

    mock_job = MagicMock()
    mock_job.dest.inlined_responses = None
    mock_job.dest.file_name = "output_file.jsonl"

    line1 = {"response": {"candidates": [{"content": {"parts": [{"text": "Ans 1"}]}}]}}
    line2 = {"response": {"candidates": [{"content": {"parts": [{"text": "Ans 2"}]}}]}}
    jsonl_bytes = (json.dumps(line1) + "\n" + json.dumps(line2) + "\n").encode("utf-8")

    provider.client.files.download = MagicMock(return_value=jsonl_bytes)
    provider._retrieve = MagicMock(return_value=mock_job)

    with pytest.raises(ValueError, match="without metadata"):
        provider.download_results("batch-123")


def test_google_batch_download_file_out_of_order():
    provider = GoogleBatchProvider(api_key="fake")
    provider.client = MagicMock()

    mock_job = MagicMock()
    mock_job.dest.inlined_responses = None
    mock_job.dest.file_name = "output_file.jsonl"

    line1 = {
        "metadata": {"custom_id": "cid-2"},
        "response": {"candidates": [{"content": {"parts": [{"text": "Ans 2"}]}}]},
    }
    line2 = {
        "metadata": {"custom_id": "cid-1"},
        "response": {"candidates": [{"content": {"parts": [{"text": "Ans 1"}]}}]},
    }
    jsonl_bytes = (json.dumps(line1) + "\n" + json.dumps(line2) + "\n").encode("utf-8")

    provider.client.files.download = MagicMock(return_value=jsonl_bytes)
    provider._retrieve = MagicMock(return_value=mock_job)

    results = provider.download_results("batch-123")
    assert results == (
        {
            "cid-1": "Ans 1",
            "cid-2": "Ans 2",
        },
        {},
    )
