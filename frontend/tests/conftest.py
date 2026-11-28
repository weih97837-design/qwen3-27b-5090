"""Shared test setup for frontend/tests.

The request trace is on by default (docs/request-trace.md): a test that starts an engine or a Recorder from the process environment
must not write trace files into the home directory by accident. Every test starts with REQUEST_TRACE=0; a test that needs the trace
sets or deletes REQUEST_TRACE itself with monkeypatch, after this fixture ran.
"""

import pytest


@pytest.fixture(autouse=True)
def _request_trace_off(monkeypatch):
    monkeypatch.setenv("REQUEST_TRACE", "0")
