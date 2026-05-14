"""
tests/test_tools.py
--------------------
Unit tests for BrowserToolkit tools.

Run with:
    pytest tests/ -v

Note: These tests require Playwright + Chromium installed.
    playwright install chromium
"""

import pytest
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from tools.browser_tools import BrowserToolkit


# ─────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
async def toolkit():
    """Shared browser toolkit for all tests."""
    os.environ["BROWSER_HEADLESS"] = "true"
    tk = BrowserToolkit()
    await tk.start()
    yield tk
    await tk.close()


# ─────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_navigate_to(toolkit):
    """Test basic navigation."""
    result = await toolkit.navigate_to("https://example.com")
    assert result["success"] is True
    assert "example" in result["result"].lower()


@pytest.mark.asyncio
async def test_navigate_adds_https(toolkit):
    """Test that missing https:// is added automatically."""
    result = await toolkit.navigate_to("example.com")
    assert result["success"] is True


@pytest.mark.asyncio
async def test_extract_content(toolkit):
    """Test text extraction from a page."""
    await toolkit.navigate_to("https://example.com")
    result = await toolkit.extract_content("main content")
    assert result["success"] is True
    assert len(result["result"]) > 50  # Should have meaningful content


@pytest.mark.asyncio
async def test_get_all_links(toolkit):
    """Test link extraction."""
    await toolkit.navigate_to("https://example.com")
    result = await toolkit.get_all_links()
    assert result["success"] is True
    assert "link" in result["result"].lower() or "Found" in result["result"]


@pytest.mark.asyncio
async def test_scroll_page(toolkit):
    """Test page scrolling."""
    await toolkit.navigate_to("https://example.com")
    result = await toolkit.scroll_page("down", 200)
    assert result["success"] is True
    assert "down" in result["result"].lower()


@pytest.mark.asyncio
async def test_take_screenshot(toolkit):
    """Test screenshot capture."""
    await toolkit.navigate_to("https://example.com")
    result = await toolkit.take_screenshot("test")
    assert result["success"] is True
    assert result["screenshot"] is not None
    assert os.path.exists(result["screenshot"])


@pytest.mark.asyncio
async def test_search_on_page(toolkit):
    """Test in-page search."""
    await toolkit.navigate_to("https://example.com")
    result = await toolkit.search_on_page("Example")
    assert result["success"] is True


@pytest.mark.asyncio
async def test_unknown_tool(toolkit):
    """Test that unknown tool names are handled gracefully."""
    result = await toolkit.execute_tool("nonexistent_tool", {})
    assert result["success"] is False
    assert "Unknown tool" in result["result"]


@pytest.mark.asyncio
async def test_tool_descriptions(toolkit):
    """Test that tool descriptions are returned."""
    descriptions = toolkit.get_tool_descriptions()
    assert "navigate_to" in descriptions
    assert "google_search" in descriptions
    assert "extract_content" in descriptions


def test_tool_descriptions_sync():
    """Test tool descriptions don't require browser."""
    tk = BrowserToolkit()
    descriptions = tk.get_tool_descriptions()
    assert len(descriptions) > 100
