"""
tools/browser_tools.py
-----------------------
All Playwright-based browser tool functions used by the agent executor.

Each tool:
- Accepts a running Playwright Page object + parameters
- Returns a dict with {"success": bool, "result": str, "screenshot": str | None}
- Is named and described for LLM tool-calling

Usage:
    from tools.browser_tools import BrowserToolkit
    toolkit = BrowserToolkit()
    await toolkit.start()
    result = await toolkit.navigate_to("https://google.com")
    await toolkit.close()
"""

import os
import base64
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional

from playwright.async_api import async_playwright, Page, Browser, BrowserContext


SCREENSHOT_DIR = os.getenv("SCREENSHOT_DIR", "screenshots")
BROWSER_HEADLESS = os.getenv("BROWSER_HEADLESS", "false").lower() == "true"
BROWSER_WIDTH = int(os.getenv("BROWSER_WIDTH", "1280"))
BROWSER_HEIGHT = int(os.getenv("BROWSER_HEIGHT", "800"))
BROWSER_TIMEOUT = int(os.getenv("BROWSER_TIMEOUT", "30000"))


class BrowserToolkit:
    """
    Wraps Playwright browser lifecycle and exposes named browser tools
    that the LangGraph agent can call as steps.
    """

    def __init__(self):
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._screenshot_dir = Path(SCREENSHOT_DIR)
        self._screenshot_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────

    async def start(self):
        """Launch Chromium browser and open a blank page."""
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=BROWSER_HEADLESS,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        self._context = await self._browser.new_context(
            viewport={"width": BROWSER_WIDTH, "height": BROWSER_HEIGHT},
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        )
        self._page = await self._context.new_page()
        self._page.set_default_timeout(BROWSER_TIMEOUT)

    async def close(self):
        """Gracefully close browser and Playwright."""
        if self._page:
            await self._page.close()
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()

    # ─────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────

    async def _screenshot(self, label: str = "step") -> str:
        """Take a screenshot and save it. Returns the file path."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = self._screenshot_dir / f"{label}_{timestamp}.png"
        await self._page.screenshot(path=str(path), full_page=False)
        return str(path)

    async def _get_page_info(self) -> dict:
        """Return current URL and page title."""
        return {
            "url": self._page.url,
            "title": await self._page.title(),
        }

    def _ok(self, result: str, screenshot: Optional[str] = None, **extra) -> dict:
        return {"success": True, "result": result, "screenshot": screenshot, **extra}

    def _err(self, error: str, screenshot: Optional[str] = None) -> dict:
        return {"success": False, "result": f"ERROR: {error}", "screenshot": screenshot}

    # ─────────────────────────────────────────────
    # Browser Tools
    # ─────────────────────────────────────────────

    async def navigate_to(self, url: str) -> dict:
        """
        Tool: navigate_to
        Navigate the browser to a given URL.

        Args:
            url: Full URL (e.g., "https://google.com")
        """
        try:
            if not url.startswith("http"):
                url = "https://" + url
            await self._page.goto(url, wait_until="domcontentloaded")
            info = await self._get_page_info()
            shot = await self._screenshot("navigate")
            return self._ok(
                f"Navigated to {info['url']} — Page title: '{info['title']}'",
                shot,
                **info,
            )
        except Exception as e:
            return self._err(str(e))

    async def click_element(self, selector: str) -> dict:
        """
        Tool: click_element
        Click on a page element identified by a CSS selector or text.

        Args:
            selector: CSS selector, XPath, or visible text (e.g., "button:has-text('Search')")
        """
        try:
            await self._page.click(selector, timeout=10000)
            await self._page.wait_for_load_state("domcontentloaded")
            shot = await self._screenshot("click")
            return self._ok(f"Clicked element: {selector}", shot)
        except Exception as e:
            return self._err(f"Could not click '{selector}': {e}")

    async def type_text(self, selector: str, text: str, press_enter: bool = False) -> dict:
        """
        Tool: type_text
        Type text into an input field identified by a CSS selector.

        Args:
            selector: CSS selector for the input field
            text: Text to type
            press_enter: Whether to press Enter after typing
        """
        try:
            await self._page.fill(selector, text)
            if press_enter:
                await self._page.keyboard.press("Enter")
                await self._page.wait_for_load_state("domcontentloaded")
            shot = await self._screenshot("type")
            entered = " + Enter" if press_enter else ""
            return self._ok(f"Typed '{text}' into {selector}{entered}", shot)
        except Exception as e:
            return self._err(f"Could not type into '{selector}': {e}")

    async def extract_content(self, instruction: str = "main content") -> dict:
        """
        Tool: extract_content
        Extract visible text content from the current page.

        Args:
            instruction: What to extract (e.g., "article title and summary", "all links", "table data")
        """
        try:
            # Get page text, clean up whitespace
            content = await self._page.evaluate("""() => {
                // Remove scripts and styles
                const remove = document.querySelectorAll('script, style, nav, footer, header');
                remove.forEach(el => el.remove());
                return document.body.innerText;
            }""")
            # Truncate to avoid overwhelming the LLM
            content = content.strip()
            if len(content) > 6000:
                content = content[:6000] + "\n... [truncated]"
            info = await self._get_page_info()
            shot = await self._screenshot("extract")
            return self._ok(
                f"Extracted content from '{info['title']}' ({info['url']}):\n\n{content}",
                shot,
            )
        except Exception as e:
            return self._err(f"Could not extract content: {e}")

    async def search_on_page(self, query: str) -> dict:
        """
        Tool: search_on_page
        Use Ctrl+F to search for text on the current page (highlights occurrences).

        Args:
            query: Text to search for
        """
        try:
            results = await self._page.evaluate(f"""() => {{
                const text = "{query}";
                const body = document.body.innerText;
                const matches = (body.match(new RegExp(text, 'gi')) || []).length;
                return matches;
            }}""")
            shot = await self._screenshot("search")
            return self._ok(
                f"Found {results} occurrence(s) of '{query}' on this page.", shot
            )
        except Exception as e:
            return self._err(str(e))

    async def scroll_page(self, direction: str = "down", amount: int = 500) -> dict:
        """
        Tool: scroll_page
        Scroll the page up or down.

        Args:
            direction: "up" or "down"
            amount: Pixels to scroll (default 500)
        """
        try:
            pixels = amount if direction == "down" else -amount
            await self._page.evaluate(f"window.scrollBy(0, {pixels})")
            await asyncio.sleep(0.5)
            shot = await self._screenshot("scroll")
            return self._ok(f"Scrolled {direction} by {amount}px", shot)
        except Exception as e:
            return self._err(str(e))

    async def wait_for_element(self, selector: str, timeout: int = 10000) -> dict:
        """
        Tool: wait_for_element
        Wait for an element to appear on the page.

        Args:
            selector: CSS selector of the element to wait for
            timeout: Max wait time in milliseconds
        """
        try:
            await self._page.wait_for_selector(selector, timeout=timeout)
            shot = await self._screenshot("wait")
            return self._ok(f"Element '{selector}' found on page.", shot)
        except Exception as e:
            return self._err(f"Element '{selector}' not found within timeout: {e}")

    async def get_all_links(self) -> dict:
        """
        Tool: get_all_links
        Extract all hyperlinks from the current page.
        """
        try:
            links = await self._page.evaluate("""() => {
                return Array.from(document.querySelectorAll('a[href]'))
                    .map(a => ({ text: a.innerText.trim(), href: a.href }))
                    .filter(l => l.text && l.href.startsWith('http'))
                    .slice(0, 30);
            }""")
            formatted = "\n".join(
                [f"- {l['text']}: {l['href']}" for l in links]
            )
            shot = await self._screenshot("links")
            return self._ok(f"Found {len(links)} links:\n{formatted}", shot)
        except Exception as e:
            return self._err(str(e))

    async def google_search(self, query: str) -> dict:
        """
        Tool: google_search
        Navigate to Google and perform a search query.

        Args:
            query: Search query string
        """
        try:
            await self._page.goto("https://www.google.com", wait_until="domcontentloaded")
            await asyncio.sleep(1)

            # Handle cookie consent if present
            try:
                await self._page.click("button:has-text('Accept all')", timeout=3000)
                await asyncio.sleep(0.5)
            except Exception:
                pass

            # Type into search box
            await self._page.fill("textarea[name='q'], input[name='q']", query)
            await self._page.keyboard.press("Enter")
            await self._page.wait_for_load_state("domcontentloaded")
            await asyncio.sleep(1)

            # Extract search result titles and URLs
            results = await self._page.evaluate("""() => {
                const items = document.querySelectorAll('h3');
                return Array.from(items).slice(0, 10).map(h => ({
                    title: h.innerText,
                    url: h.closest('a') ? h.closest('a').href : ''
                })).filter(r => r.title && r.url);
            }""")

            formatted = "\n".join(
                [f"{i+1}. {r['title']}\n   {r['url']}" for i, r in enumerate(results)]
            )
            shot = await self._screenshot("google_search")
            return self._ok(
                f"Google search results for '{query}':\n\n{formatted}", shot
            )
        except Exception as e:
            return self._err(f"Google search failed: {e}")

    async def take_screenshot(self, label: str = "manual") -> dict:
        """
        Tool: take_screenshot
        Take a screenshot of the current page state.

        Args:
            label: Label for the screenshot filename
        """
        try:
            shot = await self._screenshot(label)
            info = await self._get_page_info()
            return self._ok(
                f"Screenshot taken of '{info['title']}' ({info['url']})", shot
            )
        except Exception as e:
            return self._err(str(e))

    # ─────────────────────────────────────────────
    # Tool Registry (for the agent)
    # ─────────────────────────────────────────────

    def get_tool_descriptions(self) -> str:
        """Return a formatted string describing all available tools for the LLM."""
        return """
Available browser tools:
1. navigate_to(url) — Go to a URL
2. google_search(query) — Search Google for a query
3. click_element(selector) — Click a page element by CSS selector or text
4. type_text(selector, text, press_enter=False) — Type into an input field
5. extract_content(instruction) — Extract visible text from the current page
6. scroll_page(direction, amount) — Scroll up or down
7. wait_for_element(selector, timeout) — Wait for an element to appear
8. get_all_links() — Get all hyperlinks on the current page
9. search_on_page(query) — Search for text occurrences on the page
10. take_screenshot(label) — Take a screenshot of the current state
"""

    async def execute_tool(self, tool_name: str, params: dict) -> dict:
        """
        Dispatch a tool call by name.

        Args:
            tool_name: One of the tool names above
            params: Dictionary of parameters for the tool
        """
        tools_map = {
            "navigate_to": self.navigate_to,
            "google_search": self.google_search,
            "click_element": self.click_element,
            "type_text": self.type_text,
            "extract_content": self.extract_content,
            "scroll_page": self.scroll_page,
            "wait_for_element": self.wait_for_element,
            "get_all_links": self.get_all_links,
            "search_on_page": self.search_on_page,
            "take_screenshot": self.take_screenshot,
        }

        tool_fn = tools_map.get(tool_name)
        if not tool_fn:
            return {"success": False, "result": f"Unknown tool: {tool_name}", "screenshot": None}

        try:
            return await tool_fn(**params)
        except TypeError as e:
            return {"success": False, "result": f"Invalid params for {tool_name}: {e}", "screenshot": None}
