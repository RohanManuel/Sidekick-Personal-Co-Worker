import os
import uuid
import requests
from datetime import datetime
from dotenv import load_dotenv
from playwright.async_api import async_playwright
from langchain.agents import Tool
from langchain_community.agent_toolkits import FileManagementToolkit
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_experimental.tools import PythonREPLTool
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

load_dotenv(override=True)

# --- Playwright setup (async) ---
async def playwright_tools():
    """Start Playwright async and return custom tools."""
    os.system("playwright install chromium")
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=True)
    context = await browser.new_context()
    page = await context.new_page()

    async def screenshot_tool(url: str) -> str:
        await page.goto(url)
        path = f"sandbox/screenshot_{uuid.uuid4().hex}.png"
        await page.screenshot(path=path)
        return f"Screenshot saved to {path}"

    async def get_title_tool(url: str) -> str:
        await page.goto(url)
        title = await page.title()
        return f"Page title: {title}"

    tools = [
        Tool(
            name="take_screenshot",
            func=lambda url: asyncio.create_task(screenshot_tool(url)),
            description="Take a screenshot of a webpage. Args: url"
        ),
        Tool(
            name="get_page_title",
            func=lambda url: asyncio.create_task(get_title_tool(url)),
            description="Get the title of a webpage. Args: url"
        )
    ]

    return tools, browser, playwright

# --- Push notification via Pushover ---
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_user = os.getenv("PUSHOVER_USER")
pushover_url = "https://api.pushover.net/1/messages.json"

def push(text: str) -> str:
    if not pushover_token or not pushover_user:
        return "Pushover credentials not set."
    resp = requests.post(
        pushover_url,
        data={"token": pushover_token, "user": pushover_user, "message": text},
    )
    return "success" if resp.status_code == 200 else f"fail ({resp.text})"

# --- File management tools ---
def get_file_tools():
    toolkit = FileManagementToolkit(root_dir=os.getenv("FILE_TOOL_ROOT", "sandbox"))
    return toolkit.get_tools()

# --- PDF Generator ---
def generate_pdf(filename: str, content: str) -> str:
    root_dir = os.getenv("FILE_TOOL_ROOT", "sandbox")
    os.makedirs(root_dir, exist_ok=True)
    filepath = os.path.join(root_dir, filename if filename.endswith(".pdf") else f"{filename}.pdf")
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(filepath)
    story = [Paragraph(content, styles["Normal"]), Spacer(1, 12)]
    doc.build(story)
    return f"PDF generated: {filepath}"

def pdf_tools() -> list[Tool]:
    return [
        Tool(
            name="generate_pdf",
            func=generate_pdf,
            description="Generate a PDF file in FILE_TOOL_ROOT. Args: filename, content"
        )
    ]

# --- Search, Wikipedia, Python REPL ---
async def other_tools() -> list[Tool]:
    push_tool = Tool(
        name="send_push_notification",
        func=push,
        description="Send push notification via Pushover",
    )
    file_tools = get_file_tools()
    serper = GoogleSerperAPIWrapper()
    search_tool = Tool(
        name="search",
        func=serper.run,
        description="Google Serper web search"
    )
    wikipedia = WikipediaAPIWrapper()
    wiki_tool = WikipediaQueryRun(api_wrapper=wikipedia)
    python_repl = PythonREPLTool()
    return file_tools + [push_tool, search_tool, python_repl, wiki_tool]

# --- Google Calendar integration ---
def _get_calendar_service():
    creds_path = os.getenv("GOOGLE_TOKEN_PATH", "token.json")
    scopes = ["https://www.googleapis.com/auth/calendar"]
    if not os.path.exists(creds_path):
        raise FileNotFoundError(f"Google credentials not found at {creds_path}")
    creds = Credentials.from_authorized_user_file(creds_path, scopes=scopes)
    return build("calendar", "v3", credentials=creds)

def create_calendar_event(summary: str, start_iso: str, end_iso: str, description: str = "", calendar_id: str = None) -> str:
    cal_id = calendar_id or os.getenv("GOOGLE_CALENDAR_ID", "primary")
    service = _get_calendar_service()
    event = {
        "summary": summary,
        "description": description,
        "start": {"dateTime": start_iso},
        "end": {"dateTime": end_iso},
    }
    created = service.events().insert(calendarId=cal_id, body=event).execute()
    return f"Event created: {created.get('htmlLink')}"

def list_upcoming_events(calendar_id: str = None, max_results: int = 5) -> str:
    cal_id = calendar_id or os.getenv("GOOGLE_CALENDAR_ID", "primary")
    service = _get_calendar_service()
    now = datetime.utcnow().isoformat() + "Z"
    events_result = (
        service.events()
        .list(
            calendarId=cal_id,
            timeMin=now,
            maxResults=max_results,
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )
    events = events_result.get("items", [])
    if not events:
        return "No upcoming events found."
    lines = []
    for evt in events:
        start = evt["start"].get("dateTime", evt["start"].get("date"))
        lines.append(f"{start} — {evt['summary']}")
    return "\n".join(lines)

def calendar_tools() -> list[Tool]:
    return [
        Tool(
            name="create_calendar_event",
            func=create_calendar_event,
            description="Schedule an event: summary, start_iso (RFC3339), end_iso (RFC3339), [description], [calendar_id]",
        ),
        Tool(
            name="list_upcoming_events",
            func=list_upcoming_events,
            description="List upcoming events on the specified or primary calendar.",
        ),
    ]

# --- Assemble all tools ---
async def all_tools():
    pw_tools, browser, playwright = await playwright_tools()
    return pw_tools + await other_tools() + calendar_tools() + pdf_tools()
