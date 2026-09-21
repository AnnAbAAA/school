import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv
from google.adk.agents.llm_agent import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

KYIV = ZoneInfo("Europe/Kyiv")
NTFY_SERVER = os.getenv("NTFY_SERVER", "https://ntfy.sh").rstrip("/")
NTFY_TOPIC = os.getenv("NTFY_TOPIC", "")

SCHEDULE = {
    "mon": [
        ("09:00", "Українська мова", "415"),
        ("09:55", "Інтегрований курс «Література»", "404"),
        ("10:50", "Алгебра", "411"),
        ("11:45", "Громадянська освіта", "Лекційна, 3 поверх"),
        ("13:25", "Іноземна мова (англійська)", "401"),
        ("14:20", "SWIFT (Apple)", "308"),
        ("15:20", "SWIFT (Apple)", "308"),
    ],
    "tue": [
        ("09:00", "Хімія", "414"),
        ("09:55", "Іноземна мова (англійська)", "301"),
        ("10:50", "Фізика", "413"),
        ("11:45", "Робототехніка", "310"),
        ("13:25", "Фізична культура", "Big sports hall"),
        ("14:20", "Інтегрований курс «Література»", "414"),
        ("15:20", "Історія України", "404"),
        ("16:15", "Штучний інтелект", "309"),
    ],
    "wed": [
        ("09:00", "Біологія", "415"),
        ("09:55", "Всесвітня історія", "Лекційна, 3 поверх"),
        ("10:50", "Географія", "Медіатека"),
        ("11:45", "Українська мова", "404"),
        ("13:25", "Іноземна мова (англійська)", "306"),
        ("14:20", "Unity 3D", "310"),
        ("15:20", "IT-англійська", "Медіатека"),
    ],
    "thu": [
        ("09:00", "Геометрія", "405"),
        ("09:55", "Біологія", "415"),
        ("10:50", "Іноземна мова (англійська)", "312"),
        ("11:45", "Хмарні технології", "309"),
        ("13:25", "Project", "310"),
        ("14:20", "Мобільна розробка Flutter", "309"),
        ("15:20", "Python", "310"),
    ],
    "fri": [
        ("09:00", "Алгебра", "412"),
        ("09:55", "Фізична культура", "Big sports hall"),
        ("10:50", "Громадянська освіта", "301"),
        ("11:45", "Українська мова", "412"),
        ("13:25", "Геометрія", "412"),
        ("14:20", "Іноземна мова (англійська)", "307"),
        ("15:20", "Алгебра", "410"),
    ],
}

TRIGGERS = {
    "09:00": "08:55",  # first lesson: five minutes before
    "09:55": "09:45",
    "10:50": "10:40",
    "11:45": "11:35",
    "13:25": "12:30",
    "14:20": "14:10",
    "15:20": "15:05",
    "16:15": "16:05",
}


def get_next_lesson(weekday: str, lesson_start: str) -> str:
    """Return lesson data for a weekday and lesson start time.

    Args:
        weekday: Three-letter English weekday: mon, tue, wed, thu, or fri.
        lesson_start: Lesson start time in HH:MM format.
    """
    lesson = next(
        (item for item in SCHEDULE.get(weekday.lower(), []) if item[0] == lesson_start),
        None,
    )
    if lesson is None:
        return json.dumps({"found": False}, ensure_ascii=False)
    start, subject, room = lesson
    return json.dumps(
        {"found": True, "start": start, "subject": subject, "room": room},
        ensure_ascii=False,
    )


def send_push_notification(title: str, message: str) -> str:
    """Send a native push notification to Ann's iPhone through ntfy.

    Args:
        title: Short notification title.
        message: Notification body containing lesson time, subject, and room.
    """
    if not NTFY_TOPIC:
        return "Error: NTFY_TOPIC is not configured"
    response = httpx.post(
        NTFY_SERVER,
        json={
            "topic": NTFY_TOPIC,
            "title": title,
            "message": message,
            "priority": 4,
            "tags": ["school", "bell"],
        },
        timeout=15,
    )
    response.raise_for_status()
    return "Notification sent successfully"


root_agent = Agent(
    name="school_executive_assistant",
    model=os.getenv("GEMINI_MODEL", "gemini-flash-latest"),
    description="Ann's school executive assistant that sends iPhone reminders.",
    instruction=(
        "You are Ann's school executive assistant. You receive a weekday and the "
        "start time of the upcoming lesson. First call get_next_lesson. If found is "
        "false, do nothing. Otherwise call send_push_notification exactly once. "
        "Use title 'Следующий урок через перемену' and a concise Russian message in "
        "this exact structure: '<subject> — <start>\\nКабинет: <room>'. Never invent "
        "or change the lesson, time, or room."
    ),
    tools=[get_next_lesson, send_push_notification],
)

APP_NAME = "school_executive_assistant"
USER_ID = "ann"
session_service = InMemorySessionService()
runner = Runner(
    app_name=APP_NAME,
    agent=root_agent,
    session_service=session_service,
)


async def run_agent(prompt: str) -> str:
    session_id = str(uuid.uuid4())
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=session_id,
    )
    message = types.Content(
        role="user",
        parts=[types.Part.from_text(text=prompt)],
    )
    final_text = ""
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session_id,
        new_message=message,
    ):
        if event.is_final_response() and event.content and event.content.parts:
            final_text = "".join(part.text or "" for part in event.content.parts)
    return final_text


async def wake_agent(weekday: str, lesson_start: str) -> None:
    prompt = (
        f"Today is {weekday}. The upcoming lesson starts at {lesson_start}. "
        "Check it and send Ann the reminder now."
    )
    result = await run_agent(prompt)
    logging.info("Agent completed: %s", result)


def build_scheduler() -> AsyncIOScheduler:
    scheduler = AsyncIOScheduler(timezone=KYIV)
    for weekday, lessons in SCHEDULE.items():
        for lesson_start, _, _ in lessons:
            notification_time = TRIGGERS[lesson_start]
            hour, minute = map(int, notification_time.split(":"))
            scheduler.add_job(
                wake_agent,
                CronTrigger(
                    day_of_week=weekday,
                    hour=hour,
                    minute=minute,
                    timezone=KYIV,
                ),
                args=[weekday, lesson_start],
                id=f"{weekday}-{lesson_start}",
                replace_existing=True,
                misfire_grace_time=120,
            )
    return scheduler


async def test_notification() -> None:
    now = datetime.now(KYIV)
    result = await run_agent(
        "This is a setup test. Call send_push_notification exactly once with title "
        "'Агент подключён' and message 'Тестовое уведомление работает ✅'.",
    )
    logging.info("Test at %s: %s", now.isoformat(), result)


async def main() -> None:
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY is missing in .env")
    if not NTFY_TOPIC:
        raise RuntimeError("NTFY_TOPIC is missing in .env")

    if "--test" in os.sys.argv:
        await test_notification()
        return

    scheduler = build_scheduler()
    scheduler.start()
    logging.info("School agent is running in timezone Europe/Kyiv")
    try:
        await asyncio.Event().wait()
    finally:
        scheduler.shutdown(wait=False)


if __name__ == "__main__":
    asyncio.run(main())
