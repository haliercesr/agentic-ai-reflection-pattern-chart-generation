import base64
import os
import re
import traceback
import uuid

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENCODE_API_KEY = os.getenv("OPENCODE_API_KEY")
OPENCODE_BASE_URL = (
    os.getenv("OPENCODE_BASE_URL") or "https://opencode.ai/inference/go/openai/v1"
)
OPENCODE_ANTHROPIC_BASE_URL = (
    os.getenv("OPENCODE_ANTHROPIC_BASE_URL") or "https://opencode.ai/inference/go"
)
OPENCODE_SESSION = os.getenv("OPENCODE_SESSION") or f"ses_{uuid.uuid4().hex}"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

GENERATION_MODEL = os.getenv("GENERATION_MODEL")
REFLECTION_MODEL = os.getenv("REFLECTION_MODEL")

MODELS = [
    m.strip()
    for m in (os.getenv("MODELS") or "glm-5.3-flash,gemini-4-argon").split(",")
    if m.strip()
]

if not MODELS:
    MODELS = ["glm-5.3-flash", "gemini-4-argon"]

GENERATION_MODEL = GENERATION_MODEL or MODELS[0]
REFLECTION_MODEL = REFLECTION_MODEL or (MODELS[1] if len(MODELS) > 1 else MODELS[0])

if OPENCODE_API_KEY:
    client = OpenAI(
        api_key=OPENCODE_API_KEY,
        base_url=OPENCODE_BASE_URL,
        default_headers={"x-opencode-session": OPENCODE_SESSION},
    )
elif OPENAI_API_KEY:
    _client_kwargs = {"api_key": OPENAI_API_KEY}
    if OPENAI_BASE_URL:
        _client_kwargs["base_url"] = OPENAI_BASE_URL
    client = OpenAI(**_client_kwargs)
else:
    raise RuntimeError(
        "Set OPENCODE_API_KEY (preferred) or OPENAI_API_KEY in .env. "
        "Copy .env.example to .env and fill in your key."
    )

_COLUMN_ALIASES = {
    "money": "price",
    "coffee": "coffee_name",
    "coffee_type": "coffee_name",
    "datetime": "date",
    "hour": "time",
}

_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


def get_response(model: str, prompt: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def load_and_prepare_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    for old, new in _COLUMN_ALIASES.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    if "date" not in df.columns:
        raise ValueError(f"'date' column not found. Columns: {list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).reset_index(drop=True)

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter

    if "time" in df.columns:
        df["time"] = df["time"].astype(str)

    return df


def print_html(content, title: str = "", is_image: bool = False) -> None:
    from html import escape

    from IPython.display import HTML, display

    if is_image:
        media_type, b64 = encode_image_b64(content)
        body = f'<img src="data:{media_type};base64,{b64}" />'
    else:
        body = f"<pre>{escape(str(content))}</pre>"

    header = f"<h3>{escape(title)}</h3>" if title else ""
    display(HTML(header + body))


def encode_image_b64(path: str) -> tuple[str, str]:
    ext = os.path.splitext(path)[1].lower()
    media_type = _MEDIA_TYPES.get(ext, "image/png")
    with open(path, "rb") as f:
        return media_type, base64.b64encode(f.read()).decode("utf-8")


def image_openai_call(model: str, prompt: str, media_type: str, b64: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{b64}"},
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content


def image_anthropic_call(model: str, prompt: str, media_type: str, b64: str) -> str:
    import anthropic

    if OPENCODE_API_KEY:
        anthropic_client = anthropic.Anthropic(
            api_key=OPENCODE_API_KEY,
            base_url=OPENCODE_ANTHROPIC_BASE_URL,
            default_headers={"x-opencode-session": OPENCODE_SESSION},
        )
    elif ANTHROPIC_API_KEY:
        anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    else:
        raise RuntimeError("Set OPENCODE_API_KEY or ANTHROPIC_API_KEY in .env")

    message = anthropic_client.messages.create(
        model=model,
        max_tokens=4096,
        system="You are a data visualization expert.",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    return "".join(block.text for block in message.content if block.type == "text")


def ensure_execute_python_tags(code: str) -> str:
    return f"<execute_python>\n{code.strip()}\n</execute_python>"


def _extract_execute_python(text: str) -> str:
    match = re.search(r"<execute_python>([\s\S]*?)</execute_python>", text)
    return match.group(1).strip() if match else ""


def run_generated_code(
    code: str, df: pd.DataFrame, model: str, max_retries: int = 1
) -> None:
    for attempt in range(max_retries + 1):
        try:
            exec(compile(code, "<generated_chart>", "exec"), {"df": df.copy()})
            return
        except Exception:
            if attempt >= max_retries:
                raise
            error = traceback.format_exc()
            print(error)
            code = _extract_execute_python(
                get_response(
                    model,
                    f"The code below failed:\n\n{code}\n\nError:\n{error}\n\n"
                    "Return only the corrected code inside <execute_python> tags.",
                )
            ) or code
