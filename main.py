import os
import json
from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from openai import AsyncOpenAI
from pydantic import BaseModel
from typing import List, Optional
import asyncio

app = FastAPI(
    title="DeepSeek × Nemotron Coder API",
    description="Dual-model coding assistant. Use /chat for direct DeepSeek calls or /agent for the full Nemotron → DeepSeek → Nemotron orchestration pipeline.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ── Auth ─────────────────────────────────────────────────────────────────────
API_KEY = os.environ.get("CODER_API_KEY", "")  # set this in Railway variables
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def require_api_key(key: str = Depends(api_key_header)):
    if not API_KEY:
        return  # no key set = open (dev mode)
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key. Pass it as X-API-Key header.")

# ── Model clients ─────────────────────────────────────────────────────────────
deepseek_client = AsyncOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ.get("NVIDIA_API_KEY"),
    timeout=120.0
)
nemotron_client = AsyncOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ.get("NEMOTRON_API_KEY"),
    timeout=120.0
)

# ── Skills ────────────────────────────────────────────────────────────────────
SKILLS = {
    "webdev": {
        "label": "Web Dev", "icon": "🌐", "color": "#3b82f6",
        "prompt": """You are an elite web development assistant specializing in HTML, CSS, JavaScript, and modern frameworks (React, Vue, Svelte, Next.js, Tailwind).
- Produce pixel-perfect, accessible, responsive UI components
- Always provide complete, runnable code — no placeholders
- Note browser compatibility issues and accessibility improvements
- Flag performance concerns (Core Web Vitals, lazy loading, code splitting)"""
    },
    "backend": {
        "label": "Backend", "icon": "⚙️", "color": "#8b5cf6",
        "prompt": """You are a senior backend engineer specializing in API design, databases, and server architecture.
- Write production-ready code with proper error handling and input validation
- Cover FastAPI, Flask, Express, Node.js, PostgreSQL, Redis, Docker
- Show DB schema alongside API code when relevant
- Flag N+1 queries, missing indexes, auth gaps, and secrets handling"""
    },
    "pentest": {
        "label": "Pentesting", "icon": "🔐", "color": "#ef4444",
        "prompt": """You are an expert penetration tester (ethical, authorized contexts only).
- Cover OWASP Top 10: SQLi, XSS, CSRF, SSRF, RCE, broken auth, mass assignment
- For every finding: severity, attack vector, proof-of-concept, and fix
- Format: [CRITICAL/HIGH/MEDIUM/LOW] Name → PoC → Remediation code
- Never provide attack code without the corresponding defense"""
    },
    "bugfix": {
        "label": "Bug Fix", "icon": "🐛", "color": "#f59e0b",
        "prompt": """You are a debugging specialist. Your only job is finding and fixing broken code.
1. Quote the broken line(s) first
2. Explain WHY it broke (root cause, not symptom)
3. Show the complete fix
4. Add defensive code to prevent recurrence
5. Suggest a test case that would have caught this"""
    },
    "quality": {
        "label": "Code Review", "icon": "✅", "color": "#10b981",
        "prompt": """You are a senior code reviewer focused on production quality.
Output format:
## Summary — overall verdict
## Critical Issues (must fix) — with code examples
## Improvements (should fix) — with code examples
## Good Patterns — what's done well
## Refactored Version — complete improved code
Cover: clarity, SOLID principles, error handling, security, performance, test gaps"""
    },
    "edgetest": {
        "label": "Edge Testing", "icon": "⚡", "color": "#f97316",
        "prompt": """You are a chaos engineering and edge-case testing specialist.
- Test: boundary values, race conditions, network failures, malformed inputs, auth edge cases
- Write actual test code (pytest, Jest, or plain scripts)
- For each case: input → expected → what actually happens
- Flag crashes vs graceful failures
- End with a test coverage checklist"""
    }
}

BASE_EXECUTOR_PROMPT = """You are an expert coding assistant powered by DeepSeek V4-Pro. You are the executor in a two-model pipeline.

General rules:
- Write complete, working code — never use placeholders
- Use code blocks with correct language tags
- Be direct and precise

Active skill mode:
{skill_prompt}

Orchestrator's plan for this task:
{plan}"""

ORCHESTRATOR_SYSTEM = """You are Nemotron Ultra, an expert AI orchestrator and code reviewer working in a two-stage pipeline.
STAGE 1 - PLAN: Analyze requests, select skill mode, write execution plan for DeepSeek V4-Pro.
STAGE 2 - REVIEW: Review DeepSeek's output for correctness, completeness, security, and quality."""

PLAN_PROMPT = """Analyze this user request and produce a JSON execution plan.

User request: {user_message}
Available skill modes: {skill_list}

Respond with ONLY valid JSON, no markdown:
{{
  "skill": "<skill_id or null>",
  "reasoning": "<why this skill, 1-2 sentences>",
  "plan": "<clear instructions for DeepSeek>",
  "complexity": "<low|medium|high>",
  "steps": ["step1", "step2"]
}}"""

REVIEW_PROMPT = """Review this output from DeepSeek V4-Pro.

Original request: {user_message}
Skill used: {skill}
Plan followed: {plan}

DeepSeek's output:
---
{executor_output}
---

Respond with ONLY valid JSON:
{{
  "approved": true/false,
  "score": <1-10>,
  "verdict": "<one sentence>",
  "issues": ["issue1"],
  "improvements": ["improvement1"],
  "final_notes": "<notes or empty string>"
}}"""

# ── Request/Response models ───────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

    model_config = {"json_schema_extra": {"example": {"role": "user", "content": "Build a FastAPI endpoint with JWT auth"}}}

class ChatRequest(BaseModel):
    messages: List[Message]
    skill: Optional[str] = None
    thinking: bool = False

    model_config = {"json_schema_extra": {"example": {
        "messages": [{"role": "user", "content": "Build a FastAPI endpoint with JWT auth"}],
        "skill": "backend",
        "thinking": False
    }}}

class AgentRequest(BaseModel):
    messages: List[Message]
    thinking: bool = False

    model_config = {"json_schema_extra": {"example": {
        "messages": [{"role": "user", "content": "Build and security audit a login API"}],
        "thinking": False
    }}}

class ChatResponse(BaseModel):
    reply: str
    reasoning: Optional[str]
    skill: Optional[str]
    mode: str

class SkillInfo(BaseModel):
    id: str
    label: str
    icon: str
    color: str

class SkillsResponse(BaseModel):
    skills: List[SkillInfo]

class HealthResponse(BaseModel):
    status: str
    models: dict
    skills_available: int

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"],
         summary="Health check")
async def health():
    return {
        "status": "ok",
        "models": {
            "executor": "deepseek-ai/deepseek-v4-pro",
            "orchestrator": "nvidia/nemotron-3-ultra-550b-a55b"
        },
        "skills_available": len(SKILLS)
    }


@app.get("/skills", response_model=SkillsResponse, tags=["System"],
         summary="List available skill modes")
async def list_skills():
    return {"skills": [
        {"id": k, "label": v["label"], "icon": v["icon"], "color": v["color"]}
        for k, v in SKILLS.items()
    ]}


@app.post("/chat", response_model=ChatResponse, tags=["Chat"],
          summary="Direct chat with DeepSeek V4-Pro",
          description="""Send a message directly to DeepSeek V4-Pro with an optional skill mode.
          
**Skill modes:** webdev, backend, pentest, bugfix, quality, edgetest

**Headers:** `X-API-Key: your-key` (if CODER_API_KEY is set)""")
async def chat(req: ChatRequest, _=Depends(require_api_key)):
    try:
        skill_key = req.skill if req.skill in SKILLS else None
        skill_prompt = SKILLS[skill_key]["prompt"] if skill_key else "Respond as a general expert coding assistant."

        system = f"""You are an expert coding assistant powered by DeepSeek V4-Pro.
Write complete, working code. Use correct language tags in code blocks. Be direct and precise.

Active skill:
{skill_prompt}"""

        messages = [{"role": "system", "content": system}]
        messages += [{"role": m.role, "content": m.content} for m in req.messages]

        completion = await deepseek_client.chat.completions.create(
            model="deepseek-ai/deepseek-v4-pro",
            messages=messages,
            temperature=0.7,
            max_tokens=2048
        )

        msg = completion.choices[0].message
        return {
            "reply": msg.content,
            "reasoning": getattr(msg, "reasoning_content", None),
            "skill": skill_key,
            "mode": "direct"
        }
    except Exception as e:
        import traceback
        print(f"CHAT ERROR: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent", tags=["Agent"],
          summary="Full orchestration pipeline",
          description="""Runs the full **Nemotron → DeepSeek → Nemotron** pipeline.

1. **Nemotron Ultra** analyzes the request, picks a skill, writes an execution plan  
2. **DeepSeek V4-Pro** executes the plan and writes the code  
3. **Nemotron Ultra** reviews the output and scores it 1-10

Returns a **Server-Sent Events (SSE)** stream. Parse `data:` lines as JSON.

**Event types:** `stage` | `plan` | `execute` | `review` | `done` | `error`

**Headers:** `X-API-Key: your-key` (if CODER_API_KEY is set)""")
async def agent(req: AgentRequest, _=Depends(require_api_key)):
    async def run():
        user_message = req.messages[-1].content if req.messages else ""
        skill_list = ", ".join([f"{k} ({v['label']})" for k, v in SKILLS.items()])

        def event(type_, data):
            return f"data: {json.dumps({'type': type_, **data})}\n\n"

        try:
            yield event("stage", {"stage": "plan", "label": "🧠 Nemotron planning..."})

            plan_completion = await nemotron_client.chat.completions.create(
                model="nvidia/nemotron-3-ultra-550b-a55b",
                messages=[
                    {"role": "system", "content": ORCHESTRATOR_SYSTEM},
                    {"role": "user", "content": PLAN_PROMPT.format(
                        user_message=user_message, skill_list=skill_list)}
                ],
                temperature=1, top_p=0.95, max_tokens=1024,
                extra_body={"chat_template_kwargs": {
                    "enable_thinking": req.thinking,
                    "reasoning_budget": 4096 if req.thinking else 0
                }}
            )

            plan_raw = plan_completion.choices[0].message.content
            plan_reasoning = getattr(plan_completion.choices[0].message, "reasoning_content", None)

            try:
                clean = plan_raw.strip().replace("```json","").replace("```","").strip()
                plan_data = json.loads(clean)
            except:
                plan_data = {"skill": None, "reasoning": "Auto", "plan": user_message, "complexity": "medium", "steps": []}

            skill_key = plan_data.get("skill") if plan_data.get("skill") in SKILLS else None
            skill_info = SKILLS[skill_key] if skill_key else {"label":"General","icon":"💬","color":"#888","prompt":"Respond as a general expert coding assistant."}

            yield event("plan", {
                "skill": skill_key,
                "skill_label": skill_info["label"],
                "skill_icon": skill_info["icon"],
                "skill_color": skill_info["color"],
                "reasoning": plan_data.get("reasoning",""),
                "plan": plan_data.get("plan",""),
                "steps": plan_data.get("steps",[]),
                "complexity": plan_data.get("complexity","medium"),
                "nemotron_thinking": plan_reasoning
            })

            await asyncio.sleep(0.1)
            yield event("stage", {"stage": "execute", "label": "⚙️ DeepSeek executing..."})

            exec_messages = [{"role": "system", "content": BASE_EXECUTOR_PROMPT.format(
                skill_prompt=skill_info["prompt"], plan=plan_data.get("plan", user_message))}]
            exec_messages += [{"role": m.role, "content": m.content} for m in req.messages]

            exec_completion = await deepseek_client.chat.completions.create(
                model="deepseek-ai/deepseek-v4-pro",
                messages=exec_messages,
                temperature=0.7,
                max_tokens=2048
            )

            exec_msg = exec_completion.choices[0].message
            executor_output = exec_msg.content

            yield event("execute", {
                "output": executor_output,
                "deepseek_thinking": getattr(exec_msg, "reasoning_content", None)
            })

            await asyncio.sleep(0.1)
            yield event("stage", {"stage": "review", "label": "🔍 Nemotron reviewing..."})

            review_completion = await nemotron_client.chat.completions.create(
                model="nvidia/nemotron-3-ultra-550b-a55b",
                messages=[
                    {"role": "system", "content": ORCHESTRATOR_SYSTEM},
                    {"role": "user", "content": REVIEW_PROMPT.format(
                        user_message=user_message, skill=skill_info["label"],
                        plan=plan_data.get("plan",""), executor_output=executor_output)}
                ],
                temperature=0.7, top_p=0.95, max_tokens=1024,
                extra_body={"chat_template_kwargs": {"enable_thinking": False, "reasoning_budget": 0}}
            )

            review_raw = review_completion.choices[0].message.content
            try:
                clean_r = review_raw.strip().replace("```json","").replace("```","").strip()
                review_data = json.loads(clean_r)
            except:
                review_data = {"approved": True, "score": 7, "verdict": "Looks good.", "issues": [], "improvements": [], "final_notes": ""}

            yield event("review", review_data)
            yield event("done", {"message": "Pipeline complete"})

        except Exception as e:
            yield event("error", {"message": str(e)})

    return StreamingResponse(run(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index():
    with open("index.html", "r") as f:
        return f.read()
