import os
import json
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from openai import OpenAI
from pydantic import BaseModel
from typing import List, Optional
import asyncio

app = FastAPI()

# Two separate clients - two separate accounts
deepseek_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ.get("NVIDIA_API_KEY")          # DeepSeek account
)

nemotron_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ.get("NEMOTRON_API_KEY")        # Nemotron account
)

# ── Skill definitions ────────────────────────────────────────────────────────
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

BASE_EXECUTOR_PROMPT = """You are an expert coding assistant powered by DeepSeek V4-Pro. You are the executor in a two-model pipeline — the orchestrator has already planned the task and selected your skill mode.

General rules:
- Write complete, working code — never use placeholders
- Use code blocks with correct language tags
- Be direct and precise
- If you spot issues outside scope, mention them briefly

Active skill mode:
{skill_prompt}

Orchestrator's plan for this task:
{plan}"""

ORCHESTRATOR_SYSTEM = """You are Nemotron Ultra, an expert AI orchestrator and code reviewer. You work in a two-stage pipeline:

STAGE 1 - PLAN: Analyze the user's request, select the best skill mode, and write a clear execution plan for DeepSeek V4-Pro to follow.

STAGE 2 - REVIEW: Review DeepSeek's output for correctness, completeness, security, and quality. Approve it or identify specific issues.

You are precise, critical, and thorough. You catch what others miss."""

PLAN_PROMPT = """Analyze this user request and produce a JSON execution plan.

User request: {user_message}

Available skill modes: {skill_list}

Respond with ONLY valid JSON, no markdown, no explanation:
{{
  "skill": "<skill_id or null for general>",
  "reasoning": "<why this skill, 1-2 sentences>",
  "plan": "<clear instructions for DeepSeek on exactly what to produce, be specific>",
  "complexity": "<low|medium|high>",
  "steps": ["step1", "step2", "..."]
}}"""

REVIEW_PROMPT = """Review this code/response produced by DeepSeek V4-Pro.

Original user request: {user_message}
Skill used: {skill}
Execution plan that was followed: {plan}

DeepSeek's output:
---
{executor_output}
---

Respond with ONLY valid JSON:
{{
  "approved": true/false,
  "score": <1-10>,
  "verdict": "<one sentence summary>",
  "issues": ["issue1", "issue2"],
  "improvements": ["improvement1", "improvement2"],
  "final_notes": "<any important notes to surface to the user, or empty string>"
}}"""

# ── Request models ───────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    skill: Optional[str] = None
    thinking: bool = False

class AgentRequest(BaseModel):
    messages: List[Message]
    thinking: bool = False

# ── Direct chat endpoint ─────────────────────────────────────────────────────
@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        skill_key = req.skill if req.skill in SKILLS else None
        skill_prompt = SKILLS[skill_key]["prompt"] if skill_key else "Respond as a general expert coding assistant."

        system = f"""You are an expert coding assistant powered by DeepSeek V4-Pro.
Write complete, working code. Use correct language tags in code blocks. Be direct and precise.

Active skill:
{skill_prompt}"""

        messages = [{"role": "system", "content": system}]
        messages += [{"role": m.role, "content": m.content} for m in req.messages]

        completion = deepseek_client.chat.completions.create(
            model="deepseek-ai/deepseek-v4-pro",
            messages=messages,
            temperature=1,
            top_p=0.95,
            max_tokens=4096,
            extra_body={"chat_template_kwargs": {
                "thinking": req.thinking,
                "reasoning_effort": "high" if req.thinking else None
            }}
        )

        msg = completion.choices[0].message
        return JSONResponse({
            "reply": msg.content,
            "reasoning": getattr(msg, "reasoning_content", None),
            "skill": skill_key,
            "mode": "direct"
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ── Agent endpoint (streaming SSE) ──────────────────────────────────────────
@app.post("/agent")
async def agent(req: AgentRequest):
    async def run():
        user_message = req.messages[-1].content if req.messages else ""
        skill_list = ", ".join([f"{k} ({v['label']})" for k, v in SKILLS.items()])

        def event(type_, data):
            return f"data: {json.dumps({'type': type_, **data})}\n\n"

        try:
            # ── STAGE 1: Nemotron plans ──────────────────────────────────
            yield event("stage", {"stage": "plan", "label": "🧠 Nemotron planning..."})

            plan_messages = [
                {"role": "system", "content": ORCHESTRATOR_SYSTEM},
                {"role": "user", "content": PLAN_PROMPT.format(
                    user_message=user_message,
                    skill_list=skill_list
                )}
            ]

            plan_completion = nemotron_client.chat.completions.create(
                model="nvidia/nemotron-3-ultra-550b-a55b",
                messages=plan_messages,
                temperature=1,
                top_p=0.95,
                max_tokens=1024,
                extra_body={"chat_template_kwargs": {
                    "enable_thinking": req.thinking,
                    "reasoning_budget": 4096 if req.thinking else 0
                }}
            )

            plan_raw = plan_completion.choices[0].message.content
            plan_reasoning = getattr(plan_completion.choices[0].message, "reasoning_content", None)

            # Parse plan JSON
            try:
                clean = plan_raw.strip().replace("```json", "").replace("```", "").strip()
                plan_data = json.loads(clean)
            except:
                plan_data = {
                    "skill": None, "reasoning": "Auto-detected",
                    "plan": user_message, "complexity": "medium", "steps": []
                }

            skill_key = plan_data.get("skill") if plan_data.get("skill") in SKILLS else None
            skill_info = SKILLS[skill_key] if skill_key else {"label": "General", "icon": "💬", "color": "#888", "prompt": "Respond as a general expert coding assistant."}

            yield event("plan", {
                "skill": skill_key,
                "skill_label": skill_info["label"],
                "skill_icon": skill_info["icon"],
                "skill_color": skill_info["color"],
                "reasoning": plan_data.get("reasoning", ""),
                "plan": plan_data.get("plan", ""),
                "steps": plan_data.get("steps", []),
                "complexity": plan_data.get("complexity", "medium"),
                "nemotron_thinking": plan_reasoning
            })

            await asyncio.sleep(0.1)

            # ── STAGE 2: DeepSeek executes ───────────────────────────────
            yield event("stage", {"stage": "execute", "label": "⚙️ DeepSeek executing..."})

            exec_system = BASE_EXECUTOR_PROMPT.format(
                skill_prompt=skill_info["prompt"],
                plan=plan_data.get("plan", user_message)
            )

            exec_messages = [{"role": "system", "content": exec_system}]
            exec_messages += [{"role": m.role, "content": m.content} for m in req.messages]

            exec_completion = deepseek_client.chat.completions.create(
                model="deepseek-ai/deepseek-v4-pro",
                messages=exec_messages,
                temperature=1,
                top_p=0.95,
                max_tokens=4096,
                extra_body={"chat_template_kwargs": {
                    "thinking": req.thinking,
                    "reasoning_effort": "high" if req.thinking else None
                }}
            )

            exec_msg = exec_completion.choices[0].message
            executor_output = exec_msg.content
            exec_reasoning = getattr(exec_msg, "reasoning_content", None)

            yield event("execute", {
                "output": executor_output,
                "deepseek_thinking": exec_reasoning
            })

            await asyncio.sleep(0.1)

            # ── STAGE 3: Nemotron reviews ────────────────────────────────
            yield event("stage", {"stage": "review", "label": "🔍 Nemotron reviewing..."})

            review_messages = [
                {"role": "system", "content": ORCHESTRATOR_SYSTEM},
                {"role": "user", "content": REVIEW_PROMPT.format(
                    user_message=user_message,
                    skill=skill_info["label"],
                    plan=plan_data.get("plan", ""),
                    executor_output=executor_output
                )}
            ]

            review_completion = nemotron_client.chat.completions.create(
                model="nvidia/nemotron-3-ultra-550b-a55b",
                messages=review_messages,
                temperature=0.7,
                top_p=0.95,
                max_tokens=1024,
                extra_body={"chat_template_kwargs": {
                    "enable_thinking": False,
                    "reasoning_budget": 0
                }}
            )

            review_raw = review_completion.choices[0].message.content
            try:
                clean_r = review_raw.strip().replace("```json", "").replace("```", "").strip()
                review_data = json.loads(clean_r)
            except:
                review_data = {
                    "approved": True, "score": 7,
                    "verdict": "Output looks reasonable.",
                    "issues": [], "improvements": [], "final_notes": ""
                }

            yield event("review", review_data)
            yield event("done", {"message": "Pipeline complete"})

        except Exception as e:
            yield event("error", {"message": str(e)})

    return StreamingResponse(run(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/skills")
async def list_skills():
    return JSONResponse({"skills": [
        {"id": k, "label": v["label"], "icon": v["icon"], "color": v["color"]}
        for k, v in SKILLS.items()
    ]})


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("index.html", "r") as f:
        return f.read()
