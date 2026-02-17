"""
Agent Runner — execution and iterative reasoning loop.

Orchestrates the full agentic RAG pipeline:
  1. Uses AgentWorker to select relevant tools
  2. Enters a reasoning loop where the LLM decides which tool to call
  3. Executes tools, accumulates context
  4. LLM evaluates if enough information is gathered
  5. Returns a final synthesized answer

Supports multi-hop reasoning: the agent can call multiple tools
across iterations, building up context incrementally.
"""

import json
from typing import List

from src.config import llm_chat, MAX_REASONING_STEPS
from src.agent_worker import AgentWorker
from src.tool_factory import Tool


# ── System Prompt for the Reasoning Agent ──────────────────────────

AGENT_SYSTEM_PROMPT = """\
You are an intelligent research agent. You answer questions by selecting and \
using the available tools. You MUST respond with valid JSON only — no markdown, \
no extra text.

AVAILABLE TOOLS:
{tool_list}

CONTEXT GATHERED SO FAR:
{context}

RULES:
1. If you need more information, call a tool by responding with:
   {{"action": "tool_call", "tool": "<tool_name>", "reasoning": "<why you chose this tool>"}}

2. If you have enough information to answer the user's question, respond with:
   {{"action": "final_answer", "answer": "<your comprehensive answer>", "reasoning": "<how you derived the answer>"}}

3. Choose the most relevant tool. Use vector tools for specific facts and \
summary tools for overviews.
4. Do NOT call a tool you have already called in a previous step.
5. Your answer should be thorough and directly address the question.\
"""


class AgentRunner:
    """Runs the agentic reasoning loop over selected tools."""

    def __init__(self, worker: AgentWorker):
        self.worker = worker

    def run(self, query: str) -> str:
        """
        Execute the full agentic pipeline for a given query.

        Returns the final answer string.
        """
        # ── Step 1: Tool Selection ──────────────────────────────────
        selected_tools = self.worker.select_tools(query)
        if not selected_tools:
            return "I couldn't find any relevant knowledge sources for your query."

        tool_map = {t.name: t for t in selected_tools}
        called_tools = set()
        context_parts: List[str] = []

        # ── Step 2: Reasoning Loop ──────────────────────────────────
        for step in range(1, MAX_REASONING_STEPS + 1):
            print(f"\n{'='*60}")
            print(f"  REASONING STEP {step}/{MAX_REASONING_STEPS}")
            print(f"{'='*60}")

            # Build tool list description for the prompt
            tool_list_str = "\n".join(
                f"  - {t.name}: {t.description}"
                for t in selected_tools
                if t.name not in called_tools
            )
            if not tool_list_str:
                tool_list_str = "(All available tools have been called)"

            # Build context string
            context_str = "\n\n".join(context_parts) if context_parts else "(none yet)"

            # Build prompt
            system = AGENT_SYSTEM_PROMPT.format(
                tool_list=tool_list_str,
                context=context_str,
            )
            user_prompt = f"USER QUESTION: {query}"

            # ── LLM Decision ────────────────────────────────────────
            raw_response = llm_chat(user_prompt, system_prompt=system)
            print(f"\n[Agent] LLM Response:\n{raw_response[:500]}")

            # Parse JSON response
            action = self._parse_action(raw_response)
            if action is None:
                # If parsing fails, treat the raw response as the final answer
                print("[Agent] ⚠ Could not parse LLM JSON — using raw response as answer")
                return raw_response

            # ── Handle Action ───────────────────────────────────────
            if action.get("action") == "final_answer":
                answer = action.get("answer", raw_response)
                reasoning = action.get("reasoning", "")
                print(f"\n[Agent] ✓ Final reasoning: {reasoning}")
                print(f"[Agent] ✓ Returning final answer.\n")
                return answer

            elif action.get("action") == "tool_call":
                tool_name = action.get("tool", "")
                reasoning = action.get("reasoning", "")
                print(f"\n[Agent] → Tool call: {tool_name}")
                print(f"[Agent]   Reason: {reasoning}")

                if tool_name in tool_map and tool_name not in called_tools:
                    tool = tool_map[tool_name]
                    print(f"[Agent]   Executing {tool_name} ...")
                    result = tool.function(query)
                    called_tools.add(tool_name)
                    context_parts.append(
                        f"[Result from {tool_name} ({tool.tool_type} on "
                        f"'{tool.document_name}')]:\n{result}"
                    )
                    print(f"[Agent]   ✓ Got {len(result)} chars from {tool_name}")
                elif tool_name in called_tools:
                    print(f"[Agent]   ⚠ Tool '{tool_name}' already called — skipping")
                else:
                    print(f"[Agent]   ⚠ Tool '{tool_name}' not found in selected tools")
            else:
                print(f"[Agent] ⚠ Unknown action type: {action.get('action')}")

        # ── Step 3: Fallback synthesis ──────────────────────────────
        print(f"\n[Agent] Max steps reached — synthesizing final answer ...\n")
        return self._synthesize_final(query, context_parts)

    # ── Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _parse_action(raw: str) -> dict | None:
        """Try to extract a JSON object from the LLM response."""
        raw = raw.strip()
        # Try direct parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        # Try to find JSON in the text
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start:end])
            except json.JSONDecodeError:
                pass
        return None

    @staticmethod
    def _synthesize_final(query: str, context_parts: List[str]) -> str:
        """Use the LLM to synthesize a final answer from gathered context."""
        context_str = "\n\n".join(context_parts) if context_parts else "(no context)"
        prompt = (
            f"Based on the following retrieved information, provide a comprehensive "
            f"answer to the question.\n\n"
            f"QUESTION: {query}\n\n"
            f"RETRIEVED INFORMATION:\n{context_str}\n\n"
            f"ANSWER:"
        )
        return llm_chat(prompt)
