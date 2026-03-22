"""
pipeline/ — LangGraph-based triage orchestration.

Modules:
  llm_provider  : Dual-provider LLM (Groq primary, Gemini fallback)
  triage_graph  : StateGraph pipeline orchestrating all agents → LLM → triage decision
"""