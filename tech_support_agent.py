import nest_asyncio
from typing import List, Optional, Dict
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
import datetime

nest_asyncio.apply()

class SupportQuery(BaseModel):
    issue: str
    severity: str
    product: str
    user_id: str
    timestamp: datetime.datetime = datetime.datetime.now()

class SupportResponse(BaseModel):
    solution: str
    next_steps: List[str]
    escalate: bool
    priority_level: int 
    estimated_time: str

class KnowledgeBase(BaseModel):
    product: str
    known_issues: Dict[str, str]
    solutions: Dict[str, List[str]]

kb = KnowledgeBase(
        product="CloudDB",
        known_issues={
            "Can't connect to database": "Check connection string and firewall rules",
            "Database crash": "Verify system resources and restart service",
            "Slow queries": "Analyze query performance and optimize indexes"
        },
        solutions={
            "connection": ["Check credentials", "Verify network access", "Test port availability"],
            "performance": ["Run diagnostics", "Check resource usage", "Optimize queries"],
            "crash": ["Collect logs", "Check error messages", "Restart service"]
        }
    )

tech_support_agent = Agent(
    model="groq:llama3-groq-70b-8192-tool-use-preview",
    deps_type=KnowledgeBase,
    result_type=SupportResponse,
    system_prompt="""
    You are an expert technical support AI agent. Your role is to:
    1. Analyze customer issues using the provided tools
    2. Search the knowledge base for known solutions
    3. Provide clear, actionable solutions
    4. Determine if escalation is needed
    5. Estimate resolution time
    
    Always use the available tools to verify information before responding.
    """
)

#Define tools for the agent
@tech_support_agent.tool
async def search_knowledge_base(ctx: RunContext[KnowledgeBase], issue: str, product: str) -> Dict[str, str]:
    """Search the knowledge base for known issues and solutions"""
    kb = ctx.deps
    if product != kb.product:
        return {"error": "Product not found in knowledge base"}
    
    matches = {}
    for known_issue, solution in kb.known_issues.items():
        if issue.lower() in known_issue.lower():
            matches[known_issue] = solution
    return matches

@tech_support_agent.tool
async def check_severity(ctx: RunContext[KnowledgeBase], issue: str, severity: str) -> Dict[str, any]:
    """Analyze issue severity and recommend priority level"""
    severity_levels = {
        "low": 1,
        "medium": 2,
        "high": 3,
        "critical": 4
    }
    
    base_priority = severity_levels.get(severity.lower(), 1)
    
    # Check if issue contains critical keywords
    critical_keywords = ["crash", "data loss", "security", "breach"]
    if any(keyword in issue.lower() for keyword in critical_keywords):
        base_priority = max(base_priority, 3)
    
    return {
        "priority_level": base_priority,
        "needs_escalation": base_priority >= 3,
        "estimated_time": f"{base_priority * 2}h"
    }

#Support Query Handler
async def handle_support_query(query: SupportQuery) -> SupportResponse:
    # Initialize knowledge base with sample data
    kb = KnowledgeBase(
        product="CloudDB",
        known_issues={
            "Can't connect to database": "Check connection string and firewall rules",
            "Database crash": "Verify system resources and restart service",
            "Slow queries": "Analyze query performance and optimize indexes"
        },
        solutions={
            "connection": ["Check credentials", "Verify network access", "Test port availability"],
            "performance": ["Run diagnostics", "Check resource usage", "Optimize queries"],
            "crash": ["Collect logs", "Check error messages", "Restart service"]
        }
    )
    
    # Process the query using our agent
    result = await tech_support_agent.run(
        f"""
        Customer Issue:
        Product: {query.product}
        Severity: {query.severity}
        Issue: {query.issue}
        
        Please analyze this issue and provide a solution.
        """,
        deps=kb
    )
    
    return result.data

if __name__ == "__main__":
    import asyncio
    
    # Create a sample query
    query = SupportQuery(
        issue="Can't connect to database after restart",
        severity="high",
        product="CloudDB",
        user_id="user123"
    )
    
    # Run the support agent
    response = asyncio.run(handle_support_query(query))
    print(f"Solution: {response.solution}")
    print(f"Next steps: {response.next_steps}")
    print(f"Escalate: {response.escalate}")
    print(f"Priority: {response.priority_level}")
    print(f"Estimated time: {response.estimated_time}")

