import os
import logging
import datetime
from dotenv import load_dotenv

# Google Cloud
import google.cloud.logging
from google.cloud import datastore

# FastAPI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# MCP & ADK
from mcp.server.fastmcp import FastMCP
from google.adk import Agent
from google.adk.agents import SequentialAgent
from google.adk.tools.tool_context import ToolContext

# ==========================================
# 1. Setup Logging & Environment
# ==========================================

try:
    cloud_logging_client = google.cloud.logging.Client()
    cloud_logging_client.setup_logging()
except Exception:
    logging.basicConfig(level=logging.INFO)

load_dotenv()

model_name = os.getenv("MODEL", "gemini-1.5-pro")
DB_ID = os.getenv("DATABASE_ID", "hacka")

# Datastore client
db = datastore.Client(database=DB_ID)

# MCP Tool container
mcp = FastMCP("WorkspaceTools")

# ==========================================
# 2. BULLETPROOF TOOLS
# ==========================================

@mcp.tool()
def add_task(title: str) -> str:
    try:
        key = db.key('Task')
        task = datastore.Entity(key=key)
        task.update({
            'title': title,
            'completed': False,
            'created_at': datetime.datetime.now()
        })
        db.put(task)
        return f"Task '{title}' added successfully."
    except Exception as e:
        logging.error(f"Error adding task: {str(e)}")
        return f"Failed to add task: {str(e)}"


@mcp.tool()
def list_tasks() -> str:
    try:
        query = db.query(kind='Task')
        all_entities = list(query.fetch())

        if not all_entities:
            return "No tasks found."

        all_entities.sort(key=lambda x: x.get('created_at', datetime.datetime.min), reverse=True)

        task_list = []
        for task in all_entities:
            title = task.get('title') or "Untitled Task"
            status = "✅" if task.get('completed') else "❌"
            task_list.append(f"{status} {title} (ID: {task.key.id})")

        return f"📋 Tasks ({DB_ID}):\n" + "\n".join(task_list)
    except Exception as e:
        return f"Error listing tasks: {str(e)}"


@mcp.tool()
def complete_task(task_id: str) -> str:
    try:
        numeric_id = int(''.join(filter(str.isdigit, task_id)))
        key = db.key('Task', numeric_id)
        task = db.get(key)
        if task:
            task['completed'] = True
            db.put(task)
            return f"Task {numeric_id} marked as complete."
        else:
            return "Task not found."
    except Exception as e:
        return f"Error completing task: {str(e)}"


@mcp.tool()
def add_note(title: str, content: str) -> str:
    try:
        key = db.key('Note')
        note = datastore.Entity(key=key, exclude_from_indexes=['content'])
        note.update({
            'title': title,
            'content': content,
            'timestamp': datetime.datetime.now()
        })
        db.put(note)
        return f"Note '{title}' added successfully (ID: {note.key.id})."
    except Exception as e:
        return f"Failed to add note: {str(e)}"


@mcp.tool()
def list_notes() -> str:
    try:
        query = db.query(kind='Note')
        all_notes = list(query.fetch())

        if not all_notes:
            return "No notes found."

        notes_list = []
        for note in all_notes:
            title = note.get('title', 'Untitled')
            content_preview = note.get('content', '')[:50]
            notes_list.append(f"📝 {title} - {content_preview}... (ID: {note.key.id})")

        return "📒 Notes:\n" + "\n".join(notes_list)
    except Exception as e:
        return f"Error listing notes: {str(e)}"


# ==========================================
# 3. TOOL TO INJECT PROMPT INTO AGENT STATE
# ==========================================

def add_prompt_to_state(tool_context: ToolContext, prompt: str) -> dict:
    tool_context.state["PROMPT"] = prompt
    return {"status": "success"}


# ==========================================
# 4. MULTI-AGENT WORKFLOW
# ==========================================

workspace_coordinator = Agent(
    name="workspace_coordinator",
    model=model_name,
    description="Agent managing Cloud Datastore tasks and notes.",
    instruction="""
You are an Executive Assistant.
Use tools to satisfy the PROMPT provided in the shared state.

USER REQUEST: {PROMPT | default('No prompt provided.')}
""",
    tools=[add_task, list_tasks, complete_task, add_note, list_notes],
    output_key="execution_data"
)

response_formatter = Agent(
    name="response_formatter",
    model=model_name,
    description="Formats raw database output.",
    instruction="Summarize the EXECUTION DATA for the user: {execution_data | default('No data.')}"
)

workspace_workflow = SequentialAgent(
    name="workspace_workflow",
    sub_agents=[workspace_coordinator, response_formatter]
)

root_agent = Agent(
    name="workspace_greeter",
    model=model_name,
    tools=[add_prompt_to_state],
    sub_agents=[workspace_workflow]
)


# ==========================================
# 5. FASTAPI ENDPOINT
# ==========================================

app = FastAPI()


class UserRequest(BaseModel):
    prompt: str


@app.post("/api/v1/workspace/chat")
async def chat(request: UserRequest):
    try:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        enriched = f"SYSTEM TIME: {now} | User Request: {request.prompt}"

        # Reset agent state per request
        root_agent.state = {}

        # Inject the user prompt into the PROMPT variable
        add_prompt_to_state(ToolContext(root_agent.state), request.prompt)

        # Invoke the root agent workflow
        result = root_agent.invoke({"user_input": enriched})

        return {
            "status": "success",
            "reply": result.get("final_output", "Processed.")
        }

    except Exception as e:
        logging.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# 6. RUN SERVER
# ==========================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)