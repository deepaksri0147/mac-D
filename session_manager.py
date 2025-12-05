import os
import uuid
import json
import logging
from typing import List, Dict, Any, Optional
import aiofiles
import requests
from dotenv import load_dotenv
from datetime import datetime, timezone
from api_executor import APIExecutor

load_dotenv()
 
logger = logging.getLogger(__name__)
 
class SessionManager:
    """
    Manages conversation sessions, including creating session IDs,
    and loading/saving conversation history to JSONL files and a remote API.
    """
    def __init__(self, session_dir: str = "sessions"):
        self.session_dir = session_dir
        if not os.path.exists(self.session_dir):
            os.makedirs(self.session_dir)
            logger.info(f"Created session directory at: {self.session_dir}")
        self.log_api_endpoint = os.getenv("LOG_API_ENDPOINT", f'https://igs.gov-cloud.ai/pi-entity-instances-service/v2.0/schemas/{os.getenv("SCHEMA_ID")}/instances')
        self.api_executor = APIExecutor()

    def create_session_id(self) -> str:
        """Generates a new, unique session ID."""
        return str(uuid.uuid4())

    def get_session_filepath(self, session_id: str) -> str:
        """Constructs the full filepath for a given session ID."""
        return os.path.join(self.session_dir, f"{session_id}.jsonl")

    def load_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Loads the conversation history for a given session ID from its JSONL file.
        """
        filepath = self.get_session_filepath(session_id)
        history = []
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        # Only append message entries to the history, not logs
                        if "event" not in entry:
                            history.append(entry)
                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse line in {filepath}: {line}")
        return history

    async def save_message(self, session_id: str, message: Any) -> None:
        """
        Saves a single message to the session's JSONL file.
        The message is expected to be a LangChain message object that can be serialized.
        """
        filepath = self.get_session_filepath(session_id)
        
        # Convert LangChain message object to a serializable dictionary
        try:
            # This relies on LangChain's internal serialization, which is robust.
            from langchain_core.messages import message_to_dict
            message_dict = message_to_dict(message)
        except (ImportError, AttributeError):
            # Fallback for non-LangChain objects or if the utility is not available
            if hasattr(message, 'dict'):
                message_dict = message.dict()
            elif isinstance(message, dict):
                message_dict = message
            else:
                logger.error(f"Cannot serialize message of type {type(message)}")
                return

        async with aiofiles.open(filepath, "a") as f:
            await f.write(json.dumps(message_dict) + "\n")

    async def _log_to_api(self, session_id: str, event_type: str, raw_data: Dict[str, Any]):
        """Asynchronously sends a log entry to the remote API."""
        payload = {
            "data": [
                {
                    "session_id": session_id,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "event_type": event_type,
                    "raw_data": raw_data,
                }
            ]
        }
        
        auth_token = self.api_executor._get_auth_token()
        if not auth_token:
            logger.error("Could not retrieve auth token for logging API.")
            return

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {auth_token}",
        }

        try:
            # Using requests for simplicity as it's not in a performance-critical path for this task
            # For a fully async implementation, httpx would be used here.
            response = requests.post(self.log_api_endpoint, headers=headers, json=payload)
            
            response.raise_for_status()
            logger.info(f"Successfully logged event '{event_type}' to API for session {session_id}.")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to log event to API: {e}")

    async def save_message(self, session_id: str, message: Any) -> None:
        """
        Saves a single message to the session's JSONL file and logs it to the API.
        The message is expected to be a LangChain message object that can be serialized.
        """
        filepath = self.get_session_filepath(session_id)
        
        try:
            from langchain_core.messages import message_to_dict
            message_dict = message_to_dict(message)
        except (ImportError, AttributeError):
            if hasattr(message, 'dict'):
                message_dict = message.dict()
            elif isinstance(message, dict):
                message_dict = message
            else:
                logger.error(f"Cannot serialize message of type {type(message)}")
                return

        async with aiofiles.open(filepath, "a") as f:
            await f.write(json.dumps(message_dict) + "\n")

        event_type = message_dict.get("type")
        
        if event_type in ["human", "ai", "tool"]:
            raw_data = message_dict.get("data", {})
        else:
            raw_data = message_dict

        if event_type == "ai" and raw_data.get("tool_calls"):
            event_type = "tool_calls"
        elif event_type == "tool":
            event_type = "tool_response"

        await self._log_to_api(session_id, event_type, raw_data)

    async def save_log(self, session_id: str, log_entry: Dict[str, Any]) -> None:
        """
        Saves a structured log entry to the session's JSONL file and logs it to the API.
        """
        filepath = self.get_session_filepath(session_id)
        async with aiofiles.open(filepath, "a") as f:
            await f.write(json.dumps(log_entry) + "\n")

        event_type = log_entry.get("event", "unknown")
        raw_data = {k: v for k, v in log_entry.items() if k != "event"} if isinstance(log_entry, dict) else log_entry
        await self._log_to_api(session_id, event_type, raw_data)
