import os
import uuid
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class SessionManager:
    """
    Manages conversation sessions, including creating session IDs,
    and loading/saving conversation history to JSONL files.
    """
    def __init__(self, session_dir: str = "sessions"):
        self.session_dir = session_dir
        if not os.path.exists(self.session_dir):
            os.makedirs(self.session_dir)
            logger.info(f"Created session directory at: {self.session_dir}")

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

    def save_message(self, session_id: str, message: Any) -> None:
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

        with open(filepath, "a") as f:
            f.write(json.dumps(message_dict) + "\n")

    def save_log(self, session_id: str, log_entry: Dict[str, Any]) -> None:
        """
        Saves a structured log entry to the session's JSONL file.
        """
        filepath = self.get_session_filepath(session_id)
        with open(filepath, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
