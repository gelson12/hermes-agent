from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class LiveKitSession:
    room_name: str
    participant_id: str
    metadata: Dict[str, Any]


class LiveKitAdapter:

    def __init__(self, hermes_runtime):
        self.runtime = hermes_runtime
        self.sessions = {}

    async def handle_user_message(
        self,
        room_name: str,
        participant_id: str,
        text: str,
    ):

        session_key = f"{room_name}:{participant_id}"

        if session_key not in self.sessions:
            self.sessions[session_key] = LiveKitSession(
                room_name=room_name,
                participant_id=participant_id,
                metadata={}
            )

        response = await self.runtime.chat(
            session_id=session_key,
            message=text,
            platform="livekit"
        )

        return response
