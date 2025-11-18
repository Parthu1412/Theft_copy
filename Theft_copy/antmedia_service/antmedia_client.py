import logging
import json
from aiortc import RTCSessionDescription, RTCIceCandidate


class AntMediaClient:
    """Client for Ant Media Server WebRTC communications"""

    def __init__(self, stream_id, token, websocket):
        """
        Initialize the client

        Args:
            stream_id: The ID of the stream to play
            token: Authentication token
            websocket: Connected websocket instance
        """
        self.stream_id = stream_id
        self.token = token
        self.websocket = websocket
        self.logger = logging.getLogger(f"AntMediaClient-{stream_id}")
        self.peer_connections = {}

    async def play(self, peer_connection):
        """
        Send play request and set up event handlers

        Args:
            peer_connection: RTCPeerConnection instance
        """
        # Store the peer connection in our dictionary

        self.peer_connections[self.stream_id] = peer_connection

        # Add event handlers for ICE candidates
        @peer_connection.on("icecandidate")
        async def on_icecandidate(event):
            if event.candidate:
                candidate = event.candidate.candidate
                sdpMLineIndex = event.candidate.sdpMLineIndex or 0

                message = {
                    "command": "takeCandidate",
                    "streamId": self.stream_id,
                    "candidate": candidate,
                    "label": sdpMLineIndex,
                    "id": sdpMLineIndex,
                }

                await self.websocket.send_str(json.dumps(message))

        # Send play request
        play_request = {
            "command": "play",
            "streamId": self.stream_id,
        }

        # Add token if available
        if self.token:
            play_request["token"] = self.token

        self.logger.info(f"Sending play request for {self.stream_id}")
        await self.websocket.send_str(json.dumps(play_request))

    async def process_message(self, message_data, peer_connection):
        """
        Process a message from the server

        Args:
            message_data: Message data as string
            peer_connection: RTCPeerConnection instance
        """

        try:
            data = json.loads(message_data)
            command = data.get("command")

            if command == "start":
                self.logger.info(f"Stream started: {data.get('streamId')}")

            elif command == "takeConfiguration":
                await self._handle_sdp(data, peer_connection)

            elif command == "takeCandidate":
                await self._handle_ice_candidate(data)

            elif command == "notification":
                definition = data.get("definition", "")
                self.logger.info(f"Notification: {definition}")

            elif command == "error":
                self.logger.error(f"Error: {data.get('definition')}")

        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON: {message_data}")
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")

    async def _handle_sdp(self, data, peer_connection):
        """
        Handle SDP configuration from the server

        Args:
            data: SDP data from server
            peer_connection: RTCPeerConnection instance
        """
        stream_id = data.get("streamId")
        sdp_type = data.get("type")
        sdp = data.get("sdp")

        if stream_id != self.stream_id:
            return

        if sdp_type == "offer":
            self.logger.info(f"Setting remote description (offer)")
            offer = RTCSessionDescription(sdp=sdp, type=sdp_type)
            await peer_connection.setRemoteDescription(offer)

            # Create answer
            answer = await peer_connection.createAnswer()
            await peer_connection.setLocalDescription(answer)

            # Send answer
            response = {
                "command": "takeConfiguration",
                "streamId": self.stream_id,
                "type": "answer",
                "sdp": peer_connection.localDescription.sdp,
            }
            await self.websocket.send_str(json.dumps(response))

    async def _handle_ice_candidate(self, data):
        """Handle ICE candidate from the server"""
        stream_id = data.get("streamId")
        candidate_str = data.get("candidate")
        label = data.get("label", 0)

        if not stream_id or not candidate_str:
            self.logger.error("Missing streamId or candidate")
            return

        if stream_id not in self.peer_connections:
            self.logger.error(f"No peer connection for stream: {stream_id}")
            return

        pc = self.peer_connections[stream_id]
        self.logger.info(f"[ICE] Received candidate from server: {candidate_str}")

        try:
            # Parse the candidate string to extract components
            # Format: candidate:foundation component protocol priority ip port typ type [...]
            parts = candidate_str.split()
            if len(parts) < 8:
                self.logger.error(f"Invalid ICE candidate format: {candidate_str}")
                return

            # Extract the basic required fields
            foundation = parts[0].split(":")[1]  # Remove 'candidate:' prefix
            component = int(parts[1])
            protocol = parts[2]
            priority = int(parts[3])
            ip = parts[4]
            port = int(parts[5])
            # parts[6] should be "typ"
            type = parts[7]

            # Create a proper RTCIceCandidate object with parsed values
            ice_candidate = RTCIceCandidate(
                component=component,
                foundation=foundation,
                ip=ip,
                port=port,
                priority=priority,
                protocol=protocol,
                type=type,
                sdpMid=str(label),
                sdpMLineIndex=int(label),
            )

            # Add the ICE candidate
            await pc.addIceCandidate(ice_candidate)
            self.logger.info(f"Successfully added ICE candidate for stream {stream_id}")
        except Exception as e:
            self.logger.error(f"Error adding ICE candidate: {e}")
            import traceback

            self.logger.error(traceback.format_exc())

    async def close(self):
        """Close client resources"""
        # Nothing to do here as the WebSocket is managed by the caller
        pass
