from fastapi import FastAPI
import uvicorn
# from app.routes.websocket import websocket_endpoint
from app.controllers.websocket_controller import websocket_endpoint

app = FastAPI()

# WebSocket para otro fin
app.add_api_websocket_route("/ws", websocket_endpoint)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)