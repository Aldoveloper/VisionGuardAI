from fastapi import FastAPI, WebSocket
import uvicorn

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket): # Definir el endpoint
    await websocket.accept() # Aceptar la conexión
       # tipar que clase de dato se esta recibiendo  

    while True:
        data = await websocket.receive_text()
        await websocket.send_text(f"Copiado, que me dice: {data}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)



