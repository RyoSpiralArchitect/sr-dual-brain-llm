# ============================================================================
#  SpiralReality Proprietary
#  Copyright (c) 2025 SpiralReality. All Rights Reserved.
#
#  NOTICE: This file contains confidential and proprietary information of
#  SpiralReality. ANY USE, COPYING, MODIFICATION, DISTRIBUTION, DISPLAY,
#  OR DISCLOSURE OF THIS FILE, IN WHOLE OR IN PART, IS STRICTLY PROHIBITED
#  WITHOUT THE PRIOR WRITTEN CONSENT OF SPIRALREALITY.
#
#  NO LICENSE IS GRANTED OR IMPLIED BY THIS FILE. THIS SOFTWARE IS PROVIDED
#  "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
#  NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
#  PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL SPIRALREALITY OR ITS
#  SUPPLIERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN
#  AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
#  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ============================================================================

import asyncio, json, os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
app = FastAPI(); clients=set()
@app.websocket("/events")
async def events(ws: WebSocket):
    await ws.accept(); clients.add(ws)
    try:
        while True: await asyncio.sleep(60)
    except WebSocketDisconnect:
        clients.discard(ws)
@app.get("/")
def index():
    here = os.path.dirname(__file__); from fastapi.responses import FileResponse
    return FileResponse(os.path.join(here, "static", "index.html"))
async def broadcast(ev: dict):
    dead=[]; data=json.dumps(ev, ensure_ascii=False)
    for ws in list(clients):
        try: await ws.send_text(data)
        except Exception: dead.append(ws)
    for ws in dead:
        try: clients.discard(ws)
        except Exception: pass
