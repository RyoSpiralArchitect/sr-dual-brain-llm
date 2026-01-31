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

import argparse
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.events import Timeline
from core.event_bus import MemoryBus
from dashboard.server import broadcast, app
import uvicorn


async def feeder(path: str, speed: float):
    tl = Timeline(path)
    async with MemoryBus() as bus:
        bus.subscribe(broadcast)
        for ev in tl.replay(speed=speed):
            await bus.publish(ev)


async def run(args) -> None:
    config = uvicorn.Config(app, host=args.host, port=args.port, reload=False)
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())
    try:
        await feeder(args.log, args.speed)
    finally:
        server.should_exit = True
        await server_task


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--host", default="127.0.0.1"); p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    asyncio.run(run(args))
if __name__ == "__main__": main()
