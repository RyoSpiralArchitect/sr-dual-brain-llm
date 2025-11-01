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

import asyncio
from typing import Callable, Awaitable, Any, Dict, List
class EventBus:
    async def publish(self, ev: Dict[str, Any]): raise NotImplementedError
    def subscribe(self, coro: Callable[[Dict[str, Any]], Awaitable[None]]): raise NotImplementedError
    def attach_tracer(self, tracer): self._tracer = tracer
class MemoryBus(EventBus):
    def __init__(self):
        self._subs: List[Callable[[Dict[str, Any]], Awaitable[None]]] = []
        self._q: asyncio.Queue = asyncio.Queue(); self._tracer = None
        self._task = asyncio.create_task(self._pump())
    async def _pump(self):
        while True:
            ev = await self._q.get()
            if getattr(self, "_tracer", None) is not None:
                try: self._tracer.log_dict(ev)
                except Exception: pass
            for sub in list(self._subs):
                try: await sub(ev)
                except Exception: pass
    async def publish(self, ev: Dict[str, Any]):
        await self._q.put(ev)
    def subscribe(self, coro):
        self._subs.append(coro)
