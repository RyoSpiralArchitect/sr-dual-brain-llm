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
import uuid
from typing import Dict, Any

class Callosum:
    """In-memory asyncio message bus with slotting and futures."""
    def __init__(self, slot_ms: int = 250):
        self.slot_ms = slot_ms
        self.request_queue: asyncio.Queue = asyncio.Queue()
        self._response_futures: Dict[str, asyncio.Future] = {}

    async def ask_detail(self, payload: Dict[str, Any], timeout_ms: int = 3000) -> Dict[str, Any]:
        qid = payload.get("qid") or str(uuid.uuid4())
        payload["qid"] = qid
        fut = asyncio.get_event_loop().create_future()
        self._response_futures[qid] = fut
        await self.request_queue.put(payload)
        await asyncio.sleep(self.slot_ms / 1000.0)
        try:
            return await asyncio.wait_for(fut, timeout=timeout_ms/1000.0)
        except asyncio.TimeoutError:
            if qid in self._response_futures:
                del self._response_futures[qid]
            raise

    async def publish_response(self, qid: str, response: Dict[str, Any]):
        fut = self._response_futures.get(qid)
        if fut and not fut.done():
            fut.set_result(response)
            del self._response_futures[qid]

    async def recv_request(self) -> Dict[str, Any]:
        return await self.request_queue.get()
