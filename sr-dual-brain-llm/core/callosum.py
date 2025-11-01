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
from typing import Any, Dict, Mapping

from .transport_models import DetailRequest, DetailResponse, ensure_mapping

class Callosum:
    """In-memory asyncio message bus with slotting and futures."""
    def __init__(self, slot_ms: int = 250):
        self.slot_ms = slot_ms
        self.request_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue()
        self._response_futures: Dict[str, asyncio.Future[DetailResponse]] = {}

    async def ask_detail(
        self,
        payload: DetailRequest | Mapping[str, Any],
        timeout_ms: int = 3000,
    ) -> DetailResponse:
        message = ensure_mapping(payload)
        qid = message.get("qid") or str(uuid.uuid4())
        message["qid"] = qid
        message.setdefault("type", "ASK_DETAIL")

        loop = asyncio.get_event_loop()
        fut: asyncio.Future[DetailResponse] = loop.create_future()
        self._response_futures[qid] = fut
        await self.request_queue.put(message)
        await asyncio.sleep(self.slot_ms / 1000.0)
        try:
            raw = await asyncio.wait_for(fut, timeout=timeout_ms / 1000.0)
            return raw
        except asyncio.TimeoutError:
            if qid in self._response_futures:
                del self._response_futures[qid]
            raise

    async def publish_response(
        self,
        qid: str,
        response: DetailResponse | Mapping[str, Any],
    ) -> None:
        payload = ensure_mapping(response)
        future = self._response_futures.get(qid)
        if future and not future.done():
            try:
                detail = DetailResponse.from_payload(payload)
            except Exception:  # pragma: no cover - defensive guard
                future.set_result(DetailResponse(qid=qid, error="invalid response"))
            else:
                future.set_result(detail)
            del self._response_futures[qid]

    async def recv_request(self) -> Dict[str, Any]:
        return await self.request_queue.get()

    async def recv_detail_request(self) -> DetailRequest:
        payload = await self.recv_request()
        return DetailRequest.from_payload(payload)
