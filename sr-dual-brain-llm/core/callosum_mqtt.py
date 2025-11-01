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
import json
import threading
import time
import uuid
from typing import Any, Dict, Mapping
import paho.mqtt.client as mqtt

from .transport_models import DetailRequest, DetailResponse, ensure_mapping


class CallosumMQTT:
    """MQTT-backed Callosum prototype (paho-mqtt)."""
    def __init__(self, broker_host='localhost', broker_port=1883,
                 req_topic='callosum/requests', res_topic='callosum/responses',
                 slot_ms=250):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.req_topic = req_topic
        self.res_topic = res_topic
        self.slot_ms = slot_ms
        self._loop = asyncio.get_event_loop()
        self._response_futures: Dict[str, asyncio.Future[DetailResponse]] = {}

        self.client = mqtt.Client()
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.connect(self.broker_host, self.broker_port)
        self.client.loop_start()
        time.sleep(0.2)

    def _on_connect(self, client, userdata, flags, rc):
        client.subscribe(self.res_topic)

    def _on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode('utf-8'))
            qid = payload.get('qid')
            if qid and qid in self._response_futures:
                fut = self._response_futures[qid]
                try:
                    detail = DetailResponse.from_payload(payload)
                except Exception:
                    detail = DetailResponse(qid=str(qid), error='invalid response')
                self._loop.call_soon_threadsafe(self._set_future_result, qid, fut, detail)
        except Exception:
            pass

    def _set_future_result(self, qid, fut, detail):
        if fut and not fut.done():
            fut.set_result(detail)
            del self._response_futures[qid]

    async def ask_detail(
        self,
        payload: DetailRequest | Mapping[str, Any],
        timeout_ms: int = 5000,
    ) -> DetailResponse:
        message = ensure_mapping(payload)
        qid = message.get("qid") or str(uuid.uuid4())
        message["qid"] = qid
        message.setdefault("type", "ASK_DETAIL")
        fut: asyncio.Future[DetailResponse] = asyncio.get_event_loop().create_future()
        self._response_futures[qid] = fut
        self.client.publish(self.req_topic, json.dumps(message))
        await asyncio.sleep(self.slot_ms / 1000.0)
        try:
            return await asyncio.wait_for(fut, timeout=timeout_ms/1000.0)
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
        payload.setdefault("qid", qid)
        self.client.publish(self.res_topic, json.dumps(payload))
