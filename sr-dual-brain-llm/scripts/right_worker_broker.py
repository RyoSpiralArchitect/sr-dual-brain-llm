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

import os, json, asyncio
from core.shared_memory import SharedMemory
from core.models import RightBrainModel
from core.transport_models import DetailRequest, DetailResponse

# This worker listens on broker topics and replies.
# Select backend with CALLOSUM_BACKEND=kafka|mqtt

async def run_kafka():
    from core.callosum_kafka import CallosumKafka
    callosum = CallosumKafka()
    mem = SharedMemory(); right = RightBrainModel()
    print("Right worker (Kafka) listening...")
    # Polling loop: create a lightweight consumer by asking through request-topic?
    # Here we implement a simple bridging by directly consuming requests using a separate consumer:
    from kafka import KafkaConsumer, KafkaProducer
    consumer = KafkaConsumer('callosum_requests', bootstrap_servers='localhost:9092',
                             value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                             auto_offset_reset='earliest', enable_auto_commit=True,
                             group_id='right_worker')
    prod = KafkaProducer(bootstrap_servers='localhost:9092',
                         value_serializer=lambda v: json.dumps(v).encode('utf-8'))
    loop = asyncio.get_event_loop()
    for msg in consumer:
        payload = msg.value
        if payload.get("type")=="ASK_DETAIL":
            try:
                detail_req = DetailRequest.from_payload(payload)
            except Exception as exc:
                prod.send('callosum_responses', DetailResponse(qid=str(payload.get("qid", "")), error=str(exc)).to_payload()); prod.flush()
                continue
            detail = await right.deepen(
                detail_req.qid,
                detail_req.question,
                detail_req.draft_summary,
                mem,
            )
            resp = DetailResponse(
                qid=detail_req.qid,
                notes_summary=detail.get("notes_sum"),
                confidence=detail.get("confidence_r"),
            )
            prod.send('callosum_responses', resp.to_payload()); prod.flush()
        elif payload.get("type")=="ASK_LEAD":
            qid = payload.get("qid")
            lead = await right.generate_lead(
                payload.get("question", ""),
                payload.get("context", ""),
                temperature=float(payload.get("temperature", 0.85)),
            )
            prod.send('callosum_responses', {"qid": qid, "lead_notes": lead}); prod.flush()

async def run_mqtt():
    from core.callosum_mqtt import CallosumMQTT
    import paho.mqtt.client as mqtt
    callosum = CallosumMQTT()
    mem = SharedMemory(); right = RightBrainModel()
    print("Right worker (MQTT) listening...")
    import threading
    client = mqtt.Client()
    def on_connect(c, u, f, rc):
        c.subscribe('callosum/requests')
    def on_message(c, u, msg):
        try:
            payload = json.loads(msg.payload.decode('utf-8'))
            if payload.get("type")=="ASK_DETAIL":
                asyncio.run_coroutine_threadsafe(handle_mqtt_request(c, right, mem, payload), asyncio.get_event_loop())
            elif payload.get("type")=="ASK_LEAD":
                qid = payload.get("qid")
                asyncio.run_coroutine_threadsafe(handle_mqtt_lead(c, right, qid, payload), asyncio.get_event_loop())
        except Exception:
            pass
    async def handle_mqtt_request(client, right, mem, payload):
        try:
            detail_req = DetailRequest.from_payload(payload)
        except Exception as exc:
            client.publish('callosum/responses', json.dumps(DetailResponse(qid=str(payload.get("qid", "")), error=str(exc)).to_payload()))
            return
        detail = await right.deepen(
            detail_req.qid,
            detail_req.question,
            detail_req.draft_summary,
            mem,
        )
        resp = DetailResponse(
            qid=detail_req.qid,
            notes_summary=detail.get("notes_sum"),
            confidence=detail.get("confidence_r"),
        )
        client.publish('callosum/responses', json.dumps(resp.to_payload()))

    async def handle_mqtt_lead(client, right, qid, payload):
        lead = await right.generate_lead(
            payload.get("question", ""),
            payload.get("context", ""),
            temperature=float(payload.get("temperature", 0.85)),
        )
        client.publish('callosum/responses', json.dumps({"qid": qid, "lead_notes": lead}))
    client.on_connect = on_connect; client.on_message = on_message
    client.connect('localhost', 1883); client.loop_forever()

def main():
    backend = os.environ.get("CALLOSUM_BACKEND","kafka")
    if backend == "kafka":
        asyncio.run(run_kafka())
    elif backend == "mqtt":
        asyncio.run(run_mqtt())
    else:
        print("Set CALLOSUM_BACKEND=kafka or mqtt")

if __name__ == "__main__": main()
