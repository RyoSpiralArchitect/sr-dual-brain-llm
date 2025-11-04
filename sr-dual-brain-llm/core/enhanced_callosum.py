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

"""Enhanced Corpus Callosum with neurotransmitter-based filtering.

This module wraps the standard Callosum with GABA-based information filtering,
allowing the system to suppress noise and low-priority information before
transmission between hemispheres.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .callosum import Callosum
from .neurotransmitter_modulator import (
    NeurotransmitterModulator,
    InformationFilterResult,
    NeurotransmitterPulse,
)


class EnhancedCallosum:
    """Corpus Callosum with neurotransmitter modulation.
    
    Wraps the standard Callosum to provide:
    - GABA-based information filtering
    - Priority-based transmission
    - Neurotransmitter pulse tracking
    """
    
    def __init__(
        self,
        base_callosum: Optional[Callosum] = None,
        slot_ms: int = 250,
        enable_filtering: bool = True,
    ) -> None:
        self.base_callosum = base_callosum or Callosum(slot_ms=slot_ms)
        self.enable_filtering = enable_filtering
        self.modulator = NeurotransmitterModulator()
        
        # Statistics
        self.total_requests = 0
        self.filtered_requests = 0
        self.transmitted_requests = 0
    
    async def ask_detail(
        self,
        payload: Dict[str, Any],
        timeout_ms: int = 3000,
        *,
        priority: Optional[float] = None,
        novelty: Optional[float] = None,
        task_relevance: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Ask for detail with neurotransmitter filtering.
        
        Args:
            payload: Request payload
            timeout_ms: Timeout in milliseconds
            priority: Priority score (0.0 to 1.0), auto-calculated if None
            novelty: Novelty score (0.0 to 1.0), auto-calculated if None
            task_relevance: Task relevance (0.0 to 1.0), defaults to 1.0
            
        Returns:
            Response payload with neurotransmitter metadata
        """
        self.total_requests += 1
        
        # Auto-calculate priority if not provided
        if priority is None:
            priority = self._estimate_priority(payload)
        if novelty is None:
            novelty = self._estimate_novelty(payload)
        if task_relevance is None:
            task_relevance = 1.0  # Default to fully relevant
        
        # Apply neurotransmitter filtering
        filter_result = None
        if self.enable_filtering:
            filter_result, pulses = self.modulator.process_information_transfer(
                priority=priority,
                novelty=novelty,
                task_relevance=task_relevance,
                current_focus=payload.get("focus"),
            )
            
            # Add neurotransmitter metadata to payload
            payload["neurotransmitter_pulses"] = [p.to_payload() for p in pulses]
            payload["filter_result"] = filter_result.to_payload()
            
            # Check if transmission should be blocked
            if not filter_result.should_transmit:
                self.filtered_requests += 1
                # Return filtered response without actual transmission
                return {
                    "qid": payload.get("qid", ""),
                    "filtered": True,
                    "reason": filter_result.reason,
                    "suppression_strength": filter_result.suppression_strength,
                    "neurotransmitter_pulses": [p.to_payload() for p in pulses],
                }
        
        self.transmitted_requests += 1
        
        # Transmit through base callosum
        response = await self.base_callosum.ask_detail(payload, timeout_ms)
        
        # Add neurotransmitter metadata to response
        if filter_result:
            response["filter_result"] = filter_result.to_payload()
        
        return response
    
    async def publish_response(self, qid: str, response: Dict[str, Any]):
        """Publish response through base callosum."""
        await self.base_callosum.publish_response(qid, response)
    
    async def recv_request(self) -> Dict[str, Any]:
        """Receive request from base callosum."""
        return await self.base_callosum.recv_request()
    
    def _estimate_priority(self, payload: Dict[str, Any]) -> float:
        """Estimate priority from payload content.
        
        Higher priority for:
        - Explicit priority markers
        - Error/critical keywords
        - User-facing requests
        """
        # Check explicit priority
        if "priority" in payload:
            return float(payload["priority"])
        
        # Analyze content
        content = str(payload.get("content", "")).lower()
        
        # Critical keywords increase priority
        critical_keywords = ["error", "critical", "urgent", "important", "必須", "緊急"]
        priority = 0.5  # Base priority
        
        for keyword in critical_keywords:
            if keyword in content:
                priority = min(1.0, priority + 0.2)
        
        # User-facing requests have higher priority
        if payload.get("user_facing", False):
            priority = min(1.0, priority + 0.2)
        
        return priority
    
    def _estimate_novelty(self, payload: Dict[str, Any]) -> float:
        """Estimate novelty from payload content.
        
        Higher novelty for:
        - New topics
        - Unexpected requests
        - First-time patterns
        """
        # Check explicit novelty
        if "novelty" in payload:
            return float(payload["novelty"])
        
        # Simple heuristic: check for novelty markers
        content = str(payload.get("content", "")).lower()
        novelty_keywords = ["new", "novel", "unprecedented", "unexpected", "新しい", "初めて"]
        
        novelty = 0.5  # Base novelty
        for keyword in novelty_keywords:
            if keyword in content:
                novelty = min(1.0, novelty + 0.15)
        
        return novelty
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get transmission statistics."""
        filter_rate = 0.0
        if self.total_requests > 0:
            filter_rate = self.filtered_requests / self.total_requests
        
        return {
            "total_requests": self.total_requests,
            "filtered_requests": self.filtered_requests,
            "transmitted_requests": self.transmitted_requests,
            "filter_rate": filter_rate,
            "modulator_state": self.modulator.to_payload(),
        }
    
    def to_payload(self) -> Dict[str, object]:
        """Export callosum state for telemetry."""
        return {
            "enable_filtering": self.enable_filtering,
            "statistics": self.get_statistics(),
        }
