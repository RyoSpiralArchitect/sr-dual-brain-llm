"""Tests for enhanced corpus callosum with neurotransmitter filtering."""

import pytest
import asyncio
from core.enhanced_callosum import EnhancedCallosum
from core.callosum import Callosum


@pytest.mark.asyncio
async def test_enhanced_callosum_creation():
    """Test enhanced callosum can be created."""
    callosum = EnhancedCallosum()
    
    assert callosum.enable_filtering is True
    assert callosum.total_requests == 0
    assert callosum.modulator is not None


@pytest.mark.asyncio
async def test_enhanced_callosum_filters_noise():
    """Test enhanced callosum filters low-priority noise."""
    callosum = EnhancedCallosum()
    
    # Low priority request should be filtered
    payload = {
        "content": "test noise",
        "qid": "test1",
    }
    
    response = await callosum.ask_detail(
        payload,
        priority=0.1,
        novelty=0.05,
        task_relevance=0.1,
    )
    
    assert response["filtered"] is True
    assert response["reason"] == "noise_filtered"
    assert callosum.filtered_requests == 1
    assert callosum.transmitted_requests == 0


@pytest.mark.asyncio
async def test_enhanced_callosum_transmits_important():
    """Test enhanced callosum transmits high-priority information."""
    callosum = EnhancedCallosum()
    
    # Create a simple responder
    async def responder():
        request = await callosum.recv_request()
        await callosum.publish_response(
            request["qid"],
            {"status": "success", "result": "processed"}
        )
    
    # Start responder in background
    responder_task = asyncio.create_task(responder())
    
    # High priority request should be transmitted
    payload = {
        "content": "important request",
        "qid": "test2",
    }
    
    response = await callosum.ask_detail(
        payload,
        priority=0.8,
        novelty=0.6,
        task_relevance=0.9,
        timeout_ms=1000,
    )
    
    assert "filtered" not in response or response["filtered"] is False
    assert response["status"] == "success"
    assert callosum.transmitted_requests == 1
    assert callosum.filtered_requests == 0
    
    # Clean up
    responder_task.cancel()
    try:
        await responder_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_enhanced_callosum_auto_priority_detection():
    """Test enhanced callosum auto-detects priority from content."""
    callosum = EnhancedCallosum()
    
    # Create responder
    async def responder():
        while True:
            try:
                request = await callosum.recv_request()
                await callosum.publish_response(
                    request["qid"],
                    {"status": "ok"}
                )
            except asyncio.CancelledError:
                break
    
    responder_task = asyncio.create_task(responder())
    
    # Test critical keyword detection
    payload = {
        "content": "CRITICAL error in system",
        "qid": "test3",
    }
    
    response = await callosum.ask_detail(payload, timeout_ms=1000)
    
    # Should not be filtered due to critical keyword
    assert "filtered" not in response or response["filtered"] is False
    assert callosum.transmitted_requests >= 1
    
    # Clean up
    responder_task.cancel()
    try:
        await responder_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_enhanced_callosum_novelty_detection():
    """Test enhanced callosum detects novelty from content."""
    callosum = EnhancedCallosum()
    
    # Low novelty content
    priority = callosum._estimate_priority({"content": "regular request"})
    novelty = callosum._estimate_novelty({"content": "regular request"})
    
    assert priority == 0.5  # Base priority
    assert novelty == 0.5  # Base novelty
    
    # High novelty content
    novelty = callosum._estimate_novelty({"content": "new unprecedented approach"})
    assert novelty > 0.5


@pytest.mark.asyncio
async def test_enhanced_callosum_filtering_disabled():
    """Test enhanced callosum works without filtering."""
    callosum = EnhancedCallosum(enable_filtering=False)
    
    # Create responder
    async def responder():
        request = await callosum.recv_request()
        await callosum.publish_response(
            request["qid"],
            {"status": "processed"}
        )
    
    responder_task = asyncio.create_task(responder())
    
    # Even low priority should pass through
    payload = {
        "content": "noise",
        "qid": "test4",
    }
    
    response = await callosum.ask_detail(
        payload,
        priority=0.1,
        novelty=0.05,
        task_relevance=0.1,
        timeout_ms=1000,
    )
    
    assert response["status"] == "processed"
    assert callosum.filtered_requests == 0
    assert callosum.transmitted_requests == 1
    
    # Clean up
    responder_task.cancel()
    try:
        await responder_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_enhanced_callosum_statistics():
    """Test enhanced callosum tracks statistics."""
    callosum = EnhancedCallosum()
    
    # Filter one request
    await callosum.ask_detail(
        {"content": "noise", "qid": "s1"},
        priority=0.1,
        novelty=0.05,
        task_relevance=0.1,
    )
    
    stats = callosum.get_statistics()
    
    assert stats["total_requests"] == 1
    assert stats["filtered_requests"] == 1
    assert stats["transmitted_requests"] == 0
    assert stats["filter_rate"] == 1.0


@pytest.mark.asyncio
async def test_enhanced_callosum_payload_export():
    """Test enhanced callosum exports state correctly."""
    callosum = EnhancedCallosum()
    
    # Just filter a request without transmission
    await callosum.ask_detail(
        {"content": "noise", "qid": "p1"},
        priority=0.1,
        novelty=0.05,
        task_relevance=0.1,
    )
    
    payload = callosum.to_payload()
    
    assert "enable_filtering" in payload
    assert "statistics" in payload
    assert payload["enable_filtering"] is True


@pytest.mark.asyncio
async def test_enhanced_callosum_with_custom_base():
    """Test enhanced callosum can wrap custom base callosum."""
    base = Callosum(slot_ms=100)
    enhanced = EnhancedCallosum(base_callosum=base)
    
    assert enhanced.base_callosum is base
    assert enhanced.base_callosum.slot_ms == 100


@pytest.mark.asyncio
async def test_enhanced_callosum_neurotransmitter_metadata():
    """Test enhanced callosum adds neurotransmitter metadata to responses."""
    callosum = EnhancedCallosum()
    
    # Create responder
    async def responder():
        request = await callosum.recv_request()
        # Check request has neurotransmitter metadata
        assert "neurotransmitter_pulses" in request
        assert "filter_result" in request
        
        await callosum.publish_response(
            request["qid"],
            {"status": "ok"}
        )
    
    responder_task = asyncio.create_task(responder())
    
    payload = {
        "content": "important task",
        "qid": "test5",
    }
    
    response = await callosum.ask_detail(
        payload,
        priority=0.8,
        novelty=0.6,
        task_relevance=0.9,
        timeout_ms=1000,
    )
    
    # Response should have filter metadata
    assert "filter_result" in response
    
    # Clean up
    responder_task.cancel()
    try:
        await responder_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_enhanced_callosum_priority_markers():
    """Test enhanced callosum detects various priority markers."""
    callosum = EnhancedCallosum()
    
    # Test explicit priority
    priority = callosum._estimate_priority({"priority": 0.9})
    assert priority == 0.9
    
    # Test critical keywords
    priority = callosum._estimate_priority({"content": "URGENT task needed"})
    assert priority > 0.5
    
    # Test user-facing flag
    priority = callosum._estimate_priority({"user_facing": True})
    assert priority > 0.5
    
    # Test Japanese keywords
    priority = callosum._estimate_priority({"content": "緊急のタスク"})
    assert priority > 0.5
