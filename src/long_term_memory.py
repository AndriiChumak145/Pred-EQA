import json
import pickle
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
import os
import re


@dataclass
class TextMemoryEntry:
    id: str
    content: str  # Text description content
    timestamp: datetime = field(default_factory=datetime.now)  # Timestamp
    importance: float = 1.0  # Importance score
    entry_type: str = "general"  # Memory type
    step: int = 0  # Exploration step
    position: Optional[np.ndarray] = None  # Position info
    raw_response: Optional[str] = None  # Preserve raw VLM response
    structured_decision: Optional[Dict[str, Any]] = None  # Structured decision info


class TextLongTermMemory:
    """Optimized text long-term memory system."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.entries: List[TextMemoryEntry] = []
        self.id_counter = 0
        # Memory indexes for fast retrieval
        self.type_index: Dict[str, List[str]] = {}  # type -> entry_ids
        self.step_index: Dict[int, List[str]] = {}  # step -> entry_ids
        self.timestamp_index: List[str] = []  # entry_ids sorted by timestamp

    def add_entry(self, content: str, importance: float = 1.0, entry_type: str = "general",
                  step: int = 0, position: Optional[np.ndarray] = None, 
                  raw_response: Optional[str] = None, 
                  structured_decision: Optional[Dict[str, Any]] = None) -> str:
        """Add a text memory entry."""
        entry_id = f"mem_{self.id_counter}"
        self.id_counter += 1

        entry = TextMemoryEntry(
            id=entry_id,
            content=content,
            importance=importance,
            entry_type=entry_type,
            step=step,
            position=position,
            raw_response=raw_response,
            structured_decision=structured_decision
        )

        self.entries.append(entry)

        # Update indexes
        if entry_type not in self.type_index:
            self.type_index[entry_type] = []
        self.type_index[entry_type].append(entry_id)

        if step not in self.step_index:
            self.step_index[step] = []
        self.step_index[step].append(entry_id)

        # Re-sort timestamp index
        self.timestamp_index = [e.id for e in sorted(self.entries, key=lambda x: x.timestamp)]

        return entry_id


    def record_structured_agent_output(self, step: int, agent_type: str, structured_output: Dict,
                                      raw_response: str, position: Optional[np.ndarray] = None) -> str:
        """Record structured agent output information."""
        # Check if output for the same step and agent_type has already been recorded to prevent duplicates
        existing_entries = self.retrieve_by_step(step, top_k=10)
        for entry in existing_entries:
            if (entry.entry_type == f"{agent_type}_output" and
                entry.step == step and
                entry.content == f"Step {step} - {agent_type.upper()} Output: {structured_output.get('parsed_decision', 'Unknown')}"):
                print(f"Skipping duplicate {agent_type} record, step {step}")
                return entry.id

        # Extract key information
        action = structured_output.get('structured_output', {}).get('action', 'unknown')
        parsed_decision = structured_output.get('parsed_decision', 'Unknown')

        content = f"Step {step} - {agent_type.upper()} Output: {parsed_decision}"
        
        # Build structured decision information
        structured_decision = {
            "agent_type": agent_type,
            "action": action,
            "step": step,
            "position": position.tolist() if position is not None and isinstance(position, np.ndarray) else position,
            "structured_output": structured_output.get('structured_output', {}),
            # Prefer raw_response_summary; fall back to reasoning if absent (compatibility)
            "raw_response_summary": structured_output.get('raw_response_summary') or structured_output.get('reasoning', ''),
        }
        
        # Add agent-type-specific information
        if agent_type == "planner":
            structured_decision.update({
                "target_type": structured_output.get('structured_output', {}).get('target_type', 'unknown'),
                "target_id": structured_output.get('structured_output', {}).get('target_id', None)
            })
        elif agent_type == "answerer":
            answer_text = structured_output.get('structured_output', {}).get('answer_text', '')
            if len(answer_text) > 100:  # Limit answer length
                answer_text = answer_text[:100] + "...[truncated]"
            structured_decision.update({
                "answer_text": answer_text,
                "evidence_snapshot": structured_output.get('structured_output', {}).get('evidence_snapshot', None)
            })
        elif agent_type == "snapshot_manager":
            structured_decision.update({
                "retained_snapshots": structured_output.get('structured_output', {}).get('snapshot_ids', [])
            })
        elif agent_type == "frontier_manager":
            structured_decision.update({
                "retained_frontiers": structured_output.get('structured_output', {}).get('frontier_ids', [])
            })
        elif agent_type == "high_level_planner":
            # Special handling for high_level_planner's todo_list
            todo_list = structured_output.get('todo_list', structured_output.get('structured_output', {}).get('todo_list', []))
            structured_decision.update({
                "todo_list": todo_list,
                "num_tasks": len(todo_list)
            })

        return self.add_entry(
            content=content,
            importance=0.8,
            entry_type=f"{agent_type}_output",
            step=step,
            position=position,
            raw_response=raw_response,  # Preserve full raw response
            structured_decision=structured_decision
        )

    def retrieve_by_type(self, entry_type: str, top_k: int = 5) -> List[TextMemoryEntry]:
        """Retrieve memories by type."""
        # Fast retrieval using index
        entry_ids = self.type_index.get(entry_type, [])
        matching_entries = [entry for entry in self.entries if entry.id in entry_ids]
        matching_entries.sort(key=lambda x: x.importance, reverse=True)
        return matching_entries[:top_k]

    def retrieve_by_step(self, step: int, top_k: int = 5) -> List[TextMemoryEntry]:
        """Retrieve memories by step."""
        entry_ids = self.step_index.get(step, [])
        matching_entries = [entry for entry in self.entries if entry.id in entry_ids]
        matching_entries.sort(key=lambda x: x.timestamp)
        return matching_entries[:top_k]

    def retrieve_by_step_and_type(self, step: int, entry_type: str, top_k: int = 5) -> List[TextMemoryEntry]:
        """Retrieve memories by step and type."""
        # First get all entries for the specified step
        step_entry_ids = self.step_index.get(step, [])
        # Then filter out entries of the specified type
        matching_entries = [entry for entry in self.entries if entry.id in step_entry_ids and entry.entry_type == entry_type]
        matching_entries.sort(key=lambda x: x.timestamp)
        return matching_entries[:top_k]

    def retrieve_by_time(self, minutes_back: int = 60, top_k: int = 5) -> List[TextMemoryEntry]:
        """Retrieve memories by time (most recent)."""
        time_threshold = datetime.now() - timedelta(minutes=minutes_back)
        recent_entries = [entry for entry in self.entries if entry.timestamp >= time_threshold]
        recent_entries.sort(key=lambda x: x.timestamp, reverse=True)
        return recent_entries[:top_k]