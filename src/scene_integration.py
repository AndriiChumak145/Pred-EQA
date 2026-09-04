import numpy as np
import logging
from typing import Dict
from src.long_term_memory import TextLongTermMemory


class SceneIntegration:
    """New scene integration class, integrating text long-term memory and planning."""

    def __init__(self, scene, vlm_model=None, vlm_processor=None):
        self.scene = scene

        # Initialize new long-term memory and planning system
        self.long_term_memory = TextLongTermMemory()


        # Exploration history tracking
        self.current_step = 0
        self.question = ""
        self.target_objects = []
        self.exploration_path = []  # Record complete exploration path


    def record_structured_agent_output(self, step: int, agent_type: str, structured_output: Dict,
                                     raw_response: str, position: np.ndarray):
        """Record structured agent output."""
        # Record structured output using long-term memory
        self.long_term_memory.record_structured_agent_output(step, agent_type, structured_output, raw_response, position)

        logging.info(f"Recorded structured output for {agent_type} at step {step}")
