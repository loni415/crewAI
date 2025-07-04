#!/usr/bin/env python
import sys
import warnings
import json
from datetime import datetime

from jun22_xjp.crew import Jun22Xjp

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information


def run():
    """Run the crew."""
    inputs = {
        "input_event_description": "Japan seizes an active chinese fishing vessel in the Senkaku Islands, within the EEZ of Japan",
        "current_year": str(datetime.now().year),
        "aggression_level": "",
        "domain_emphasis": "",
    }

    try:
        crew_output = Jun22Xjp().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")

    if crew_output.json_dict:
        print(f"JSON Output: {json.dumps(crew_output.json_dict, indent=2)}")
    if crew_output.pydantic:
        print(f"Pydantic Output: {crew_output.pydantic}")
    print(f"Tasks Output: {crew_output.tasks_output}")
    print(f"Token Usage: {crew_output.token_usage}")
    print(crew_output)

def train():
    """Train the crew for a given number of iterations."""
    inputs = {
        "input_event_description": "Taiwan is caught conducting cyber espionage against PRC",
        "current_year": str(datetime.now().year),
        "aggression_level": "",
        "domain_emphasis": "",
    }

    try:
        crew_output = Jun22Xjp().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")

    if crew_output.json_dict:
        print(f"JSON Output: {json.dumps(crew_output.json_dict, indent=2)}")
    if crew_output.pydantic:
        print(f"Pydantic Output: {crew_output.pydantic}")
    print(f"Tasks Output: {crew_output.tasks_output}")
    print(f"Token Usage: {crew_output.token_usage}")


def replay():
    """Replay the crew execution from a specific task."""
    try:
        Jun22Xjp().crew().replay(task_id=sys.argv[1])
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")


def test():
    """Test the crew execution and returns the results."""
    inputs = {
        "topic": "AI LLMs",
        "current_year": str(datetime.now().year),
    }

    try:
        Jun22Xjp().crew().test(
            n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs
        )
    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")
