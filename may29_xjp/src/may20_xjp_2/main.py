#!/usr/bin/env python
import sys
import warnings
import json
from datetime import datetime

#from may29_xjp.src.may20_xjp_2.crew_alt import May20Xjp2
from may20_xjp_2.crew import May20Xjp2

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    inputs = {
        'input_event_description': 'Taiwan is caught conducting cyber espionage against PRC',
        'current_year': str(datetime.now().year),
        'aggression_level': '', # Default to empty (optional)
        'domain_emphasis': ''    # Default to empty (Optional)
    }

    try:
        crew_output = May20Xjp2().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")
    # Accessing the crew output
    if crew_output.json_dict:
        print(f"JSON Output: {json.dumps(crew_output.json_dict, indent=2)}")
    if crew_output.pydantic:
        print(f"Pydantic Output: {crew_output.pydantic}")
    print(f"Tasks Output: {crew_output.tasks_output}")
    print(f"Token Usage: {crew_output.token_usage}")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        'input_event_description': 'Taiwan is caught conducting cyber espionage against PRC',
        'current_year': str(datetime.now().year),
        'aggression_level': '', # Default to empty (optional)
        'domain_emphasis': ''    # Default to empty (Optional)
    }

    try:
        crew_output = May20Xjp2().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")
    # Accessing the crew output
    if crew_output.json_dict:
        print(f"JSON Output: {json.dumps(crew_output.json_dict, indent=2)}")
    if crew_output.pydantic:
        print(f"Pydantic Output: {crew_output.pydantic}")
    print(f"Tasks Output: {crew_output.tasks_output}")
    print(f"Token Usage: {crew_output.token_usage}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        May20Xjp2().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "topic": "AI LLMs",
        "current_year": str(datetime.now().year)
    }

    try:
        May20Xjp2().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")
