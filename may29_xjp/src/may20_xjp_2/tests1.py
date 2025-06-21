import yaml

yaml_snippet = """
ideological_perception_task:
  description: >
    Review {context.assess_signaling_and_recommend_strategic_path_task.output}
    to extract pivotal ideological themes and doctrinal cues.
  expected_output: |
    memo: "Concise (<=300 words) Ideological Compliance & Leverage Memo."
    key_frames:
      - "Ideological frame 1"
      - "Ideological frame 2"
  context:
    - assess_signaling_and_recommend_strategic_path_task
  agent: CCPIdeologicalAnalyst
"""

try:
    data = yaml.safe_load(yaml_snippet)
    # Check if expected_output is parsed as a dict:
    expected_output = data["ideological_perception_task"]["expected_output"]
    if isinstance(expected_output, dict):
        print("expected_output is a valid mapping:", expected_output)
    else:
        print("expected_output is not a mapping, got:", type(expected_output))
except yaml.YAMLError as e:
    print("YAML formatting error:", e)
