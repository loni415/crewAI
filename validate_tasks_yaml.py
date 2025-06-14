import yaml

with open("/Users/lukasfiller/dev/crewAI/may29_xjp/src/may20_xjp_2/config/tasks.yaml") as f:
    tasks_config = yaml.safe_load(f)

errors = []
for task_name, task_value in tasks_config.items():
    if isinstance(task_value, str):
        errors.append(f"Task '{task_name}' is a string, not a dictionary.")

if errors:
    print("Validation errors found in tasks.yaml:")
    for err in errors:
        print("  -", err)
else:
    print("All tasks are dictionaries. tasks.yaml is valid!")
