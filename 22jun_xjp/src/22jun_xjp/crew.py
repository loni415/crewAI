import os
import yaml

from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


@CrewBase
class May20Xjp2:
    """May20Xjp2 crew"""

    agents_config_path = os.path.join(_CURRENT_DIR, "config/agents.yaml")
    tasks_config_path = os.path.join(_CURRENT_DIR, "config/tasks.yaml")

    def __init__(self) -> None:
        with open(self.agents_config_path, "r", encoding="utf-8") as f:
            self.agents_config = yaml.safe_load(f)
        with open(self.tasks_config_path, "r", encoding="utf-8") as f:
            self.tasks_config = yaml.safe_load(f)

        self.llm = LLM(
            model="ollama/mixtral:8x22b-instruct-v0.1-q2_K",
            base_url="http://localhost:11434",
        )

    # --- AGENT DEFINITIONS ---
    @agent
    def CCPStrategicPolicyAdvisor(self) -> Agent:
        return Agent(
            **self.agents_config["CCPStrategicPolicyAdvisor"],
            llm=self.llm,
            verbose=True,
        )

    @agent
    def EconomicAndTechImpactAnalystAgent(self) -> Agent:
        return Agent(
            **self.agents_config["EconomicAndTechImpactAnalystAgent"],
            llm=self.llm,
            verbose=True,
        )

    @agent
    def ForeignPolicyEventAnalystAgent(self) -> Agent:
        return Agent(
            **self.agents_config["ForeignPolicyEventAnalystAgent"],
            llm=self.llm,
            verbose=True,
        )

    @agent
    def StrategicSignalingAssessmentAgent(self) -> Agent:
        return Agent(
            **self.agents_config["StrategicSignalingAssessmentAgent"],
            llm=self.llm,
            verbose=True,
        )

    @agent
    def PLAOptionsStrategistAgent(self) -> Agent:
        return Agent(
            **self.agents_config["PLAOptionsStrategistAgent"],
            llm=self.llm,
            verbose=True,
        )

    @agent
    def MFADiplomaticStrategistAgent(self) -> Agent:
        return Agent(
            **self.agents_config["MFADiplomaticStrategistAgent"],
            llm=self.llm,
            verbose=True,
        )

    @agent
    def StrategicNarrativeAndInfluenceAgent(self) -> Agent:
        return Agent(
            **self.agents_config["StrategicNarrativeAndInfluenceAgent"],
            llm=self.llm,
            verbose=True,
        )

    @agent
    def ResponseSynthesizerAgent(self) -> Agent:
        return Agent(
            **self.agents_config["ResponseSynthesizerAgent"],
            llm=self.llm,
            verbose=True,
        )

    @agent
    def CCPIdeologicalAnalyst(self) -> Agent:
        return Agent(
            **self.agents_config["CCPIdeologicalAnalyst"],
            llm=self.llm,
            verbose=True,
        )

    @agent
    def DomesticSentimentStabilityAnalyst(self) -> Agent:
        return Agent(
            **self.agents_config["DomesticSentimentStabilityAnalyst"],
            llm=self.llm,
            verbose=True,
        )

    @agent
    def HistoricalPrecedentAnalyst(self) -> Agent:
        return Agent(
            **self.agents_config["HistoricalPrecedentAnalyst"],
            llm=self.llm,
            verbose=True,
        )

    @agent
    def ContextCuratorAgent(self) -> Agent:
        return Agent(
            **self.agents_config["ContextCuratorAgent"],
            llm=self.llm,
            verbose=True,
        )

    # --- TASK DEFINITIONS ---
    @task
    def analyze_event_task(self) -> Task:
        return Task(
            config=self.tasks_config["analyze_event_task"],
            agent=self.ForeignPolicyEventAnalystAgent(),
        )

    @task
    def assess_economic_tech_impact_task(self) -> Task:
        return Task(
            config=self.tasks_config["assess_economic_tech_impact_task"],
            agent=self.EconomicAndTechImpactAnalystAgent(),
        )

    @task
    def historical_context_task(self) -> Task:
        return Task(
            config=self.tasks_config["historical_context_task"],
            agent=self.HistoricalPrecedentAnalyst(),
        )

    @task
    def internal_impact_narrative_task(self) -> Task:
        return Task(
            config=self.tasks_config["internal_impact_narrative_task"],
            agent=self.DomesticSentimentStabilityAnalyst(),
        )

    @task
    def develop_active_strategic_postures_task(self) -> Task:
        return Task(
            config=self.tasks_config["develop_active_strategic_postures_task"],
            agent=self.CCPStrategicPolicyAdvisor(),
        )

    @task
    def assess_signaling_and_recommend_strategic_path_task(self) -> Task:
        return Task(
            config=self.tasks_config[
                "assess_signaling_and_recommend_strategic_path_task"
            ],
            agent=self.StrategicSignalingAssessmentAgent(),
        )

    @task
    def generate_active_pla_options_task(self) -> Task:
        return Task(
            config=self.tasks_config["generate_active_pla_options_task"],
            agent=self.PLAOptionsStrategistAgent(),
        )

    @task
    def ideological_perception_task(self) -> Task:
        return Task(
            config=self.tasks_config["ideological_perception_task"],
            agent=self.CCPIdeologicalAnalyst(),
        )

    @task
    def develop_active_diplomatic_strategy_task(self) -> Task:
        return Task(
            config=self.tasks_config["develop_active_diplomatic_strategy_task"],
            agent=self.MFADiplomaticStrategistAgent(),
        )

    @task
    def curate_context_digest_task(self) -> Task:
        return Task(
            config=self.tasks_config["curate_context_digest_task"],
            agent=self.ContextCuratorAgent(),
        )

    @task
    def develop_strategic_communication_plan_task(self) -> Task:
        return Task(
            config=self.tasks_config["develop_strategic_communication_plan_task"],
            agent=self.StrategicNarrativeAndInfluenceAgent(),
        )

    @task
    def format_final_response_task(self) -> Task:
        return Task(
            config=self.tasks_config["format_final_response_task"],
            agent=self.ResponseSynthesizerAgent(),
        )

    @crew
    def crew(self) -> Crew:
        """Creates the May20Xjp2 crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            memory=True,
            manager_llm=self.llm,
        )
