import os
import yaml
from typing import List
from pydantic import BaseModel, Field

from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.memory import LongTermMemory, ShortTermMemory, EntityMemory
from crewai.memory.storage.rag_storage import RAGStorage
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
from crewai.knowledge.knowledge import Knowledge
from crewai.knowledge.knowledge_config import KnowledgeConfig  # change src/crewai/knowledge/knowledge_config.py
from crewai.knowledge.source.crew_docling_source import CrewDoclingSource
import os
import yaml
from typing import List
from pydantic import BaseModel, Field

from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.memory import LongTermMemory, ShortTermMemory, EntityMemory
from crewai.memory.storage.rag_storage import RAGStorage
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage
from crewai.knowledge.knowledge import Knowledge
from crewai.knowledge.knowledge_config import KnowledgeConfig  # change src/crewai/knowledge/knowledge_config.py
from crewai.knowledge.source.crew_docling_source import CrewDoclingSource

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
@CrewBase
class May20Xjp2:
    """May20Xjp2 crew"""

    agents_config_path = os.path.join(_CURRENT_DIR, 'config/agents.yaml')
    tasks_config_path = os.path.join(_CURRENT_DIR, 'config/tasks.yaml')

    def __init__(self):
        # Load the YAML configurations and assign as dicts
        with open(self.agents_config_path, 'r') as f:
            self.agents_config = yaml.safe_load(f)
        with open(self.tasks_config_path, 'r') as f:
            self.tasks_config = yaml.safe_load(f)


        # DEBUG: Check the type and content of the problematic task config
        ipt_config = self.tasks_config.get('ideological_perception_task')
        print(f"DEBUG: Type of ideological_perception_task config: {type(ipt_config)}")
        print(f"DEBUG: Content of ideological_perception_task config: {ipt_config}")
        # END DEBUG

        # Initialize LLM once for the crew instance
        self.llm = LLM(
            model="ollama/mixtral:8x22b-instruct-v0.1-q2_K",
            base_url="http://localhost:11434"
        )

    @agent
    def CCPStrategicPolicyAdvisor(self) -> Agent:
        return Agent(**self.agents_config['CCPStrategicPolicyAdvisor'], llm=self.llm, verbose=True)

    @agent
    def EconomicAndTechImpactAnalystAgent(self) -> Agent:
        return Agent(**self.agents_config['EconomicAndTechImpactAnalystAgent'], llm=self.llm, verbose=True)

    @agent
    def ForeignPolicyEventAnalystAgent(self) -> Agent:
        return Agent(**self.agents_config['ForeignPolicyEventAnalystAgent'], llm=self.llm, verbose=True)

    @agent
    def StrategicSignalingAssessmentAgent(self) -> Agent:
        return Agent(**self.agents_config['StrategicSignalingAssessmentAgent'], llm=self.llm, verbose=True)

    @agent
    def PLAOptionsStrategistAgent(self) -> Agent:
        return Agent(**self.agents_config['PLAOptionsStrategistAgent'], llm=self.llm, verbose=True)

    @agent
    def MFADiplomaticStrategistAgent(self) -> Agent:
        return Agent(**self.agents_config['MFADiplomaticStrategistAgent'], llm=self.llm, verbose=True)

    @agent
    def StrategicNarrativeAndInfluenceAgent(self) -> Agent:
        return Agent(**self.agents_config['StrategicNarrativeAndInfluenceAgent'], llm=self.llm, verbose=True)

    @agent
    def ResponseSynthesizerAgent(self) -> Agent:
        return Agent(**self.agents_config['ResponseSynthesizerAgent'], llm=self.llm, verbose=True)

    @agent
    def CCPIdeologicalAnalyst(self) -> Agent:
        return Agent(**self.agents_config['CCPIdeologicalAnalyst'], llm=self.llm, verbose=True)

    @agent
    def DomesticSentimentStabilityAnalyst(self) -> Agent:
        return Agent(**self.agents_config['DomesticSentimentStabilityAnalyst'], llm=self.llm, verbose=True)

    @agent
    def HistoricalPrecedentAnalyst(self) -> Agent:
        return Agent(**self.agents_config['HistoricalPrecedentAnalyst'], llm=self.llm, verbose=True)

    @agent
    def ContextCuratorAgent(self) -> Agent:
        return Agent(**self.agents_config['ContextCuratorAgent'], llm=self.llm, verbose=True)

    # --- TASK DEFINITIONS ---
    # This explicit wiring is the robust way to ensure your crew works,
    # bypassing any potential bugs in the framework's automatic wiring.

    @task
    def analyze_event_task(self) -> Task:
        return Task(config=self.tasks_config['analyze_event_task'], agent=self.ForeignPolicyEventAnalystAgent())

    @task
    def assess_economic_tech_impact_task(self) -> Task:
        return Task(config=self.tasks_config['assess_economic_tech_impact_task'], agent=self.EconomicAndTechImpactAnalystAgent())

    @task
    def historical_context_task(self) -> Task:
        return Task(config=self.tasks_config['historical_context_task'], agent=self.HistoricalPrecedentAnalyst())

    @task
    def internal_impact_narrative_task(self) -> Task:
        return Task(config=self.tasks_config['internal_impact_narrative_task'], agent=self.DomesticSentimentStabilityAnalyst())

    @task
    def develop_active_strategic_postures_task(self) -> Task:
        return Task(config=self.tasks_config['develop_active_strategic_postures_task'], agent=self.CCPStrategicPolicyAdvisor())

    @task
    def assess_signaling_and_recommend_strategic_path_task(self) -> Task:
        return Task(config=self.tasks_config['assess_signaling_and_recommend_strategic_path_task'], agent=self.StrategicSignalingAssessmentAgent())

    @task
    def generate_active_pla_options_task(self) -> Task:
        return Task(config=self.tasks_config['generate_active_pla_options_task'], agent=self.PLAOptionsStrategistAgent())

    @task
    def ideological_perception_task(self) -> Task:
        task_config_to_pass = self.tasks_config['ideological_perception_task']
        print(f"DEBUG INSIDE TASK METHOD: Type of task_config_to_pass for ideological_perception_task: {type(task_config_to_pass)}")
        print(f"DEBUG INSIDE TASK METHOD: Content: {task_config_to_pass}")
        return Task(config=task_config_to_pass, agent=self.CCPIdeologicalAnalyst())

    @task
    def develop_active_diplomatic_strategy_task(self) -> Task:
        return Task(config=self.tasks_config['develop_active_diplomatic_strategy_task'], agent=self.MFADiplomaticStrategistAgent())

    @task
    def curate_context_digest_task(self) -> Task:
        return Task(config=self.tasks_config['curate_context_digest_task'], agent=self.ContextCuratorAgent())

    @task
    def develop_strategic_communication_plan_task(self) -> Task:
        return Task(config=self.tasks_config['develop_strategic_communication_plan_task'], agent=self.StrategicNarrativeAndInfluenceAgent())

    @task
    def format_final_response_task(self) -> Task:
        return Task(config=self.tasks_config['format_final_response_task'], agent=self.ResponseSynthesizerAgent())

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
