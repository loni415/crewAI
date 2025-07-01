# 22jun_xjp/src/22jun_xjp/crew.py

import os
import yaml
from typing import List

from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent

from crewai.memory import LongTermMemory, ShortTermMemory, EntityMemory
from crewai.memory.storage.rag_storage import RAGStorage

from crewai.memory.long_term.long_term_memory import LongTermMemory
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage


from crewai.knowledge.source.crew_docling_source import CrewDoclingSource


llm = LLM(
    model="ollama/phi4-reasoning:14B-plus-fp16",
    base_url="http://localhost:11434"
)

embedding_config={
    "provider": "ollama",
    "config": {
        "model": "bge-m3",
    }
}

content_source_planner = CrewDoclingSource(
    file_paths=[
        "PRC_GrayZone_Planner.md"
    ],
)

content_source_instruct = CrewDoclingSource(
    file_paths=[
        "XJP_Instructions_1.md"
    ],
)

# Set up long-term memory (can customize db_path)
long_term_memory = LongTermMemory(
    storage=LTMSQLiteStorage(db_path="./long_term_memory.db")
)

entity_memory = EntityMemory(
    storage=RAGStorage(type="entities", embedder_config=embedding_config)
    )

@CrewBase
class Jun22Xjp:
    """Jun22Xjp crew"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    agents: List[BaseAgent]
    tasks: List[Task]


    #agents_config = 'src/jun22_xjp/config/agents.yaml'
    #tasks_config = 'src/jun22_xjp/config/tasks.yaml'

    # --- AGENT DEFINITIONS ---
    @agent
    def CCPStrategicPolicyAdvisor(self) -> Agent:
      return Agent(
        config=self.agents_config['CCPStrategicPolicyAdvisor'], # type: ignore[index]
        verbose=True,
        max_iter=2,
        max_reasoning_attempts=1,
        llm=llm,
      )

    @agent
    def EconomicAndTechImpactAnalystAgent(self) -> Agent:
        return Agent(
            config=self.agents_config['EconomicAndTechImpactAnalystAgent'], # type: ignore[index]
            llm=llm,
            verbose=True,
            max_iter=2,
            max_reasoning_attempts=1,
        )

    @agent
    def ForeignPolicyEventAnalystAgent(self) -> Agent:
        return Agent(
            config=self.agents_config['ForeignPolicyEventAnalystAgent'], # type: ignore[index]
            llm=llm,
            verbose=True,
            max_iter=2,
            max_reasoning_attempts=1,
        )

    @agent
    def StrategicSignalingAssessmentAgent(self) -> Agent:
        return Agent(
            config=self.agents_config['StrategicSignalingAssessmentAgent'], # type: ignore[index]
            llm=llm,
            verbose=True,
            max_iter=2,
            max_reasoning_attempts=1,
        )

    @agent
    def PLAOptionsStrategistAgent(self) -> Agent:
        return Agent(
            config=self.agents_config['PLAOptionsStrategistAgent'], # type: ignore[index]
            llm=llm,
            verbose=True,
            max_iter=2,
            max_reasoning_attempts=1,
        )

    @agent
    def MFADiplomaticStrategistAgent(self) -> Agent:
        return Agent(
            config=self.agents_config['MFADiplomaticStrategistAgent'], # type: ignore[index]
            llm=llm,
            verbose=True,
            max_iter=2,
            max_reasoning_attempts=1,
        )

    @agent
    def StrategicNarrativeAndInfluenceAgent(self) -> Agent:
        return Agent(
            config=self.agents_config['StrategicNarrativeAndInfluenceAgent'], # type: ignore[index]
            llm=llm,
            verbose=True,
            max_iter=2,
            max_reasoning_attempts=1,
        )

    @agent
    def ResponseSynthesizerAgent(self) -> Agent:
        return Agent(
            config=self.agents_config['ResponseSynthesizerAgent'], # type: ignore[index]
            llm=llm,
            verbose=True,
            max_iter=2,
            max_reasoning_attempts=1,
        )
    @agent
    def CCPIdeologicalAnalyst(self) -> Agent:
        return Agent(
            config=self.agents_config['CCPIdeologicalAnalyst'], # type: ignore[index]
            llm=llm,
            verbose=True,
            max_iter=2,
            max_reasoning_attempts=1,
        )

    @agent
    def DomesticSentimentStabilityAnalyst(self) -> Agent:
        return Agent(
            config=self.agents_config['DomesticSentimentStabilityAnalyst'], # type: ignore[index]
            llm=llm,
            verbose=True,
            max_iter=2,
            max_reasoning_attempts=1,
        )

    @agent
    def HistoricalPrecedentAnalyst(self) -> Agent:
        return Agent(
            config=self.agents_config['HistoricalPrecedentAnalyst'], # type: ignore[index]
            llm=llm,
            verbose=True,
            max_iter=2,
            max_reasoning_attempts=1,
        )

    @agent
    def ContextCuratorAgent(self) -> Agent:
        return Agent(
            config=self.agents_config['ContextCuratorAgent'], # type: ignore[index]
            llm=llm,
            verbose=True,
            max_iter=2,
            max_reasoning_attempts=1,
        )

    # --- TASK DEFINITIONS ---
    @task
    def analyze_event_task(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_event_task'], # type: ignore[index]
            agent=self.ForeignPolicyEventAnalystAgent(),
        )

    @task
    def assess_economic_tech_impact_task(self) -> Task:
        return Task(
            config=self.tasks_config['assess_economic_tech_impact_task'], # type: ignore[index]
            agent=self.EconomicAndTechImpactAnalystAgent(),
        )

    @task
    def historical_context_task(self) -> Task:
        return Task(
            config=self.tasks_config['historical_context_task'], # type: ignore[index]
            agent=self.HistoricalPrecedentAnalyst(),
        )

    @task
    def internal_impact_narrative_task(self) -> Task:
        return Task(
            config=self.tasks_config['internal_impact_narrative_task'], # type: ignore[index]
            agent=self.DomesticSentimentStabilityAnalyst(),
        )

    @task
    def develop_active_strategic_postures_task(self) -> Task:
        return Task(
            config=self.tasks_config['develop_active_strategic_postures_task'], # type: ignore[index]
            agent=self.CCPStrategicPolicyAdvisor(),
        )

    @task
    def assess_signaling_and_recommend_strategic_path_task(self) -> Task:
        return Task(
            config=self.tasks_config[
                'assess_signaling_and_recommend_strategic_path_task'], # type: ignore[index]
            agent=self.StrategicSignalingAssessmentAgent(),
        )

    @task
    def generate_active_pla_options_task(self) -> Task:
        return Task(
            config=self.tasks_config['generate_active_pla_options_task'], # type: ignore[index]
            agent=self.PLAOptionsStrategistAgent(),
        )

    @task
    def ideological_perception_task(self) -> Task:
        return Task(
            config=self.tasks_config['ideological_perception_task'], # type: ignore[index]
            agent=self.CCPIdeologicalAnalyst(),
        )

    @task
    def develop_active_diplomatic_strategy_task(self) -> Task:
        return Task(
            config=self.tasks_config['develop_active_diplomatic_strategy_task'], # type: ignore[index]
            agent=self.MFADiplomaticStrategistAgent(),
        )

    @task
    def curate_context_digest_task(self) -> Task:
        return Task(
            config=self.tasks_config['curate_context_digest_task'], # type: ignore[index]
            agent=self.ContextCuratorAgent(),
        )

    @task
    def develop_strategic_communication_plan_task(self) -> Task:
        return Task(
            config=self.tasks_config['develop_strategic_communication_plan_task'], # type: ignore[index]
            agent=self.StrategicNarrativeAndInfluenceAgent(),
        )

    @task
    def format_final_response_task(self) -> Task:
        return Task(
            config=self.tasks_config['format_final_response_task'], # type: ignore[index]
            agent=self.ResponseSynthesizerAgent(),
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            knowledge_sources=[
                content_source_planner,
                content_source_instruct
            ],
            process=Process.sequential,
            verbose=True,
            memory=False,
            output_log_file="log.txt",
            manager_llm=llm,
            embedder=embedding_config
        )
