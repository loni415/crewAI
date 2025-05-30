from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai.memory import LongTermMemory, ShortTermMemory, EntityMemory
from crewai.memory.storage.rag_storage import RAGStorage
from pydantic import BaseModel, Field

from crewai.knowledge.source.crew_docling_source import CrewDoclingSource
from crewai.knowledge.knowledge_config import KnowledgeConfig
#from crewai.memory.long_term.long_term_memory import LongTermMemory
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage

llm = LLM(
    model="ollama/qwen3:32b-q8_0",
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

knowledge_config = KnowledgeConfig(results_limit=10, score_threshold=0.5)

# Set up long-term memory (can customize db_path)
long_term_memory = LongTermMemory(
    storage=LTMSQLiteStorage(db_path="./long_term_memory.db")
)

entity_memory = EntityMemory(
    storage=RAGStorage(type="entities", embedder_config=embedding_config)
    )

@CrewBase
class May20Xjp2():
    """May20Xjp2 crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def CCPStrategicPolicyAdvisor(self) -> Agent:
        return Agent(
            config=self.agents_config['CCPStrategicPolicyAdvisor'], # type: ignore[index]
            verbose=True,
            reasoning=True,
            max_reasoning_attempts=3,
            memory=True,
            llm=llm,
            knowledge_sources=[
                content_source_planner
            ],
        )
    @agent
    def EconomicAndTechImpactAnalystAgent(self) -> Agent:
        return Agent(
            config=self.agents_config['EconomicAndTechImpactAnalystAgent'], # type: ignore[index]
            verbose=True,
            reasoning=True,
            max_reasoning_attempts=3,
            memory=True,
            llm=llm,
            knowledge_sources=[
                content_source_planner
            ],
        )
    @agent
    def ForeignPolicyEventAnalystAgent(self) -> Agent:
        return Agent(
            config=self.agents_config['ForeignPolicyEventAnalystAgent'], # type: ignore[index]
            verbose=True,
            reasoning=True,
            max_reasoning_attempts=3,
            memory=True,
            llm=llm,
            knowledge_sources=[
                content_source_planner
            ],
        )
    @agent
    def StrategicSignalingAssessmentAgent(self) -> Agent:
        return Agent(
            config=self.agents_config['StrategicSignalingAssessmentAgent'], # type: ignore[index]
            verbose=True,
            reasoning=True,
            max_reasoning_attempts=3,
            memory=True,
            llm=llm,
            knowledge_sources=[
                content_source_planner
            ],
        )

    @agent
    def PLAOptionsStrategistAgent(self) -> Agent:
        return Agent(
            config=self.agents_config['PLAOptionsStrategistAgent'], # type: ignore[index]
            verbose=True,
            reasoning=True,
            max_reasoning_attempts=3,
            memory=True,
            llm=llm,
            knowledge_sources=[
                content_source_planner
            ],
        )

    @agent
    def MFADiplomaticStrategistAgent(self) -> Agent:
        return Agent(
            config=self.agents_config['MFADiplomaticStrategistAgent'], # type: ignore[index]
            verbose=True,
            reasoning=True,
            max_reasoning_attempts=3,
            memory=True,
            llm=llm,
            knowledge_sources=[
                content_source_planner
            ],
        )

    @agent
    def StrategicNarrativeAndInfluenceAgent(self) -> Agent:
        return Agent(
            config=self.agents_config['StrategicNarrativeAndInfluenceAgent'], # type: ignore[index]
            verbose=True,
            reasoning=True,
            max_reasoning_attempts=3,
            memory=True,
            llm=llm,
            knowledge_sources=[
                content_source_planner
            ],
        )

    @agent
    def ResponseSynthesizerAgent(self) -> Agent:
        return Agent(
            config=self.agents_config['ResponseSynthesizerAgent'], # type: ignore[index]
            verbose=True,
            reasoning=True,
            max_reasoning_attempts=3,
            memory=True,
            llm=llm,
            knowledge_sources=[
                content_source_planner
            ],
        )

    @agent
    def CCPIdeologicalAnalyst(self) -> Agent:
        return Agent(
            config=self.agents_config['CCPIdeologicalAnalyst'], # type: ignore[index]
            verbose=True,
            reasoning=True,
            max_reasoning_attempts=3,
            memory=True,
            llm=llm,
            knowledge_sources=[
                content_source_planner
            ],
        )

    @agent
    def DomesticSentimentStabilityAnalyst(self) -> Agent:
        return Agent(
            config=self.agents_config['DomesticSentimentStabilityAnalyst'], # type: ignore[index]
            verbose=True,
            reasoning=True,
            max_reasoning_attempts=3,
            memory=True,
            llm=llm,
            knowledge_sources=[
                content_source_planner
            ],
        )

    @agent
    def HistoricalPrecedentAnalyst(self) -> Agent:
        return Agent(
            config=self.agents_config['HistoricalPrecedentAnalyst'], # type: ignore[index]
            verbose=True,
            reasoning=True,
            max_reasoning_attempts=3,
            memory=True,
            llm=llm,
            knowledge_sources=[
                content_source_planner
            ],
        )

    # --- Task Definitions ---

    @task
    def analyze_event_task(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_event_task'], # type: ignore[index]
        )

    @task
    def assess_economic_tech_impact_task(self) -> Task:
        return Task(
            config=self.tasks_config['assess_economic_tech_impact_task'], # type: ignore[index]
        )

    @task
    def historical_context_task(self) -> Task:
        return Task(
            config=self.tasks_config['historical_context_task'], # type: ignore[index]
        )

    @task
    def develop_active_strategic_postures_task(self) -> Task:
        return Task(
            config=self.tasks_config['develop_active_strategic_postures_task'], # type: ignore[index]
        )

    @task
    def assess_signaling_and_recommend_strategic_path_task(self) -> Task:
        return Task(
            config=self.tasks_config['assess_signaling_and_recommend_strategic_path_task'], # type: ignore[index]
        )
    @task
    def generate_active_pla_options_task(self) -> Task:
        return Task(
            config=self.tasks_config['generate_active_pla_options_task'], # type: ignore[index]
        )

    @task
    def ideological_perception_task(self) -> Task:
        return Task(
            config=self.tasks_config['ideological_perception_task'], # type:
         )

    @task
    def develop_active_diplomatic_strategy_task(self) -> Task:
        return Task(
            config=self.tasks_config['develop_active_diplomatic_strategy_task'], # type: ignore[index]
        )

    @task
    def curate_context_digest_task(self) -> Task:
        return Task(
            config=self.tasks_config['curate_context_digest_task'],  # ← fixed key
        )

    @task
    def internal_impact_narrative_task(self) -> Task:
        return Task(
            config=self.tasks_config['internal_impact_narrative_task'], # type: ignore[index]
        )

    @task
    def develop_strategic_communication_plan_task(self) -> Task:
        return Task(
            config=self.tasks_config['develop_strategic_communication_plan_task'], # type: ignore[index]
        )

    @task
    def format_final_response_task(self) -> Task:
        return Task(
            config=self.tasks_config['format_final_response_task'], # type: ignore[index]
            output_file="output/final_response.md"
        )

    @crew
    def crew(self) -> Crew:
        """Creates the May20Xjp2 crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # This will now include all agents
            tasks=self.tasks,   # This will now include all tasks
            process=Process.sequential,
            planning=True,
            planning_llm=llm,
            knowledge_sources=[
                content_source_planner,
                content_source_instruct
            ],
            verbose=True,
            function_calling_llm=llm,
            chat_llm=llm,
            memory=True,
            output_log_file="log.txt",
            long_term_memory=long_term_memory,
            entity_memory=entity_memory,
            knowledge_config=knowledge_config,
            embedder=embedding_config,
        )


