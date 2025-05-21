from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai.knowledge.source.crew_docling_source import CrewDoclingSource
from crewai.knowledge.knowledge_config import KnowledgeConfig


llm = LLM(
    model="ollama/qwen3-30b-custom",
    base_url="http://localhost:11434"
)

embedding_config={
    "provider": "ollama",
    "config": {
        "model": "mxbai-embed-large",
    }
}

content_source = CrewDoclingSource(
    file_paths=[
        "### Comprehensive Planning Tool for PRC Employing Gray Zone Tactics.md",
        "XJP Instructions 1.md",
    ],
)

#pdf_source = PDFKnowledgeSource(
#    file_paths=["doc1.pdf", "doc2.pdf"]
#)
#in crew knowledge_sources=[pdf_source]


knowledge_config = KnowledgeConfig(results_limit=10, score_threshold=0.5)


@CrewBase
class May20Xjp2():
    """May20Xjp2 crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended

    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    # CCPStrategicPolicyAdvisor
    @agent
    def CCPStrategicPolicyAdvisor(self) -> Agent:
        return Agent(
            config=self.agents_config['CCPStrategicPolicyAdvisor'], # type: ignore[index]
            verbose=True,
            embedder={
                "provider": "ollama",
                "config": {
                    "model": "mxbai-embed-large",
                    "base_url": "http://localhost:11434"
                }
            },
        )

    # ForeignPolicyEventAnalystAgent
    @agent
    def ForeignPolicyEventAnalystAgent(self) -> Agent:
        return Agent(
            config=self.agents_config['ForeignPolicyEventAnalystAgent'], # type: ignore[index]
            verbose=True,
            embedder={
                "provider": "ollama",
                "config": {
                    "model": "mxbai-embed-large",
                    "base_url": "http://localhost:11434"
                }
            },
        )
    # StrategicSignalingAssessmentAgent
    @agent
    def StrategicSignalingAssessmentAgent(self) -> Agent:
        return Agent(
            config=self.agents_config['StrategicSignalingAssessmentAgent'], # type: ignore[index]
            verbose=True,
            embedder={
                "provider": "ollama",
                "config": {
                    "model": "mxbai-embed-large",
                    "base_url": "http://localhost:11434"
                }
            },
        )

    # PLAOptionsStrategistAgent
    @agent
    def PLAOptionsStrategistAgent(self) -> Agent:
        return Agent(
            config=self.agents_config['PLAOptionsStrategistAgent'], # type: ignore[index]
            verbose=True,
            embedder={
                "provider": "ollama",
                "config": {
                    "model": "mxbai-embed-large",
                    "base_url": "http://localhost:11434"
                }
            },
        )

    # MFADiplomaticStrategistAgent
    @agent
    def MFADiplomaticStrategistAgent(self) -> Agent:
        return Agent(
            config=self.agents_config['MFADiplomaticStrategistAgent'], # type: ignore[index]
            verbose=True,
            embedder={
                "provider": "ollama",
                "config": {
                    "model": "mxbai-embed-large",
                    "base_url": "http://localhost:11434"
                }
            },
        )

        # To learn more about structured task outputs,
        # task dependencies, and task callbacks, check out the documentation:
        # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    # --- Task Definitions ---

    @task
    def analyze_event_task(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_event_task'], # type: ignore[index]
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
    def develop_active_diplomatic_strategy_task(self) -> Task:
        return Task(
            config=self.tasks_config['develop_active_diplomatic_strategy_task'], # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the May20Xjp2 crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            knowledge_sources=[
                content_source
            ],
            verbose=True,
            chat_llm=llm,
            memory=True,
            embedder=embedding_config,
        )
