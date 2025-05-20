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
        "lkoc_history_document.pdf",
        "acs09nar_kailuacdp_3yr.pdf",
    ],
)

#pdf_source = PDFKnowledgeSource(
#    file_paths=["doc1.pdf", "doc2.pdf"]
#)
#in crew knowledge_sources=[pdf_source]


knowledge_config = KnowledgeConfig(results_limit=10, score_threshold=0.5)


@CrewBase
class May191():
    """May191 crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended

    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'], # type: ignore[index]
            verbose=True,
            #knosledge_sources=[],
            embedder={
                "provider": "ollama",
                "config": {
                    "model": "mxbai-embed-large",
                    "base_url": "http://localhost:11434"
                    }
                  },
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['reporting_analyst'], # type: ignore[index]
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
    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'], # type: ignore[index]
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'], # type: ignore[index]
            output_file='tpsreport.md'
        )


    @crew
    def crew(self) -> Crew:
        """Creates the May191 crew"""
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

