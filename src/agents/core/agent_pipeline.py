from src.agents.core.agent import Agent, AgentInput, AgentOutput
from typing import Dict, Callable


#!------------------------# Normal Pipeline #-------------------------#
# pipeline = AgentPipeline([AAgent(), BAgent()])                      #
#                                                                     #
# output = pipeline.run(AgentInput(data="4"))                         #
# print(output)  # -> AgentOutput(result=16, metadata={'agent': 'B'}) #
#!--------------------------------------------------------------------#
class AgentPipeline:
    def __init__(self, agents: list[Agent]):
        self.agents = agents

    def run(self, input: AgentInput) -> AgentOutput:
        current_output = input
        for agent in self.agents:
            current_output = agent.run(current_output)  # type: ignore
        return current_output

#!---------------# Conditional Pipeline #----------------#
# Condition function defined OUTSIDE the pipeline        #
# def language_condition_fn(output: AgentOutput) -> str: #
#     return output.metadata.get("language", "unknown")  #
#                                                        #
# pipeline = ConditionalPipeline(                        #
#     entry_agent=LanguageDetector(),                    #
#     routes={                                           #
#         "en": EnglishAgent(),                          #
#         "es": SpanishAgent()                           #
#     },                                                 #
#     condition_fn=language_condition_fn                 #
# )                                                      #
#                                                        #
# print(pipeline.run(AgentInput("hello world")))         #
# # -> Processed English: hello world                    #
#                                                        #
# print(pipeline.run(AgentInput("hola amigo")))          #
# # -> Procesado Español: hola amigo                     #
#!-------------------------------------------------------#
class ConditionalPipeline:
    def __init__(self, entry_agent: Agent,
                 routes: Dict[str, Agent],
                 condition_fn: Callable[[AgentOutput], str]):
        """
        routes: dictionary mapping condition -> agent
        condition_fn: function that decides which route to use,
                      given the entry agent's output
        """
        self.entry_agent = entry_agent
        self.routes = routes
        self.condition_fn = condition_fn

    def run(self, input: AgentInput) -> AgentOutput:
        first_output = self.entry_agent.run(input)
        condition = self.condition_fn(first_output)

        if condition in self.routes:
            return self.routes[condition].run(AgentInput(
                data=first_output.result,
                metadata=first_output.metadata
            ))
        else:
            return AgentOutput(result=first_output.result,
                               metadata={"error": f"No route for condition '{condition}'"})
            
#!----------------------# Route Pipeline #-----------------------#
# Define agents                                                  #
# agents = {                                                     #  
#     "counter": CounterAgent(),                                 #
#     "finish": FinishAgent()                                    #
# }                                                              #
#                                                                #
# # Define routes with loop                                      #
# def counter_route(output: AgentOutput) -> str:                 #
#     if output.metadata["count"] < 3:                           #
#         return "counter"  # loop back                          #
#     return "finish"       # stop after 3                       #
#                                                                #
# routes = {                                                     #
#     "counter": counter_route                                   #
# }                                                              #
#                                                                #
# pipeline = RoutePipeline(agents, routes, start="counter")      #
#                                                                #
# out = pipeline.run(AgentInput(data="start"))                   #
# print(out)                                                     #
# # -> AgentOutput(result='Finished at 3', metadata={'count': 3})#
#!---------------------------------------------------------------#
class RoutePipeline:
    def __init__(self,
                 agents: Dict[str, Agent],
                 routes: Dict[str, Callable[[AgentOutput], str]],
                 start: str,
                 max_steps: int = 50):
        """
        agents: available agents by name
        routes: mapping agent_name -> condition_fn returning next agent_name
        start: entry agent name
        max_steps: safety limit to prevent infinite loops
        """
        self.agents = agents
        self.routes = routes
        self.start = start
        self.max_steps = max_steps

    def run(self, input: AgentInput) -> AgentOutput:
        current_agent_name = self.start
        current_input = input

        for step in range(self.max_steps):
            agent = self.agents[current_agent_name]
            output = agent.run(current_input)

            if current_agent_name not in self.routes:
                # No route → end
                return output

            next_agent_name = self.routes[current_agent_name](output)

            if next_agent_name is None:
                # Explicit stop
                return output

            # Prepare input for next agent
            current_agent_name = next_agent_name
            # Pass the result as data, preserving the structure
            current_input = AgentInput(data=output.result, metadata=output.metadata)

        raise RuntimeError("Pipeline exceeded max_steps (possible infinite loop)")