# agent.py
from schema_transform_tools import SchemaTransformTools
from llm_models import *
from prompts import init_template, ultimate_task, examples
from langchain import PromptTemplate
from action import ActionGraph
import random
from mcts import mcts

from copy import deepcopy
from mcts import mcts
from functools import reduce
import operator


class Agent:
    def __init__(self, source_schema, target_schema, source_examples, target_examples, max_step=10):
        self.source_schema = source_schema
        self.target_schema = target_schema
        self.source_examples = source_examples
        self.target_examples = target_examples
        self.max_step = max_step
        self.state = None
        self.action_graph = ActionGraph()
        self.performed_actions = set()
        self.mcts = mcts(timeLimit=1000)
        template = PromptTemplate(
            input_variables=["examples", "source_examples", "target_examples", "source_schema", "target_schema"],
            template=init_template)
        self.prompt = template.format(
            examples=examples,
            source_schema=source_schema,
            target_schema=target_schema,
            source_examples=source_examples,
            target_examples=target_examples
        )
        self.ultimate_task = ultimate_task
        self.transformer = SchemaTransformTools(source_schema, target_schema, source_examples, target_examples)

    def get_state(self):
        return self.state

    def execute_action(self, action, transformer=None, reason_history=None):
        finish = False
        # print(f"Executing action: {action}")
        if (action.strip().startswith("TypePredict")):
            # print("TypePredict")
            observation = transformer.type_predict()
            self.performed_actions.add("TypePredict")
        elif action.strip().startswith("DirectMapping"):
            # print("DirectMapping")
            observation = transformer.column_mapping()
            self.performed_actions.add("DirectMapping")
        elif action.strip().startswith("Aggregation"):
            # print("Aggregation")
            observation = transformer.aggregation()
            self.performed_actions.add("Aggregation")
        elif action.strip().startswith("Clarify"):
            # print("Clarify")
            question = action.strip()[len("Clarify["):-1]
            observation = transformer.clarify(question)
            self.performed_actions.add("Clarify")
        elif action.strip().startswith("Conditional"):
            # print("Conditional")
            observation = transformer.conditional()
            self.performed_actions.add("Conditional")
        elif action.strip().startswith("Finish"):
            # print("Finish")
            response = action.strip()#[len("Finish["):-1]
            observation = transformer.finish(response)
            finish = True
            self.performed_actions.add("Finish")
        else:
            observation = "Invalid action: {}".format(action)

        return observation, finish

    def run(self, to_print=True):
        self.state = self.prompt + self.ultimate_task + "\n"

        n_calls, n_badcalls = 0, 0
        picker = [-1] *10
        #picker[2] = 1
        for i in range(1, self.max_step + 1):
            n_calls += 1

            if picker[i] == 1:#random.random() < 0.1:
                # Use MCTS for action selection
                self.react_state = ReactState(self.performed_actions, self.state, self.action_graph, self.transformer, i)
                bestChild = self.mcts.search(
                    self.react_state,
                    needDetails=False
                )
                action = bestChild.state.current_action
                self.performed_actions.add(action)
                observation = bestChild.state.current_observation
                finish = bestChild.state.is_terminal

                # Craft a detailed prompt for generating thought
                #observation, finish = self.execute_action(action, transformer=self.transformer)
                step_str = f"Thought {i}: MCTS - {action}\nAction {i}: {action}\nObservation {i}: {observation}\n"
            else:
                thought_action = gpt3(self.state + f"Thought {i}:", stop=[f"\nObservation {i}:"])
                try:
                    thought, action = thought_action.strip().split(f"\nAction {i}:")
                except:
                    print('ohh...', thought_action)
                    n_badcalls += 1
                    n_calls += 1
                    thought = thought_action.strip().split('\n')[0]
                    action = gpt3(self.state + f"Thought {i}: {thought}\nAction {i}:", stop=[f"\nObservation {i}:"]).strip()

                observation, finish = self.execute_action(action, transformer=self.transformer)
                step_str = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {observation}\n"
            self.state += step_str
            if to_print:
                print(step_str)
            if finish:
                break

        return observation, n_calls, n_badcalls


class ReactState():
    def __init__(self, performed_actions, state, action_graph, transformer, i):
        self.current_action = None
        self.current_observation = None

        self.performed_actions = performed_actions
        self.action_graph = action_graph
        self.state_in_mcts = state
        self.transformer = transformer
        self.is_terminal = False
        self.i = i

    def getPossibleActions(self):
        print('performed actions', self.performed_actions)
        possible_actions = self.action_graph.get_possible_actions(self.performed_actions)
        print('possible actions', possible_actions)
        return possible_actions

    def takeAction(self, action):
        newState = deepcopy(self)
        print(f"Taking action: {action}")
        newState.performed_actions.add(action)
        newState.current_action = action  # result of the observation
        newState.is_terminal = True if action == 'Finish' else False
        # TODO: add action support
        finish = False
        # print(f"Executing action: {action}")
        if (action.strip().startswith("TypePredict")):
            # print("TypePredict")
            observation = self.transformer.type_predict()
            self.performed_actions.add("TypePredict")
        elif action.strip().startswith("DirectMapping"):
            # print("DirectMapping")
            observation = self.transformer.column_mapping()
            self.performed_actions.add("DirectMapping")
        elif action.strip().startswith("Aggregation"):
            # print("Aggregation")
            observation = self.transformer.aggregation()
            self.performed_actions.add("Aggregation")
        elif action.strip().startswith("Clarify"):
            # print("Clarify")
            prompt_q_mcts = f"""
            You are a Postgres SQL developer. Given the following prompt:\n{self.state_in_mcts}\nWhat would you ask for clarification? Provide only one concise question. Wrap the question between [START] and [END]
            """
            question_response = gpt3(prompt_q_mcts)
            question = self.transformer.result_extractor(question_response)
            #question = action.strip()[len("Clarify"):-1]
            observation = self.transformer.clarify(question)
            self.performed_actions.add("Clarify")
            newState.state_in_mcts = self.state_in_mcts + f"Thought {self.i}: Use MCTS\nAction {self.i}: {action}+'['+{question}+']'+\nObservation {self.i}: {observation}\n"
            return newState
        elif action.strip().startswith("Conditional"):
            # print("Conditional")
            observation = self.transformer.conditional()
            self.performed_actions.add("Conditional")
        elif action.strip().startswith("Finish"):
            # print("Finish")
            response = action.strip()  # [len("Finish["):-1]
            observation = self.transformer.finish(response)
            finish = True
            self.performed_actions.add("Finish")
        else:
            observation = "Invalid action: {}".format(action)

        newState.current_observation = observation
        newState.state_in_mcts = self.state_in_mcts + f"Thought {self.i}: Use MCTS\nAction {self.i}: {action}\nObservation {self.i}: {observation}\n"
        return newState

    def isTerminal(self):
        if self.current_action == 'Finish':
            return True
        else:
            return False

    def getReward(self):
        print('get reward')
        return random.random() * 100
