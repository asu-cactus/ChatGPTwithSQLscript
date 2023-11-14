class ActionGraph:
    def __init__(self):
        self.dependencies = {}  # key is an action, value is a set of actions that depend on the key
        self.repeatable_actions = set()  # Set of actions that can be repeated
        self._initialize_graph()

    def _initialize_graph(self):
        actions_with_dependencies = {
            'TypePredict': None,
            'DirectMapping': ['TypePredict'],
            'Aggregation': ['TypePredict'],
            #'Join': ['TypePredict'],
            #'Clarify': None,
            'Conditional': ['TypePredict', 'DirectMapping', 'Aggregation'],
            'Finish': ['TypePredict', 'DirectMapping', 'Aggregation', 'Conditional'],
        }

        repeatable_actions = {}#{'Clarify'}

        for action, dependencies in actions_with_dependencies.items():
            if dependencies:  # If there are dependencies
                for dependency in dependencies:
                    self.add_action(action, depends_on=dependency)
            else:  # If there are no dependencies
                self.add_action(action)
            if action in repeatable_actions:
                self.set_repeatable(action)

    def add_action(self, action, depends_on=None):
        if depends_on is not None:
            # If the action has dependencies, add them
            if action not in self.dependencies:
                self.dependencies[action] = set()
            self.dependencies[action].add(depends_on)
        else:
            # If the action has no dependencies, ensure it has an entry with an empty set
            self.dependencies.setdefault(action, set())

    def set_repeatable(self, action):
        # Mark an action as repeatable
        self.repeatable_actions.add(action)

    def get_possible_actions(self, performed_actions):
        # Return all actions where their dependencies are a subset of the performed_actions
        possible_actions = set()
        for action, dependencies in self.dependencies.items():
            if dependencies.issubset(performed_actions):
                possible_actions.add(action)
        # Add repeatable actions back into the possible actions
        return list ((possible_actions - performed_actions) |  self.repeatable_actions)

# Example usage:

if __name__ == "__main__":
    action_graph = ActionGraph()
    performed_actions = {'TypePredict', 'Clarify'}
    print(action_graph.get_possible_actions(performed_actions))  # Should now include 'Clarify' and other possible actions

    performed_actions = {'Clarify'}
    print(action_graph.get_possible_actions(performed_actions))  # Should now only include 'Clarify' and other possible actions
