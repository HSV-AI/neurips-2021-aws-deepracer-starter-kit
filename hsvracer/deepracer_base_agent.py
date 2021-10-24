class DeepracerAgent():
    def __init__(self):
        self.agent_type = None

    def register_reset(self, observations):
        raise NotImplementedError

    def compute_action(self, observations, info):
        raise NotImplementedError