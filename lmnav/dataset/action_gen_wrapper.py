from abc import ABC, abstractmethod

class ActionGeneratorWrapper(ABC):

    @abstractmethod
    def action_generator(config, num_envs, deterministic):
        """
        generator that consumes observations to produce actions
        """
        pass
