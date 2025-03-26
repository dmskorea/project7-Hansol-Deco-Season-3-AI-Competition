import importlib

from .env import EnvDefineUnit



class ConfigDefineTool:
    def __init__(self, exp_name="Experiment_0"):
        self.exp_name = exp_name

    def get_env(self):
        self.env = EnvDefineUnit()
        return self.env

    def get_exp(self):
        module = importlib.import_module(".experiments", package=__package__)
        exp_class = getattr(module, self.exp_name)
        self.exp = exp_class()
        return self.exp



if __name__ == "__main__":
    tool = ConfigDefineTool()
    environment = tool.get_env()  # 한번만 호출하고 반환값 사용
    experiment = tool.get_exp()  # 한번만 호출하고 반환값 사용
    print(environment)
    print(experiment)
