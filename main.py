import os
import shutil
import fire
import yaml


class TrainerCLI(object):
    """Module for model training and evaluating.

    To train new model, use `train` function, to evaluate, use 'evaluate' function.

    """

    def train(self, yaml_path: str, seed: int = 0) -> None:  # noqa: F811
        """Train new one or finetuning an existing model.

        Parameters
        ----------
        yaml_path : str
            Path to YAML config file of experiment
        seed : int, optional (default=1)

        """
        fix_seed(seed)

        with open(yaml_path) as f:
            trainer = yaml.load(f, yaml.Loader)

        os.makedirs(trainer.save_folder, exist_ok=True)
        shutil.copy2(yaml_path, os.path.join(trainer.save_folder, "config.yml"))

        trainer.train()
        print("Training is completed.")

    def evaluate(self, yaml_path: str, seed: int = 0) -> None:
        """Evaluate model.

        Parameters
        ----------
        yaml_path : str
            Path to YAML config file of evaluating.

        """
        fix_seed(seed)

        with open(yaml_path) as f:
            evaluator = yaml.load(f, yaml.Loader)

        os.makedirs(evaluator.save_folder, exist_ok=True)
        shutil.copy2(yaml_path, os.path.join(evaluator.save_folder, "config_eval.yml"))

        evaluator.evaluate()
        print("Evaluation is completed.")


if __name__ == "__main__":
    fire.Fire(TrainerCLI)