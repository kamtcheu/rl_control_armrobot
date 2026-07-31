import os
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from rl_control_armrobot.trajectory_env import ArmRobot


class SACAgentManager:
    """
    Manager zum Trainieren und Evaluieren eines SAC-Agenten auf einer
    selbstgeschriebenen (nicht registrierten) MuJoCo Gymnasium-Umgebung.
    """
    def __init__(
        self, 
        train_env: gym.Env, 
        log_dir: str = "./tb_logs", 
        model_dir: str = "./saved_models"
    ):
        self.log_dir = log_dir
        self.model_dir = model_dir
        
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Die eigene Umgebung direkt in einen Monitor einpacken (wichtig für SB3 TensorBoard Logs)
        self.env = Monitor(train_env)
        
        # SAC-Modell initialisieren
        self.model = SAC(
            policy="MlpPolicy", 
            env=self.env, 
            verbose=1, 
            tensorboard_log=self.log_dir
        )

    def train(self, total_timesteps: int = 100_000, tb_log_name: str = "sac_run"):
        """Trainiert den Agenten und loggt via TensorBoard."""
        print(f"Starte Training für {total_timesteps} Schritte...")
        self.model.learn(
            total_timesteps=total_timesteps, 
            tb_log_name=tb_log_name,
            progress_bar=True
        )
        
        save_path = os.path.join(self.model_dir, "sac_model")
        self.model.save(save_path)
        print(f"Modell erfolgreich unter '{save_path}' gespeichert.")

    def evaluate(self, eval_env: gym.Env, n_eval_episodes: int = 10):
        """
        Evaluiert die aktuelle Policy auf einer übergebenen Testumgebung.
        Es empfiehlt sich, hierfür eine frische Instanz der Custom Env zu nutzen.
        """
        print(f"Starte Evaluation über {n_eval_episodes} Episoden...")
        
        # Auch die Eval-Umgebung überwachen
        eval_env_monitored = Monitor(eval_env)
        
        mean_reward, std_reward = evaluate_policy(
            self.model, 
            eval_env_monitored, 
            n_eval_episodes=n_eval_episodes, 
            deterministic=True
        )
        
        print(f"Evaluationsergebnis:")
        print(f" -> Mittlere Belohnung: {mean_reward:.2f} +/- {std_reward:.2f}")
        
        eval_env_monitored.close()
        return mean_reward, std_reward

    def load_model(self, path: str):
        """Lädt ein bereits trainiertes Modell."""
        self.model = SAC.load(path, env=self.env)
        print(f"Modell aus '{path}' erfolgreich geladen.")


# --- Anwendungsbeispiel mit einer fiktiven Custom-Umgebung ---
if __name__ == "__main__":
    
    # Angenommen, das hier ist deine eigene MuJoCo-Klasse:
    # class MyCustomMuJoCoEnv(gymnasium.Env):
    #     ...
    
    # Wichtig: Da wir zwei separate Instanzen brauchen (eine fürs Training, eine für die Evaluation),
    # instanziieren wir deine Umgebung hier zweimal.
    
    custom_train_env = ArmRobot()
    custom_eval_env = ArmRobot()

    assert custom_train_env is not custom_eval_env, "Trainings- und Evaluationsumgebung müssen unterschiedliche Instanzen sein!"
    assert custom_train_env.observation_space == custom_eval_env.observation_space, "Beide Umgebungen müssen denselben Observation Space haben!"
    assert custom_train_env.action_space == custom_eval_env.action_space, "Beide Umgebungen müssen denselben Action Space haben!"
    assert custom_train_env.metadata == custom_eval_env.metadata, "Beide Umgebungen müssen dieselben Metadaten haben!"
    assert custom_train_env.model_path == custom_eval_env.model_path, "Beide Umgebungen müssen dasselbe MuJoCo-Modell verwenden!"
    assert custom_train_env.frame_skip == custom_eval_env.frame_skip, "Beide Umgebungen müssen denselben Frame Skip haben!"
    assert custom_train_env.current_pos is not None and custom_eval_env.current_pos is not None, "Beide Umgebungen müssen initialisierte Positionen haben! please check the reset_model() method in your ArmRobot class."
    assert custom_train_env.current_target_pos is not None and custom_eval_env.current_target_pos is not None, "Beide Umgebungen müssen initialisierte Zielpositionen haben! Please check the reset_model() method in your ArmRobot class."
    # Manager mit der Instanz deiner Custom-Umgebung füttern
    manager = SACAgentManager(train_env=custom_train_env)
    
    # Trainieren
    manager.train(total_timesteps=50_000, tb_log_name="nadel_env_run")
    
    # Evaluieren mit der separaten Custom-Instanz
    manager.evaluate(eval_env=custom_eval_env, n_eval_episodes=5)