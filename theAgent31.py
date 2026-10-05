import os
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList, BaseCallback, CheckpointCallback
#from trajectory_env import ArmRobot
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ["MUJOCO_GL"] = "glfw" 
os.environ["RICH_JUPYTER"] = "0"

# Increase the default limit (usually 1000) to 10,000
#sys.setrecursionlimit(10000) 

# record_mean accumulates a running average since the 
# last flush — that's what you want for a per-step value, 
# since SB3 flushes to TensorBoard automatically every 
# rollout (PPO) or every N episodes (SAC/TD3), not every
# single step. Use plain record only for values you want
# overwritten rather than averaged.
class RewardComponentsCallback(BaseCallback):
    """Logs the components from ArmRobot._get_reward()/info dict to TensorBoard."""
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        
        infos = self.locals.get("infos", {})
        for info in infos:  # one dict per parallel env
            self.logger.record_mean("reward/position",         info["reward_position"])
            self.logger.record_mean("reward/orientation",      info["reward_orientation"])
            self.logger.record_mean("reward/progress",         info["reward_progress"])
            self.logger.record_mean("reward/control_cost",     info["reward_control_cost"])
            self.logger.record_mean("reward/orientation_cost", info["orientation_cost"])
            self.logger.record("reward/bonus",            info["bonus_current_target_reached"])
            self.logger.record_mean("reward/penalty",          info["penalty_leaving_workspace"])
            self.logger.record("task/pos_err",            info["pos_err"])
            self.logger.record_mean("reward/penalty",          info["penalty_leaving_workspace"])
            self.logger.record("task/s",            info["s"])
            self.logger.record("task/deviation",            info["deviation"])
            self.logger.record("task/ori_err",            info["ori_err"])
            self.logger.record("task/current_target_idx", info["current_target_idx"])
            self.logger.record_mean("task/cross_error", info["cross_error_vec"])
        return True
class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Access info from the training environment
        infos = self.locals.get("infos", [])
        for info in infos:
            if "position_error" in info:
                # Log to TensorBoard
                self.logger.record("custom/position_error", info["position_error"])
                self.logger.record("custom/orientation_error", info["orientation_error"])
        return True

class SACAgentManager:
    """
    Manager zum Trainieren und Evaluieren eines SAC-Agenten auf einer
    selbstgeschriebenen (nicht registrierten) MuJoCo Gymnasium-Umgebung.
    """
    def __init__(
        self, 
        train_env: gym.Env, 
        log_dir: str = "./tb_logs", 
        model_dir: str = "./saved_models",
        name_p: str = "",
        loadmodel: bool = False
    ):
        self.log_dir = log_dir
        self.model_dir = model_dir
        self.name_p = name_p
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        self.modelpath = os.path.join(model_dir, f"{name_p}_sac_model_env")
        # Die eigene Umgebung direkt in einen Monitor einpacken (wichtig für SB3 TensorBoard Logs)
        self.env = Monitor(train_env)
        self.loadmodel = loadmodel
        # SAC-Modell initialisieren
        if not self.loadmodel:
            self.model = SAC(
                policy="MlpPolicy", 
                env=self.env, 
                verbose=1, 
                tensorboard_log=self.log_dir
            )
        else:
          self.model = SAC.load(self.modelpath, env=self.env)

        self.reward_callback = RewardComponentsCallback()

        # Configure the callback to only save the model weights
        self.checkpoint_callback = CheckpointCallback(
            save_freq=20000,           # Save every 500,000 timesteps
            save_path="./sac_eval_checkpoints/", 
            name_prefix=f"{self.name_p}_sac_eval",
            save_replay_buffer=False   # Keeps file sizes small for manual evaluation
        )

    def train(self, total_timesteps: int = 100_000, tb_log_name: str = f"sac_run_env", 
              save_checkpoints: bool = True, log_reward: bool = True):
        """Trainiert den Agenten und loggt via TensorBoard."""
        print(f"Starte Training für {total_timesteps} Schritte...")

        callback_list = []
        if save_checkpoints:
            callback_list.append(self.checkpoint_callback)
        if log_reward:
            callback_list.append(self.reward_callback)
        self.model.learn(
            total_timesteps=total_timesteps, 
            callback=CallbackList(callback_list) if len(callback_list)>0 else None,
            tb_log_name=tb_log_name,
            reset_num_timesteps=(not self.loadmodel),
            progress_bar=True
        )
        
        save_path = os.path.join(self.model_dir, f"{self.name_p}_sac_model_env")
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
    gym.register(id="my_master_envs/ArmRobot-T31", entry_point="theEnv31:ArmRobot")
    custom_train_env = gym.make("my_master_envs/ArmRobot-T31", qpos_noise=False, episode_len=200, num_points=20) #ArmRobot(render_mode="human")#
    # custom_eval_env = gym.make("my_master_envs/ArmRobot-v2", render_mode="human") #ArmRobot(render_mode="human")#
    #custom_eval_env.render_mode = "human"  # Setze den Render-Modus für die Evaluationsumgebung
    # assert custom_train_env is not custom_train_env, "Trainings- und Evaluationsumgebung müssen unterschiedliche Instanzen sein!"
    # assert custom_train_env.observation_space == custom_train_env.observation_space, "Beide Umgebungen müssen denselben Observation Space haben!"
    # assert custom_train_env.action_space == custom_train_env.action_space, "Beide Umgebungen müssen denselben Action Space haben!"
    # assert custom_train_env.metadata == custom_train_env.metadata, "Beide Umgebungen müssen dieselben Metadaten haben!"
    # assert custom_train_env.fullpath == custom_eval_env.fullpath, "Beide Umgebungen müssen dasselbe MuJoCo-Modell verwenden!"
    # assert custom_train_env.frame_skip == custom_eval_env.frame_skip, "Beide Umgebungen müssen denselben Frame Skip haben!"
    # assert custom_train_env.current_pos is not None and custom_eval_env.current_pos is not None, "Beide Umgebungen müssen initialisierte Positionen haben! please check the reset_model() method in your ArmRobot class."
    # assert custom_train_env.current_target_pos is not None and custom_eval_env.current_target_pos is not None, "Beide Umgebungen müssen initialisierte Zielpositionen haben! Please check the reset_model() method in your ArmRobot class."
    # Manager mit der Instanz deiner Custom-Umgebung füttern
    manager = SACAgentManager(train_env=custom_train_env,name_p="nolook_envTv33", loadmodel=False)
    
    # Trainieren
    manager.train(total_timesteps=2_000_000, tb_log_name="nadel_envTv33_run")
    
    # Evaluieren mit der separaten Custom-Instanz
    #manager.evaluate(eval_env=custom_eval_env, n_eval_episodes=5)