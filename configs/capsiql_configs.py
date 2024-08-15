from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class CapsIQLTrainConfig:
    # wandb params
    project: str = "OSRL"
    group: str = None
    name: Optional[str] = None
    prefix: Optional[str] = "CapsIQL"
    suffix: Optional[str] = ""
    logdir: Optional[str] = "logs"
    verbose: bool = True
    # dataset params
    outliers_percent: float = None
    noise_scale: float = None
    inpaint_ranges: Tuple[Tuple[float, float, float, float], ...] = None
    epsilon: float = None
    density: float = 1.0
    # training params
    task: str = "OfflineCarCircle-v0"
    dataset: str = None
    seed: int = 0
    device: str = "cuda"
    threads: int = 4
    reward_scale: float = 0.1
    cost_scale: float = 1
    actor_lr: float = 3e-4
    q_lr: float = 3e-4
    value_lr: float = 3e-4
    episode_len: int = 300
    batch_size: int = 512
    update_steps: int = 100_000
    num_workers: int = 8
    # model params
    num_heads: int = 2
    iql_deterministic: bool = False
    hidden_dim: int = 512
    iql_tau: float = 0.7
    iql_tau_cost: float = 0.7
    beta: float = 3.0
    beta_cost: float = 3.0
    normalize_cost: int = 0
    normalize_rewards: int = 0
    max_steps: int = 1000000
    gamma: float = 0.99
    tau: float = 0.005
    # evaluation params
    eval_episodes: int = 10
    eval_every: int = 25000


@dataclass
class CapsIQLCarCircleConfig(CapsIQLTrainConfig):
    pass


@dataclass
class CapsIQLAntRunConfig(CapsIQLTrainConfig):
    # training params
    task: str = "OfflineAntRun-v0"
    episode_len: int = 200


@dataclass
class CapsIQLDroneRunConfig(CapsIQLTrainConfig):
    # training params
    task: str = "OfflineDroneRun-v0"
    episode_len: int = 200


@dataclass
class CapsIQLDroneCircleConfig(CapsIQLTrainConfig):
    # training params
    task: str = "OfflineDroneCircle-v0"
    episode_len: int = 300


@dataclass
class CapsIQLCarRunConfig(CapsIQLTrainConfig):
    # training params
    task: str = "OfflineCarRun-v0"
    episode_len: int = 200


@dataclass
class CapsIQLAntCircleConfig(CapsIQLTrainConfig):
    # training params
    task: str = "OfflineAntCircle-v0"
    episode_len: int = 500


@dataclass
class CapsIQLBallRunConfig(CapsIQLTrainConfig):
    # training params
    task: str = "OfflineBallRun-v0"
    episode_len: int = 100


@dataclass
class CapsIQLBallCircleConfig(CapsIQLTrainConfig):
    # training params
    task: str = "OfflineBallCircle-v0"
    episode_len: int = 200


@dataclass
class CapsIQLCarButton1Config(CapsIQLTrainConfig):
    # training params
    task: str = "OfflineCarButton1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class CapsIQLCarButton2Config(CapsIQLTrainConfig):
    # training params
    task: str = "OfflineCarButton2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class CapsIQLCarCircle1Config(CapsIQLTrainConfig):
    # training params
    task: str = "OfflineCarCircle1Gymnasium-v0"
    episode_len: int = 500


@dataclass
class CapsIQLCarCircle2Config(CapsIQLTrainConfig):
    # training params
    task: str = "OfflineCarCircle2Gymnasium-v0"
    episode_len: int = 500


@dataclass
class CapsIQLCarGoal1Config(CapsIQLTrainConfig):
    # training params
    task: str = "OfflineCarGoal1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class CapsIQLCarGoal2Config(CapsIQLTrainConfig):
    # training params
    task: str = "OfflineCarGoal2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class CapsIQLCarPush1Config(CapsIQLTrainConfig):
    # training params
    task: str = "OfflineCarPush1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class CapsIQLCarPush2Config(CapsIQLTrainConfig):
    # training params
    task: str = "OfflineCarPush2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class CapsIQLPointButton1Config(CapsIQLTrainConfig):
    # training params
    task: str = "OfflinePointButton1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class CapsIQLPointButton2Config(CapsIQLTrainConfig):
    # training params
    task: str = "OfflinePointButton2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class CapsIQLPointCircle1Config(CapsIQLTrainConfig):
    # training params
    task: str = "OfflinePointCircle1Gymnasium-v0"
    episode_len: int = 500


@dataclass
class CapsIQLPointCircle2Config(CapsIQLTrainConfig):
    # training params
    task: str = "OfflinePointCircle2Gymnasium-v0"
    episode_len: int = 500


@dataclass
class CapsIQLPointGoal1Config(CapsIQLTrainConfig):
    # training params
    task: str = "OfflinePointGoal1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class CapsIQLPointGoal2Config(CapsIQLTrainConfig):
    # training params
    task: str = "OfflinePointGoal2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class CapsIQLPointPush1Config(CapsIQLTrainConfig):
    # training params
    task: str = "OfflinePointPush1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class CapsIQLPointPush2Config(CapsIQLTrainConfig):
    # training params
    task: str = "OfflinePointPush2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class CapsIQLAntVelocityConfig(CapsIQLTrainConfig):
    # training params
    task: str = "OfflineAntVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class CapsIQLHalfCheetahVelocityConfig(CapsIQLTrainConfig):
    # training params
    task: str = "OfflineHalfCheetahVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class CapsIQLHopperVelocityConfig(CapsIQLTrainConfig):
    # training params
    task: str = "OfflineHopperVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class CapsIQLSwimmerVelocityConfig(CapsIQLTrainConfig):
    # training params
    task: str = "OfflineSwimmerVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class CapsIQLWalker2dVelocityConfig(CapsIQLTrainConfig):
    # training params
    task: str = "OfflineWalker2dVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class CapsIQLEasySparseConfig(CapsIQLTrainConfig):
    # training params
    task: str = "OfflineMetadrive-easysparse-v0"
    iql_tau: float = 0.5
    iql_tau_cost: float = 0.5
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class CapsIQLEasyMeanConfig(CapsIQLTrainConfig):
    # training params
    task: str = "OfflineMetadrive-easymean-v0"
    iql_tau: float = 0.5
    iql_tau_cost: float = 0.5
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class CapsIQLEasyDenseConfig(CapsIQLTrainConfig):
    # training params
    task: str = "OfflineMetadrive-easydense-v0"
    iql_tau: float = 0.5
    iql_tau_cost: float = 0.5
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class CapsIQLMediumSparseConfig(CapsIQLTrainConfig):
    # training params
    task: str = "OfflineMetadrive-mediumsparse-v0"
    iql_tau: float = 0.5
    iql_tau_cost: float = 0.5
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class CapsIQLMediumMeanConfig(CapsIQLTrainConfig):
    # training params
    task: str = "OfflineMetadrive-mediummean-v0"
    iql_tau: float = 0.5
    iql_tau_cost: float = 0.5
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class CapsIQLMediumDenseConfig(CapsIQLTrainConfig):
    # training params
    task: str = "OfflineMetadrive-mediumdense-v0"
    iql_tau: float = 0.5
    iql_tau_cost: float = 0.5
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class CapsIQLHardSparseConfig(CapsIQLTrainConfig):
    # training params
    task: str = "OfflineMetadrive-hardsparse-v0"
    iql_tau: float = 0.5
    iql_tau_cost: float = 0.5
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class CapsIQLHardMeanConfig(CapsIQLTrainConfig):
    # training params
    task: str = "OfflineMetadrive-hardmean-v0"
    iql_tau: float = 0.5
    iql_tau_cost: float = 0.5
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class CapsIQLHardDenseConfig(CapsIQLTrainConfig):
    # training params
    task: str = "OfflineMetadrive-harddense-v0"
    iql_tau: float = 0.5
    iql_tau_cost: float = 0.5
    episode_len: int = 1000
    update_steps: int = 200_000


CapsIQL_DEFAULT_CONFIG = {
    # bullet_safety_gym
    "OfflineCarCircle-v0": CapsIQLCarCircleConfig,
    "OfflineAntRun-v0": CapsIQLAntRunConfig,
    "OfflineDroneRun-v0": CapsIQLDroneRunConfig,
    "OfflineDroneCircle-v0": CapsIQLDroneCircleConfig,
    "OfflineCarRun-v0": CapsIQLCarRunConfig,
    "OfflineAntCircle-v0": CapsIQLAntCircleConfig,
    "OfflineBallCircle-v0": CapsIQLBallCircleConfig,
    "OfflineBallRun-v0": CapsIQLBallRunConfig,
    # safety_gymnasium
    "OfflineCarButton1Gymnasium-v0": CapsIQLCarButton1Config,
    "OfflineCarButton2Gymnasium-v0": CapsIQLCarButton2Config,
    "OfflineCarCircle1Gymnasium-v0": CapsIQLCarCircle1Config,
    "OfflineCarCircle2Gymnasium-v0": CapsIQLCarCircle2Config,
    "OfflineCarGoal1Gymnasium-v0": CapsIQLCarGoal1Config,
    "OfflineCarGoal2Gymnasium-v0": CapsIQLCarGoal2Config,
    "OfflineCarPush1Gymnasium-v0": CapsIQLCarPush1Config,
    "OfflineCarPush2Gymnasium-v0": CapsIQLCarPush2Config,
    # safety_gymnasium: point
    "OfflinePointButton1Gymnasium-v0": CapsIQLPointButton1Config,
    "OfflinePointButton2Gymnasium-v0": CapsIQLPointButton2Config,
    "OfflinePointCircle1Gymnasium-v0": CapsIQLPointCircle1Config,
    "OfflinePointCircle2Gymnasium-v0": CapsIQLPointCircle2Config,
    "OfflinePointGoal1Gymnasium-v0": CapsIQLPointGoal1Config,
    "OfflinePointGoal2Gymnasium-v0": CapsIQLPointGoal2Config,
    "OfflinePointPush1Gymnasium-v0": CapsIQLPointPush1Config,
    "OfflinePointPush2Gymnasium-v0": CapsIQLPointPush2Config,
    # safety_gymnasium: velocity
    "OfflineAntVelocityGymnasium-v1": CapsIQLAntVelocityConfig,
    "OfflineHalfCheetahVelocityGymnasium-v1": CapsIQLHalfCheetahVelocityConfig,
    "OfflineHopperVelocityGymnasium-v1": CapsIQLHopperVelocityConfig,
    "OfflineSwimmerVelocityGymnasium-v1": CapsIQLSwimmerVelocityConfig,
    "OfflineWalker2dVelocityGymnasium-v1": CapsIQLWalker2dVelocityConfig,
    # safe_metadrive
    "OfflineMetadrive-easysparse-v0": CapsIQLEasySparseConfig,
    "OfflineMetadrive-easymean-v0": CapsIQLEasyMeanConfig,
    "OfflineMetadrive-easydense-v0": CapsIQLEasyDenseConfig,
    "OfflineMetadrive-mediumsparse-v0": CapsIQLMediumSparseConfig,
    "OfflineMetadrive-mediummean-v0": CapsIQLMediumMeanConfig,
    "OfflineMetadrive-mediumdense-v0": CapsIQLMediumDenseConfig,
    "OfflineMetadrive-hardsparse-v0": CapsIQLHardSparseConfig,
    "OfflineMetadrive-hardmean-v0": CapsIQLHardMeanConfig,
    "OfflineMetadrive-harddense-v0": CapsIQLHardDenseConfig
}