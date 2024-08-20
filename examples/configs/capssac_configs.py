from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

from pyrallis import field


@dataclass
class CapsSACTrainConfig:
    # wandb params
    project: str = "OSRL-baselines"
    group: str = None
    name: Optional[str] = None
    prefix: Optional[str] = "CapsSAC"
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
    device: str = "cuda:0"
    threads: int = 4
    reward_scale: float = 0.1
    cost_scale: float = 1
    actor_lr: float = 0.0001
    critic_lr: float = 0.001
    alpha_lr: float = 0.0001
    vae_lr: float = 0.001
    episode_len: int = 300
    batch_size: int = 512
    update_steps: int = 100_000
    num_workers: int = 8
    # model params
    a_hidden_sizes: List[float] = field(default=[512, 512], is_mutable=True)
    c_hidden_sizes: List[float] = field(default=[512, 512], is_mutable=True)
    init_temperature: float = 0.1
    alphas_betas: List[float] = field(default=[0.9, 0.999], is_mutable=True)
    actor_betas: List[float] = field(default=[0.9, 0.999], is_mutable=True)
    critic_betas: List[float] = field(default=[0.9, 0.999], is_mutable=True)
    BC: bool = True
    learnable_temperature: bool = True
    gamma: float = 0.99
    tau: float = 0.005
    beta: float = 0.5
    num_heads: int = 2
    num_q: int = 2
    num_qc: int = 2
    # evaluation params
    eval_episodes: int = 10
    eval_every: int = 2500


@dataclass
class CapsSACCarCircleConfig(CapsSACTrainConfig):
    pass


@dataclass
class CapsSACAntRunConfig(CapsSACTrainConfig):
    # training params
    task: str = "OfflineAntRun-v0"
    episode_len: int = 200


@dataclass
class CapsSACDroneRunConfig(CapsSACTrainConfig):
    # training params
    task: str = "OfflineDroneRun-v0"
    episode_len: int = 200


@dataclass
class CapsSACDroneCircleConfig(CapsSACTrainConfig):
    # training params
    task: str = "OfflineDroneCircle-v0"
    episode_len: int = 300


@dataclass
class CapsSACCarRunConfig(CapsSACTrainConfig):
    # training params
    task: str = "OfflineCarRun-v0"
    episode_len: int = 200


@dataclass
class CapsSACAntCircleConfig(CapsSACTrainConfig):
    # training params
    task: str = "OfflineAntCircle-v0"
    episode_len: int = 500


@dataclass
class CapsSACBallRunConfig(CapsSACTrainConfig):
    # training params
    task: str = "OfflineBallRun-v0"
    episode_len: int = 100


@dataclass
class CapsSACBallCircleConfig(CapsSACTrainConfig):
    # training params
    task: str = "OfflineBallCircle-v0"
    episode_len: int = 200


@dataclass
class CapsSACCarButton1Config(CapsSACTrainConfig):
    # training params
    task: str = "OfflineCarButton1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class CapsSACCarButton2Config(CapsSACTrainConfig):
    # training params
    task: str = "OfflineCarButton2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class CapsSACCarCircle1Config(CapsSACTrainConfig):
    # training params
    task: str = "OfflineCarCircle1Gymnasium-v0"
    episode_len: int = 500


@dataclass
class CapsSACCarCircle2Config(CapsSACTrainConfig):
    # training params
    task: str = "OfflineCarCircle2Gymnasium-v0"
    episode_len: int = 500


@dataclass
class CapsSACCarGoal1Config(CapsSACTrainConfig):
    # training params
    task: str = "OfflineCarGoal1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class CapsSACCarGoal2Config(CapsSACTrainConfig):
    # training params
    task: str = "OfflineCarGoal2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class CapsSACCarPush1Config(CapsSACTrainConfig):
    # training params
    task: str = "OfflineCarPush1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class CapsSACCarPush2Config(CapsSACTrainConfig):
    # training params
    task: str = "OfflineCarPush2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class CapsSACPointButton1Config(CapsSACTrainConfig):
    # training params
    task: str = "OfflinePointButton1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class CapsSACPointButton2Config(CapsSACTrainConfig):
    # training params
    task: str = "OfflinePointButton2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class CapsSACPointCircle1Config(CapsSACTrainConfig):
    # training params
    task: str = "OfflinePointCircle1Gymnasium-v0"
    episode_len: int = 500


@dataclass
class CapsSACPointCircle2Config(CapsSACTrainConfig):
    # training params
    task: str = "OfflinePointCircle2Gymnasium-v0"
    episode_len: int = 500


@dataclass
class CapsSACPointGoal1Config(CapsSACTrainConfig):
    # training params
    task: str = "OfflinePointGoal1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class CapsSACPointGoal2Config(CapsSACTrainConfig):
    # training params
    task: str = "OfflinePointGoal2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class CapsSACPointPush1Config(CapsSACTrainConfig):
    # training params
    task: str = "OfflinePointPush1Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class CapsSACPointPush2Config(CapsSACTrainConfig):
    # training params
    task: str = "OfflinePointPush2Gymnasium-v0"
    episode_len: int = 1000


@dataclass
class CapsSACAntVelocityConfig(CapsSACTrainConfig):
    # training params
    task: str = "OfflineAntVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class CapsSACHalfCheetahVelocityConfig(CapsSACTrainConfig):
    # training params
    task: str = "OfflineHalfCheetahVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class CapsSACHopperVelocityConfig(CapsSACTrainConfig):
    # training params
    task: str = "OfflineHopperVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class CapsSACSwimmerVelocityConfig(CapsSACTrainConfig):
    # training params
    task: str = "OfflineSwimmerVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class CapsSACWalker2dVelocityConfig(CapsSACTrainConfig):
    # training params
    task: str = "OfflineWalker2dVelocityGymnasium-v1"
    episode_len: int = 1000


@dataclass
class CapsSACEasySparseConfig(CapsSACTrainConfig):
    # training params
    task: str = "OfflineMetadrive-easysparse-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class CapsSACEasyMeanConfig(CapsSACTrainConfig):
    # training params
    task: str = "OfflineMetadrive-easymean-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class CapsSACEasyDenseConfig(CapsSACTrainConfig):
    # training params
    task: str = "OfflineMetadrive-easydense-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class CapsSACMediumSparseConfig(CapsSACTrainConfig):
    # training params
    task: str = "OfflineMetadrive-mediumsparse-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class CapsSACMediumMeanConfig(CapsSACTrainConfig):
    # training params
    task: str = "OfflineMetadrive-mediummean-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class CapsSACMediumDenseConfig(CapsSACTrainConfig):
    # training params
    task: str = "OfflineMetadrive-mediumdense-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class CapsSACHardSparseConfig(CapsSACTrainConfig):
    # training params
    task: str = "OfflineMetadrive-hardsparse-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class CapsSACHardMeanConfig(CapsSACTrainConfig):
    # training params
    task: str = "OfflineMetadrive-hardmean-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


@dataclass
class CapsSACHardDenseConfig(CapsSACTrainConfig):
    # training params
    task: str = "OfflineMetadrive-harddense-v0"
    episode_len: int = 1000
    update_steps: int = 200_000


CapsSAC_DEFAULT_CONFIG = {
    # bullet_safety_gym
    "OfflineCarCircle-v0": CapsSACCarCircleConfig,
    "OfflineAntRun-v0": CapsSACAntRunConfig,
    "OfflineDroneRun-v0": CapsSACDroneRunConfig,
    "OfflineDroneCircle-v0": CapsSACDroneCircleConfig,
    "OfflineCarRun-v0": CapsSACCarRunConfig,
    "OfflineAntCircle-v0": CapsSACAntCircleConfig,
    "OfflineBallCircle-v0": CapsSACBallCircleConfig,
    "OfflineBallRun-v0": CapsSACBallRunConfig,
    # safety_gymnasium
    "OfflineCarButton1Gymnasium-v0": CapsSACCarButton1Config,
    "OfflineCarButton2Gymnasium-v0": CapsSACCarButton2Config,
    "OfflineCarCircle1Gymnasium-v0": CapsSACCarCircle1Config,
    "OfflineCarCircle2Gymnasium-v0": CapsSACCarCircle2Config,
    "OfflineCarGoal1Gymnasium-v0": CapsSACCarGoal1Config,
    "OfflineCarGoal2Gymnasium-v0": CapsSACCarGoal2Config,
    "OfflineCarPush1Gymnasium-v0": CapsSACCarPush1Config,
    "OfflineCarPush2Gymnasium-v0": CapsSACCarPush2Config,
    # safety_gymnasium: point
    "OfflinePointButton1Gymnasium-v0": CapsSACPointButton1Config,
    "OfflinePointButton2Gymnasium-v0": CapsSACPointButton2Config,
    "OfflinePointCircle1Gymnasium-v0": CapsSACPointCircle1Config,
    "OfflinePointCircle2Gymnasium-v0": CapsSACPointCircle2Config,
    "OfflinePointGoal1Gymnasium-v0": CapsSACPointGoal1Config,
    "OfflinePointGoal2Gymnasium-v0": CapsSACPointGoal2Config,
    "OfflinePointPush1Gymnasium-v0": CapsSACPointPush1Config,
    "OfflinePointPush2Gymnasium-v0": CapsSACPointPush2Config,
    # safety_gymnasium: velocity
    "OfflineAntVelocityGymnasium-v1": CapsSACAntVelocityConfig,
    "OfflineHalfCheetahVelocityGymnasium-v1": CapsSACHalfCheetahVelocityConfig,
    "OfflineHopperVelocityGymnasium-v1": CapsSACHopperVelocityConfig,
    "OfflineSwimmerVelocityGymnasium-v1": CapsSACSwimmerVelocityConfig,
    "OfflineWalker2dVelocityGymnasium-v1": CapsSACWalker2dVelocityConfig,
    # safe_metadrive
    "OfflineMetadrive-easysparse-v0": CapsSACEasySparseConfig,
    "OfflineMetadrive-easymean-v0": CapsSACEasyMeanConfig,
    "OfflineMetadrive-easydense-v0": CapsSACEasyDenseConfig,
    "OfflineMetadrive-mediumsparse-v0": CapsSACMediumSparseConfig,
    "OfflineMetadrive-mediummean-v0": CapsSACMediumMeanConfig,
    "OfflineMetadrive-mediumdense-v0": CapsSACMediumDenseConfig,
    "OfflineMetadrive-hardsparse-v0": CapsSACHardSparseConfig,
    "OfflineMetadrive-hardmean-v0": CapsSACHardMeanConfig,
    "OfflineMetadrive-harddense-v0": CapsSACHardDenseConfig
}