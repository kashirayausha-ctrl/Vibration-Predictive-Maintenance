from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class SignalConfig:
    """Configuration for vibration signal processing."""

    sampling_rate_hz: Optional[float] = None
    window_duration_s: float = 1.0
    overlap_ratio: float = 0.5

    # Filter configuration.
    # Disabled until the real sensor spectrum is characterized.
    filter_enabled: bool = False
    low_cutoff_hz: Optional[float] = None
    high_cutoff_hz: Optional[float] = None

    def __post_init__(self) -> None:
        if self.window_duration_s <= 0:
            raise ValueError("window_duration_s must be greater than 0.")

        if not 0.0 <= self.overlap_ratio < 1.0:
            raise ValueError("overlap_ratio must be in the range [0, 1).")

        if self.sampling_rate_hz is not None and self.sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be greater than 0.")

        if self.filter_enabled:
            if self.low_cutoff_hz is None or self.high_cutoff_hz is None:
                raise ValueError(
                    "Both low_cutoff_hz and high_cutoff_hz are required "
                    "when filtering is enabled."
                )

            if self.low_cutoff_hz <= 0:
                raise ValueError("low_cutoff_hz must be greater than 0.")

            if self.high_cutoff_hz <= self.low_cutoff_hz:
                raise ValueError(
                    "high_cutoff_hz must be greater than low_cutoff_hz."
                )


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for machine-learning models."""

    random_state: int = 42
    test_size: float = 0.20
    validation_size: float = 0.20

    def __post_init__(self) -> None:
        if not 0.0 < self.test_size < 1.0:
            raise ValueError("test_size must be between 0 and 1.")

        if not 0.0 < self.validation_size < 1.0:
            raise ValueError("validation_size must be between 0 and 1.")

        if self.test_size + self.validation_size >= 1.0:
            raise ValueError(
                "test_size + validation_size must be less than 1."
            )


@dataclass(frozen=True)
class ProjectConfig:
    """Project-wide configuration."""

    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
    )

    signal: SignalConfig = field(default_factory=SignalConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    @property
    def models_dir(self) -> Path:
        return self.project_root / "ai" / "models"

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    def ensure_directories(self) -> None:
        """Create required project directories if they don't exist."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)


CONFIG = ProjectConfig()