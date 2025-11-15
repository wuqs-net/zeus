"""Optimizers that select the optimum frequency.

This module contains the following pieces:

- [`GlobalFrequencyOptimizer`][zeus.optimizer.frequency.GlobalFrequencyOptimizer]
  is the main class that implements the state machine
  and the logic for profiling frequencies and selecting
  the optimum frequency.
- [`FrequencyMeasurement`][zeus.optimizer.frequency.FrequencyMeasurement] and various
  state classes are helpers that support the state machine.
- [`OptimumSelector`][zeus.optimizer.frequency.OptimumSelector]
  is an abstract base class for selecting the optimum frequency
  from a list of frequency profiling results. There are concrete classes
  that implement different selection strategies, like
  [minimizing energy][zeus.optimizer.frequency.Energy],
  [minimizing time][zeus.optimizer.frequency.Time],
  [minimizing the Zeus time-energy cost][zeus.optimizer.frequency.ZeusCost],
  or [selecting the lowest frequency that meets the given maximum training time slowdown factor][zeus.optimizer.frequency.MaxSlowdownConstraint].
- [`HFGlobalFrequencyOptimizer`][zeus.optimizer.frequency.HFGlobalFrequencyOptimizer]
  is a wrapper for the Hugging Face `TrainerCallback` class that uses `GlobalFrequencyOptimizer`.
"""

from __future__ import annotations

import atexit
from pathlib import Path
from abc import ABC, abstractmethod
import time

from zeus.callback import Callback
from zeus.monitor import ZeusMonitor, FrequencyMonitor
from zeus.utils.framework import all_reduce, is_distributed
from zeus.utils.logging import get_logger
from zeus.utils.metric import zeus_cost
from zeus.utils.pydantic_v1 import BaseModel, PositiveInt, PositiveFloat
from zeus.device import get_gpus
from zeus.device.gpu import ZeusGPUNoPermissionError

from typing import TYPE_CHECKING


class OptimumSelector(ABC):
    """Base class for optimum frequency selectors."""

    @abstractmethod
    def select(self, measurements: list[FrequencyMeasurement]) -> tuple[int, int]:
        """Select the optimal frequency (MHz) from measurements."""


class Energy(OptimumSelector):
    """Selects the frequency that minimizes energy consumption."""

    def select(self, measurements: list[FrequencyMeasurement]) -> tuple[int, int]:
        """Select the optimal frequency (MHz) from measurements."""
        return (lambda x: (x.min_frequency, x.max_frequency))(min(measurements, key=lambda x: x.energy))


class Time(OptimumSelector):
    """Selects the frequency that minimizes training time.

    This may not necessarily choose the maximum frequency, as time profiling
    results can be slightly noisy. However, we believe that's actually better
    because it means that training time is very similar among higher frequencies,
    but lower frequencies will consume less power.
    """

    def select(self, measurements: list[FrequencyMeasurement]) -> tuple[int, int]:
        """Select the optimal frequency (MHz) from measurements."""
        return (lambda x: (x.min_frequency, x.max_frequency))(min(measurements, key=lambda x: x.time))


class ZeusCost(OptimumSelector):
    r"""Selects the frequency that minimizes a linear Zeus time-energy cost function.

    Cost function is $\eta \cdot \text{Energy} + (1 - \eta) \cdot \text{MaxPower} \cdot \text{Time}$.
    """

    def __init__(self, eta_knob: float, world_size: int = 1) -> None:
        r"""Initialize the selector.

        Args:
            eta_knob: The $0 \le \eta \le 1$ knob for the Zeus time-energy cost function.
            world_size: The number of GPUs in the training job. Defaults to 1.
        """
        if eta_knob < 0 or eta_knob > 1:
            raise ValueError("eta_knob must be between 0 and 1, inclusive both sides.")
        if world_size < 1:
            raise ValueError("world_size must be greater than or equal to 1.")

        self.eta_knob = eta_knob
        self.world_size = world_size

    def select(self, measurements: list[FrequencyMeasurement]) -> tuple[int, int]:
        """Select the optimal frequency (MHz) from measurements."""
        max_power = (
            get_gpus(True).getPowerManagementLimitConstraints(0)[1]
            * self.world_size
        )
        zeus_cost_map = {
            (measurement.min_frequency, measurement.max_frequency): zeus_cost(
                energy=measurement.energy,
                time=measurement.time,
                eta_knob=self.eta_knob,
                max_power=max_power,
            )
            for measurement in measurements
        }
        return min(zeus_cost_map, key=lambda x: zeus_cost_map[x])


class MaxSlowdownConstraint(OptimumSelector):
    """Selects the minumum frequency that does not slow down training by more than the given factor."""

    def __init__(self, factor: float) -> None:
        """Initialize the selector.

        Args:
            factor: The maximum allowed slowdown factor. Greater than or equal to 1.0.
        """
        if factor < 1.0:
            raise ValueError(
                f"max_slowdown_factor must be greater than or equal to 1.0. Got {factor}.",
            )

        self.factor = factor

    def select(self, measurements: list[FrequencyMeasurement]) -> tuple[int, int]:
        """Select the optimal frequency (MHz) from measurements."""
        feasible_frequencies = []
        max_frequency = max(measurement.max_frequency for measurement in measurements)
        shortest_time = next(
            measurement.time
            for measurement in measurements
            if measurement.max_frequency == max_frequency
        )
        for measurement in measurements:
            if measurement.time <= self.factor * shortest_time:
                feasible_frequencies.append(measurement.max_frequency)
        return min(feasible_frequencies), min(feasible_frequencies)


class Ready(BaseModel):
    """State for when we are ready to start measuring the next frequency.

    Initial state of the state machine if no previous profiling results were given.
    `Ready` -> `Warmup` after `step`'th `on_step_begin`.
    """

    next_min_frequency: PositiveInt
    next_max_frequency: PositiveInt
    steps: PositiveInt


class Warmup(BaseModel):
    """State for when we are warming up for a frequency.

    `Warmup` -> `Profiling` on the `steps`'th `on_step_begin`.
    `Warmup` -> `Ready` on `on_epoch_end` before `steps`'th `on_step_begin`.
    """

    current_min_frequency: PositiveInt
    current_max_frequency: PositiveInt
    steps: PositiveInt


class Profiling(BaseModel):
    """State for when we are profiling a frequency.

    `Profiling` -> `Warmup` after `steps`'th `on_step_begin` and
        there are still frequencies left to profile.
    `Profiling` -> `Done` after `steps`'th `on_step_begin` and
        there are no more frequencies left to profile.
    `Profiling` -> `Ready` on `on_epoch_end` before `steps`'th `on_step_begin`.
    """

    current_min_frequency: PositiveInt
    current_max_frequency: PositiveInt
    steps: PositiveInt


class Done(BaseModel):
    """State for when we are done profiling all frequencies.

    Initial state of the state machine if previous profiling results were given.
    Final state of the state machine in any case.
    """

    optimal_min_frequency: PositiveInt
    optimal_max_frequency: PositiveInt


class FrequencyMeasurement(BaseModel):
    """POD for GPU energy and time measurements for one frequency lock (MHz)."""

    min_frequency: PositiveInt  # In MHz.
    max_frequency: PositiveInt  # In MHz.
    energy: PositiveFloat
    time: PositiveFloat
    frequency: dict[int, list[tuple[float, float]]]


class _FrequencyMeasurementList(BaseModel):
    """Proxy class to save and load a list of `FrequencyMeasurement`s."""

    measurements: list[FrequencyMeasurement]


class GlobalFrequencyOptimizer(Callback):
    """Optimizer for the frequency knob.

    This optimizer uses the JIT profiling log to determine the optimal frequency.

    ## Usage with distributed data parallelism

    The global frequency optimizer expects one process to control each GPU used for training.
    For instance, `torchrun` will automatically spawn one process for each GPU on the node.
    Correspondingly, the [`ZeusMonitor`][zeus.monitor.energy.ZeusMonitor] instance passed in
    should be monitoring **one GPU**: the one being managed by the current process. The index of
    this GPU would typically match the local rank of the process. In the case of PyTorch, users would have
    called `torch.cuda.set_device` early on, so `torch.cuda.current_device` will give you the GPU index.
    `GlobalFrequencyOptimizer` will internally do an AllReduce across all GPUs to aggregate
    time and energy measurements, and then select the globally optimal frequency.


    ```python
    monitor = ZeusMonitor(gpu_indices=[local_rank])  # pass in local rank to gpu_indices.
    fo = GlobalFrequencyOptimizer(monitor)
    ```
    """

    def __init__(
        self,
        monitor: ZeusMonitor,
        frequency_monitor: FrequencyMonitor | None = None,
        optimum_selector: OptimumSelector | None = None,
        wait_steps: int = 1,
        warmup_steps: int = 10,
        profile_steps: int = 40,
        freq_step: int = 100,
        profile_path: str | Path | None = None,
    ) -> None:
        r"""Initialize the optimizer.

        GPU indices to profile and optimize for are taken from `monitor.gpu_indices`.

        Args:
            monitor: `ZeusMonitor` instance used to profile GPU time and energy consumption.
            optimum_selector: The optimum selector to use. If not given, use `ZeusCost` with \eta=0.5.
            wait_steps: Number of steps to pass by before doing anything at the beginning.
                Useful if you have something like `torch.backends.cudnn.benchmark=True`,
                because the first iteration won't be representative of the rest of the iterations.
            warmup_steps: Number of warmup iterations for each frequency.
            profile_steps: Number of profile iterations for each frequency.
            freq_step: The stride between frequencies to explore, in units of MHz.
            profile_path: If the path points to an existing file, load the profile from the file
                and do not run any profiling. If the path points to a non-existing file, profile
                and save the profile to the file. If `None`, do not save or load any profile.
        """
        # Sanity checks.
        if wait_steps < 0:
            raise ValueError("wait_steps must be non-negative.")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative.")
        if profile_steps <= 0:
            raise ValueError("profile_steps must be positive.")
        if freq_step <= 0:
            raise ValueError("freq_step must be positive.")

        self.monitor = monitor
        self.frequency_monitor = frequency_monitor
        self.optimum_selector = optimum_selector or ZeusCost(
            eta_knob=0.5,
            world_size=len(monitor.gpu_indices),
        )
        self.warmup_steps = warmup_steps
        self.profile_steps = profile_steps
        self.profile_path = (
            Path(profile_path) if isinstance(profile_path, str) else profile_path
        )

        # Setup logging.
        self.logger = get_logger(type(self).__name__)

        gpus = get_gpus(ensure_homogeneous=True)

        # Warn if distributed training is enabled with multiple GPUs monitored.
        if is_distributed() and len(monitor.gpu_indices) > 1:
            self.logger.warning(
                "Distributed training is enabled with %d GPUs monitored. "
                "For distributed training, it is recommended to monitor only one GPU per `ZeusMonitor` instance "
                "since `GlobalFrequencyOptimizer` performs an all-reduce operation internally over all devices.",
                len(monitor.gpu_indices),
            )

        # Set the range of frequencies to explore.
        # Assert that supported frequency ranges are uniform across GPUs.
        freqs = []
        for index in monitor.gpu_indices:
            max_mem_freq = max(gpus.getSupportedMemoryClocks(index))
            freqs.append(gpus.getSupportedGraphicsClocks(index, max_mem_freq))
        if not all(freqs[0] == freq for freq in freqs):
            raise ValueError("Frequency ranges are not uniform across GPUs.")
        self.frequencies: list[int] = []
        last_freq = 0
        for freq in sorted(freqs[0]):
            if freq - last_freq >= freq_step or last_freq == 0:
                self.frequencies.append(freq)
                last_freq = freq

        # Turn on persistence mode.
        try:
            for index in monitor.gpu_indices:
                gpus.setPersistenceMode(index, enabled=True)
        except ZeusGPUNoPermissionError as ze:
            raise RuntimeError(
                "SYS_ADMIN capability is required to modify GPU frequency locks. See "
                "https://ml.energy/zeus/getting_started/#system-privileges "
                "for more information."
            ) from ze
        self.current_frequency = (0, 0)

        # Store `Measurement` objects in a list, one for each frequency.
        self.measurements: list[FrequencyMeasurement] = []

        # State for the profiler state machine.
        self.state: Ready | Warmup | Profiling | Done

        # Initialize JIT profiling states.
        if self.profile_path is None:
            self.logger.info("JIT profiling enabled.")
            self.logger.info("Will wait %d step(s) before profiling.", wait_steps)
            self.state = Ready(
                next_min_frequency=self.frequencies[0],
                next_max_frequency=self.frequencies[0],
                steps=wait_steps + 1
            )
            self.logger.info("Reset frequency lock before starting.")
            self._reset_frequency()
        elif not self.profile_path.exists():
            self.logger.info(
                "JIT Profiling enabled. Profile will be saved to '%s'.",
                str(self.profile_path),
            )
            self.logger.info("Will wait %d step(s) before profiling.", wait_steps)
            self.state = Ready(
                next_min_frequency=self.frequencies[0],
                next_max_frequency=self.frequencies[0],
                steps=wait_steps + 1,
            )
            self.logger.info("Reset frequency lock before starting.")
            self._reset_frequency()
        else:
            self.measurements = _FrequencyMeasurementList.parse_file(
                self.profile_path,
            ).measurements
            # self.measurements = _PowerLimitMeasurementList.model_validate_json(
            #     open(self.profile_path).read(),
            #     strict=True,
            # ).measurements
            self.logger.info(
                "Loaded previous profiling results from '%s'.", str(self.profile_path)
            )
            optimal_min_frequency, optimal_max_frequency = self._compute_optimal_frequency()
            self.logger.info(
                "Optimal frequency is %d MHz - %d MHz.", optimal_min_frequency, optimal_max_frequency
            )
            self.state = Done(
                optimal_min_frequency=optimal_min_frequency,
                optimal_max_frequency=optimal_max_frequency,
            )
            self._set_frequency(self.state.optimal_min_frequency, self.state.optimal_max_frequency)

        # Restore all GPUs back to no frequency lock on exit.
        atexit.register(lambda: self._reset_frequency())

    def on_epoch_end(self) -> None:
        """Mark the end of a training epoch."""
        if isinstance(self.state, Ready):
            pass

        elif isinstance(self.state, (Warmup, Profiling)):
            # Warmup/Profiling stage interrupted by the end of an epoch.
            self.logger.info(
                "%s phase for %d MHz - %d MHz interrupted by the end of a training epoch.",
                type(self.state).__name__,
                self.state.current_min_frequency,
                self.state.current_max_frequency,
            )
            if isinstance(self.state, Profiling):
                self.monitor.end_window(
                    f"__GlobalFrequencyOptimizer_{self.state.current_min_frequency}_{self.state.current_max_frequency}",
                    cancel=True,
                )
            self.state = Ready(
                next_min_frequency=self.state.current_min_frequency,
                next_max_frequency=self.state.current_max_frequency,
                steps=1
            )
            self._reset_frequency()

        elif isinstance(self.state, Done):
            pass

    def on_step_begin(self) -> None:
        """Mark the beginning of a training step."""
        if isinstance(self.state, Ready):
            self.state.steps -= 1
            if self.state.steps == 0:
                self.logger.info(
                    "Starting warmup for frequency %d - %d MHz.",
                    self.state.next_min_frequency,
                    self.state.next_max_frequency,
                )
                self._set_frequency(self.state.next_min_frequency, self.state.next_max_frequency)
                self.state = Warmup(
                    current_min_frequency=self.state.next_min_frequency,
                    current_max_frequency=self.state.next_max_frequency,
                    steps=self.warmup_steps,
                )

        elif isinstance(self.state, Warmup):
            self.state.steps -= 1
            if self.state.steps == 0:
                self.logger.info(
                    "Starting actual profiling for frequency %d - %d MHz.",
                    self.state.current_min_frequency,
                    self.state.current_max_frequency,
                )
                self.state = Profiling(
                    current_min_frequency=self.state.current_min_frequency,
                    current_max_frequency=self.state.current_max_frequency,
                    steps=self.profile_steps,
                )
                self.start_time = time.time()
                self.monitor.begin_window(
                    f"__GlobalFrequencyOptimizer_{self.state.current_min_frequency}_{self.state.current_max_frequency}",
                )

        elif isinstance(self.state, Profiling):
            self.state.steps -= 1
            if self.state.steps == 0:
                measurement = self.monitor.end_window(
                    f"__GlobalFrequencyOptimizer_{self.state.current_min_frequency}_{self.state.current_max_frequency}",
                )
                freq_measurement = dict()
                if self.frequency_monitor is not None:
                    freq_measurement = self.frequency_monitor.get_frequency_timeline(0, self.start_time, time.time())
                self.logger.info(
                    "Finished profiling for frequency %d - %d MHz.",
                    self.state.current_min_frequency,
                    self.state.current_max_frequency,
                )

                self.measurements.append(
                    FrequencyMeasurement(
                        min_frequency=self.state.current_min_frequency,
                        max_frequency=self.state.current_max_frequency,
                        energy=sum(
                            all_reduce(
                                list(measurement.gpu_energy.values()), operation="sum"
                            )
                        ),
                        time=max(all_reduce([measurement.time], operation="max")),
                        frequency=freq_measurement,
                    )
                )
                # If we're done profiling all frequencies, compute the optimal
                # frequency and transition to the Done state. Otherwise, move
                # on to the Warmup phase for the next frequency.
                current_frequency_index = self.frequencies.index(
                    self.state.current_max_frequency
                )
                if current_frequency_index == len(self.frequencies) - 1:
                    optimal_min_frequency, optimal_max_frequency =  self._compute_optimal_frequency()
                    self.state = Done(
                        optimal_min_frequency=optimal_min_frequency,
                        optimal_max_frequency=optimal_max_frequency,
                    )
                    self._set_frequency(self.state.optimal_min_frequency, self.state.optimal_max_frequency)
                    self._save_profile()
                else:
                    next_frequency = self.frequencies[current_frequency_index + 1]
                    self.logger.info(
                        "Starting warmup for frequency %d - %d MHz.",
                        next_frequency, next_frequency
                    )
                    self._set_frequency(next_frequency, next_frequency)
                    self.state = Warmup(
                        current_min_frequency=next_frequency,
                        current_max_frequency=next_frequency,
                        steps=self.warmup_steps,
                    )

        elif isinstance(self.state, Done):
            pass

    def _set_frequency(self, min_frequency: int, max_frequency: int) -> None:
        """Set the frequency for all GPUs.

        Args:
            min_frequency: The minimum frequency to set, in MHz.
            max_frequency: The maximum frequency to set, in MHz.
        """
        gpus = get_gpus()
        self.logger.info("Setting frequency to %d MHz - %d MHz.", min_frequency, max_frequency)
        if self.current_frequency == (min_frequency, max_frequency):
            return
        for index in self.monitor.gpu_indices:
            gpus.setGpuLockedClocks(index, min_frequency, max_frequency)
        self.current_frequency = (min_frequency, max_frequency)
    
    def _reset_frequency(self) -> None:
        """Reset the frequency lock for all GPUs."""
        gpus = get_gpus()
        self.logger.info("Resetting frequency lock.")
        for index in self.monitor.gpu_indices:
            gpus.resetGpuLockedClocks(index)
        self.current_frequency = (0, 0)

    def _compute_optimal_frequency(self) -> tuple[int, int]:
        """Compute the optimal frequency in MHz."""
        optimal_min_frequency, optimal_max_frequency = self.optimum_selector.select(self.measurements)
        self.logger.info("Optimal frequency is %d MHz.", optimal_max_frequency)
        return optimal_min_frequency, optimal_max_frequency

    def _save_profile(self) -> None:
        """Save JIT profiling results and the optimal frequency to a JSON file."""
        if self.profile_path is None:
            return

        assert isinstance(self.state, Done)
        with self.profile_path.open("w", encoding="utf-8") as f:
            f.write(
                _FrequencyMeasurementList(measurements=self.measurements).json(
                    indent=4
                ),
            )
        self.logger.info("JIT profiling results saved to '%s'.", str(self.profile_path))


# Only import HuggingFace Classes when type checking, to avoid hard dependency on HuggingFace Transformers
if TYPE_CHECKING:
    from transformers.training_args import TrainingArguments
    from transformers.trainer_callback import TrainerState, TrainerControl
    from transformers.modeling_utils import PreTrainedModel

try:
    from transformers.trainer_callback import TrainerCallback

    transformers_available = True
except ModuleNotFoundError:
    transformers_available = False
    TrainerCallback = object  # Fallback base class


class HFGlobalPowerLimitOptimizer(TrainerCallback):  # type: ignore
    """[Wrapped for Hugging Face Trainer Callback] Optimizer for the power limit knob.

    This optimizer uses the JIT profiling log to determine the optimal power limit.
    See [`GlobalPowerLimitOptimizer`][zeus.optimizer.power_limit.GlobalPowerLimitOptimizer]
    for the underlying optimizer implementation.
    """

    def __init__(
        self,
        monitor: ZeusMonitor,
        optimum_selector: OptimumSelector | None = None,
        wait_steps: int = 1,
        warmup_steps: int = 10,
        profile_steps: int = 40,
        pl_step: int = 25,
        profile_path: str | Path | None = None,
    ) -> None:
        r"""Initialize the optimizer.

        GPU indices to profile and optimize for are taken from `monitor.gpu_indices`.

        Args:
            monitor: `ZeusMonitor` instance used to profile GPU time and energy consumption.
            optimum_selector: The optimum selector to use. If not given, use `ZeusCost` with \eta=0.5.
            wait_steps: Number of steps to pass by before doing anything at the beginning.
                Useful if you have something like `torch.backends.cudnn.benchmark=True`,
                because the first iteration won't be representative of the rest of the iterations.
            warmup_steps: Number of warmup iterations for each power limit.
            profile_steps: Number of profie iterations for each power limit.
            pl_step: The stride between power limits to explore, in unites of Watts.
            profile_path: If the path points to an existing file, load the profile from the file
                and do not run any profiling. If the path points to a non-existing file, profile
                and save the profile to the file. If `None`, do not save or load any profile.
        """
        if not transformers_available:
            raise ImportError(
                "The transformers package is not installed. Please install it to use the HFGlobalPowerLimitOptimizer."
            )

        self.optimizer = GlobalPowerLimitOptimizer(
            monitor=monitor,
            optimum_selector=optimum_selector,
            wait_steps=wait_steps,
            warmup_steps=warmup_steps,
            profile_steps=profile_steps,
            pl_step=pl_step,
            profile_path=profile_path,
        )

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: PreTrainedModel,
        **kwargs,
    ) -> None:
        """Mark the end of a training epoch."""
        self.optimizer.on_epoch_end()

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: PreTrainedModel,
        **kwargs,
    ) -> None:
        """Mark the beginning of a training step."""
        self.optimizer.on_step_begin()
