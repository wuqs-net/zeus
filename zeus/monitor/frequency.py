"""Monitor the frequency of GPUs."""

from __future__ import annotations

import bisect
import collections
import multiprocessing as mp
import weakref
from time import time, sleep
from dataclasses import dataclass
from queue import Empty
from typing import TYPE_CHECKING

from zeus.device.gpu.common import ZeusGPUNotSupportedError
from zeus.utils.logging import get_logger
from zeus.device import get_gpus

if TYPE_CHECKING:
    from multiprocessing.synchronize import Event as EventClass
    from multiprocessing.context import SpawnProcess

logger = get_logger(__name__)


def _cleanup_frequency_process(
    stop_event: EventClass,
    process: SpawnProcess,
) -> None:
    """Idempotent cleanup function for frequency monitoring process."""
    # Signal the process to stop
    stop_event.set()

    # Wait for the process to complete
    if process.is_alive():
        process.join(timeout=2.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=1.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=1.0)


@dataclass
class FrequencySample:
    """A single frequency measurement sample."""

    timestamp: float
    gpu_index: int
    frequency_mhz: int


class FrequencyMonitor:
    """Monitor GPU frequency over time.

    This class provides:
    1. Continuous frequency monitoring in a background process
    2. Timeline export with deduplication
    3. Point-in-time frequency queries

    !!! Note
        The current implementation only supports cases where all GPUs are homogeneous
        (i.e., the same model).

    !!! Warning
        Since the monitor spawns child processes, **it should not be instantiated as a global variable**.
        Refer to the "Safe importing of main module" section in the
        [Python documentation](https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods)
        for more details.
    """

    def __init__(
        self,
        gpu_indices: list[int] | None = None,
        update_period: float = 1.0,
        max_samples_per_gpu: int | None = None,
    ) -> None:
        """Initialize the frequency monitor.

        Args:
            gpu_indices: Indices of the GPUs to monitor. If None, monitor all GPUs.
            update_period: Update period of the frequency monitor in seconds.
                Defaults to 1.0 second. Frequency typically doesn't change as
                rapidly as power, so a longer update period is reasonable.
            max_samples_per_gpu: Maximum number of frequency samples to keep per GPU
                in memory. If None (default), unlimited samples are kept.
        """
        if gpu_indices is not None and not gpu_indices:
            raise ValueError("`gpu_indices` must be either `None` or non-empty")

        # Get GPUs
        gpus = get_gpus(ensure_homogeneous=True)

        # Configure GPU indices
        self.gpu_indices = (
            gpu_indices if gpu_indices is not None else list(range(len(gpus)))
        )
        if not self.gpu_indices:
            raise ValueError("At least one GPU index must be specified")
        logger.info("Monitoring frequency of GPUs %s", self.gpu_indices)

        self.update_period = update_period

        # Frequency samples are collected for each device index.
        self.frequency_samples: dict[int, collections.deque[FrequencySample]] = {}
        for gpu_idx in self.gpu_indices:
            self.frequency_samples[gpu_idx] = collections.deque(
                maxlen=max_samples_per_gpu
            )

        # Spawn frequency collector process
        ctx = mp.get_context("spawn")
        self.frequency_queue = ctx.Queue()
        self.frequency_ready_event = ctx.Event()
        self.frequency_stop_event = ctx.Event()
        self.frequency_process = ctx.Process(
            target=_frequency_polling_process,
            kwargs=dict(
                gpu_indices=self.gpu_indices,
                data_queue=self.frequency_queue,
                ready_event=self.frequency_ready_event,
                stop_event=self.frequency_stop_event,
                update_period=update_period,
            ),
            daemon=True,
            name="zeus-frequency-monitor",
        )
        self.frequency_process.start()

        # Cleanup function
        self._finalizer = weakref.finalize(
            self,
            _cleanup_frequency_process,
            self.frequency_stop_event,
            self.frequency_process,
        )

        # Wait for subprocess to signal it's ready
        logger.info("Waiting for frequency monitoring subprocess to be ready...")
        if not self.frequency_ready_event.wait(timeout=10.0):
            logger.warning(
                "Frequency monitor subprocess did not signal ready within timeout"
            )
        logger.info("Frequency monitoring subprocess is ready")

    def stop(self) -> None:
        """Stop the monitoring process."""
        if self._finalizer.alive:
            self._finalizer()

    def _process_frequency_queue_data(self) -> None:
        """Process all pending frequency samples from the queue."""
        if not hasattr(self, "frequency_queue"):
            return

        while True:
            try:
                sample = self.frequency_queue.get_nowait()
                if sample == "STOP":
                    break
                assert isinstance(sample, FrequencySample)
                self.frequency_samples[sample.gpu_index].append(sample)
            except Empty:
                break

    def get_frequency_timeline(
        self,
        gpu_index: int | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> dict[int, list[tuple[float, int]]]:
        """Get frequency timeline for specific GPU(s).

        Args:
            gpu_index: Specific GPU index, or None for all GPUs
            start_time: Start time filter (unix timestamp)
            end_time: End time filter (unix timestamp)

        Returns:
            Dictionary mapping GPU indices to timeline data.
            Timeline data is list of (timestamp, frequency_mhz) tuples.
        """
        # Process any pending queue data
        self._process_frequency_queue_data()

        # Determine which GPUs to query
        target_gpus = [gpu_index] if gpu_index is not None else self.gpu_indices

        result = {}
        for gpu_idx in target_gpus:
            if gpu_idx not in self.frequency_samples:
                continue

            # Extract timeline from samples
            timeline = []
            for sample in self.frequency_samples[gpu_idx]:
                # Apply time filters
                if start_time is not None and sample.timestamp < start_time:
                    continue
                if end_time is not None and sample.timestamp > end_time:
                    continue

                timeline.append((sample.timestamp, sample.frequency_mhz))

            # Sort by timestamp
            timeline.sort(key=lambda x: x[0])
            result[gpu_idx] = timeline

        return result

    def get_frequency(self, time: float | None = None) -> dict[int, int] | None:
        """Get the GPU frequency at a specific time point.

        Args:
            time: Time point to get the frequency at. If None, get the frequency
                at the last recorded time point.

        Returns:
            A dictionary mapping GPU indices to the frequency of the GPU at the
            specified time point. If there are no frequency readings, return None.
        """
        # Process any pending queue data
        self._process_frequency_queue_data()

        result = {}
        for gpu_idx in self.gpu_indices:
            samples = self.frequency_samples[gpu_idx]
            if not samples:
                return None

            if time is None:
                # Get the most recent sample
                latest_sample = samples[-1]
                result[gpu_idx] = latest_sample.frequency_mhz
            else:
                # Find the closest sample to the requested time using bisect
                timestamps = [sample.timestamp for sample in samples]
                pos = bisect.bisect_left(timestamps, time)

                if pos == 0:
                    closest_sample = samples[0]
                elif pos == len(samples):
                    closest_sample = samples[-1]
                else:
                    # Check the closest sample before and after the requested time
                    before = samples[pos - 1]
                    after = samples[pos]
                    closest_sample = (
                        before
                        if time - before.timestamp <= after.timestamp - time
                        else after
                    )
                result[gpu_idx] = closest_sample.frequency_mhz

        return result


def _frequency_polling_process(
    gpu_indices: list[int],
    data_queue: mp.Queue,
    ready_event: EventClass,
    stop_event: EventClass,
    update_period: float,
) -> None:
    """Polling process for GPU frequency with deduplication."""
    try:
        # Get GPUs
        gpus = get_gpus()

        # Track previous frequency values for deduplication
        prev_frequency: dict[int, int] = {}

        # Signal that this process is ready to start monitoring
        ready_event.set()

        # Start polling loop
        while not stop_event.is_set():
            timestamp = time()

            for gpu_index in gpu_indices:
                try:
                    frequency_mhz = gpus.getGpuFrequency(gpu_index)

                    # Deduplication: only send if frequency changed
                    # if (
                    #     gpu_index in prev_frequency
                    #     and prev_frequency[gpu_index] == frequency_mhz
                    # ):
                    #     continue

                    prev_frequency[gpu_index] = frequency_mhz

                    # Create and send frequency sample
                    sample = FrequencySample(
                        timestamp=timestamp,
                        gpu_index=gpu_index,
                        frequency_mhz=frequency_mhz,
                    )

                    data_queue.put(sample)
                except ZeusGPUNotSupportedError as e:
                    logger.warning(
                        "GPU %d frequency reading not supported: %s",
                        gpu_index,
                        e,
                    )
                    # Don't keep trying if it's not supported
                    break
                except Exception as e:
                    logger.exception(
                        "Error polling frequency for GPU %d: %s",
                        gpu_index,
                        e,
                    )
                    raise

            # Sleep for the remaining time
            elapsed = time() - timestamp
            sleep_time = update_period - elapsed
            if sleep_time > 0:
                sleep(sleep_time)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.exception(
            "Exiting frequency polling process due to error: %s",
            e,
        )
        raise e
    finally:
        # Send stop signal
        data_queue.put("STOP")
