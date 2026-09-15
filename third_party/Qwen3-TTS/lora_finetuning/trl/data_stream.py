"""Robust streaming access for WebDataset rows with heterogeneous JSON."""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from itertools import islice
import tarfile
import time
from typing import Any, Iterable
import warnings


def _retryable_stream_error(error: Exception) -> bool:
    if isinstance(
        error,
        (tarfile.ReadError, EOFError, ConnectionError, TimeoutError, OSError),
    ):
        return True
    # HTTP clients use separate exception hierarchies, depending on the Hub
    # filesystem backend selected by the installed package versions.
    root_module = type(error).__module__.partition(".")[0]
    return root_module in {
        "aiohttp",
        "fsspec",
        "httpcore",
        "httpx",
        "huggingface_hub",
        "requests",
        "urllib3",
    }


def _shard_label(kwargs: dict[str, Any]) -> str:
    for name in ("tar_paths", "files", "paths"):
        value = kwargs.get(name)
        if isinstance(value, (list, tuple)) and value:
            return str(value[0])
        if value:
            return str(value)
    return "<unknown-shard>"


@dataclass
class RetryingExamplesGenerator:
    """Retry one shard without replaying examples already yielded from it."""

    generate_examples_fn: Any
    max_retries: int = 3
    backoff_sec: float = 1.0
    skip_failed_shards: bool = True

    def __call__(self, **kwargs: Any):
        emitted = 0
        shard = _shard_label(kwargs)
        for attempt in range(self.max_retries + 1):
            try:
                iterator = self.generate_examples_fn(**kwargs)
                for key_example in islice(iterator, emitted, None):
                    emitted += 1
                    yield key_example
                return
            except Exception as error:
                if not _retryable_stream_error(error):
                    raise
                if attempt >= self.max_retries:
                    if not self.skip_failed_shards:
                        raise
                    warnings.warn(
                        f"[trl-data-stream] skipping {shard} after "
                        f"{self.max_retries} retries and {emitted} emitted rows: {error!r}",
                        RuntimeWarning,
                    )
                    return
                delay = self.backoff_sec * (2**attempt)
                warnings.warn(
                    f"[trl-data-stream] retry {attempt + 1}/{self.max_retries} "
                    f"for {shard} after {emitted} emitted rows in {delay:.1f}s: {error!r}",
                    RuntimeWarning,
                )
                if delay > 0:
                    time.sleep(delay)


def without_feature_casting(
    dataset: Any,
    required_columns: Iterable[str],
    *,
    max_retries: int = 3,
    retry_backoff_sec: float = 1.0,
    skip_failed_shards: bool = True,
) -> Any:
    """Keep WebDataset sharding while bypassing an invalid inferred schema.

    ``youtube_balalaika`` contains a few empty strings in JSON members that the
    Hub schema declares as floats. ``datasets`` otherwise raises before a map
    function can reject or normalize such a row. Reusing the raw examples
    iterable preserves its 966 shards, authentication and worker sharding; an
    untyped ``DatasetInfo`` prevents the premature nested float conversion.

    This deliberately targets the pinned datasets 5.x runtime used by this
    training folder. Fail loudly if its streaming internals change.
    """

    features = getattr(dataset, "features", None)
    missing = [name for name in required_columns if not features or name not in features]
    if missing:
        raise ValueError(f"streaming dataset is missing required columns: {missing}")
    if not hasattr(dataset, "_ex_iterable") or not hasattr(dataset, "_token_per_repo_id"):
        raise RuntimeError(
            "The installed datasets runtime no longer exposes the raw streaming iterable"
        )

    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    if retry_backoff_sec < 0:
        raise ValueError("retry_backoff_sec must be non-negative")

    from datasets import IterableDataset
    from datasets.iterable_dataset import ExamplesIterable

    source_iterable = dataset._ex_iterable
    if not isinstance(source_iterable, ExamplesIterable):
        raise RuntimeError(
            "Expected a shard-aware datasets ExamplesIterable for WebDataset streaming"
        )
    retrying_iterable = ExamplesIterable(
        RetryingExamplesGenerator(
            source_iterable.generate_examples_fn,
            max_retries=max_retries,
            backoff_sec=retry_backoff_sec,
            skip_failed_shards=skip_failed_shards,
        ),
        source_iterable.kwargs,
        source_iterable.generate_more_kwargs_fn,
        source_iterable._sleep_on_threads_shutdown,
    )

    info = copy(dataset.info)
    info.features = None
    return IterableDataset(
        retrying_iterable,
        info=info,
        split=dataset.split,
        formatting=None,
        distributed=getattr(dataset, "_distributed", None),
        token_per_repo_id=dataset._token_per_repo_id,
    )


def load_untyped_streaming_dataset(
    dataset_name: str,
    *,
    split: str,
    revision: str,
    token: str | None,
    required_columns: Iterable[str],
    max_retries: int = 3,
    retry_backoff_sec: float = 1.0,
    skip_failed_shards: bool = True,
) -> Any:
    """Load a pinned Hub stream without applying its nested feature schema."""

    from datasets import load_dataset

    dataset = load_dataset(
        path=dataset_name,
        split=split,
        revision=revision,
        streaming=True,
        token=token,
    )
    return without_feature_casting(
        dataset,
        required_columns,
        max_retries=max_retries,
        retry_backoff_sec=retry_backoff_sec,
        skip_failed_shards=skip_failed_shards,
    )


__all__ = [
    "RetryingExamplesGenerator",
    "load_untyped_streaming_dataset",
    "without_feature_casting",
]
