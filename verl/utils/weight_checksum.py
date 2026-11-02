# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pure helpers for merging and diffing rank-qualified weight digests.

The weight checker in the rollout engine reports one digest map per worker,
keyed by a rank-qualified parameter name::

    dp{dp}:pp{pp}:pcp{pcp}:tp{tp}:{parameter_name}

The functions here merge those maps and diff two of them. They carry no heavy
imports on purpose, so the logic that decides "did the weights land" stays
unit-testable without a GPU or an inference engine.
"""

from typing import Iterable


def merge_worker_checksums(per_worker: Iterable[dict[str, str]]) -> dict[str, str]:
    """Merge one digest map per worker into a single rank-qualified map.

    A key carries the parallel ranks of the worker that produced it, so the
    same parameter appears once per shard and two workers must never report the
    same key. An overlap means a worker failed to qualify its keys, and it is
    raised rather than resolved: silently keeping one of the two values would
    let the later comparison pass for the wrong reason.

    Args:
        per_worker: One digest map per worker, as the engine reports them.

    Returns:
        The merged map.

    Raises:
        RuntimeError: If two workers report the same key.
    """
    merged: dict[str, str] = {}
    for worker_checksums in per_worker:
        duplicate_keys = merged.keys() & worker_checksums.keys()
        if duplicate_keys:
            duplicates = ", ".join(sorted(duplicate_keys)[:5])
            raise RuntimeError(f"Duplicate weight checksum keys: {duplicates}")
        merged.update(worker_checksums)
    return merged


def compare_checksums(baseline: dict[str, str], current: dict[str, str]) -> tuple[bool, list[str]]:
    """Diff two digest maps, treating a one-sided key as a mismatch.

    A key that only one side carries is reported: the parameter was either
    dropped by the update or added by it, and either way the two states do not
    describe the same weights.

    Args:
        baseline: The digests recorded before the reset.
        current: The digests read back after the update.

    Returns:
        Whether every key matches, and the sorted keys that do not.
    """
    mismatches = sorted(key for key in baseline.keys() | current.keys() if baseline.get(key) != current.get(key))
    return not mismatches, mismatches
