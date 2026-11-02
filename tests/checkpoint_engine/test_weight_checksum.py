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
"""Unit tests for the weight-checker digest helpers.

These cover the decisions that make the check trustworthy -- a duplicate key is
an error rather than a coin flip, and a key present on only one side is a
mismatch -- without needing a GPU, an inference engine or Ray.
"""

import asyncio

import pytest

from verl.utils.weight_checksum import compare_checksums, merge_worker_checksums


class _Awaitable:
    """Stand-in for a Ray ObjectRef: something ``await`` can resolve."""

    def __init__(self, value):
        self._value = value

    def __await__(self):
        async def _resolve():
            return self._value

        return _resolve().__await__()


class _RemoteMethod:
    def __init__(self, value):
        self._value = value

    def remote(self, *args, **kwargs):
        return _Awaitable(self._value)


class _FakeServer:
    """A server actor that answers every check_weights call the same way."""

    def __init__(self, value):
        self.check_weights = _RemoteMethod(value)


def test_merge_keeps_parameters_of_distinct_ranks_apart():
    merged = merge_worker_checksums([{"dp0:tp0:w": "a"}, {"dp0:tp1:w": "b"}, {"dp1:tp0:w": "c"}])
    assert merged == {"dp0:tp0:w": "a", "dp0:tp1:w": "b", "dp1:tp0:w": "c"}


def test_merge_of_nothing_is_empty():
    assert merge_worker_checksums([]) == {}


def test_merge_rejects_duplicate_keys_even_when_the_values_match():
    """A collision means a worker dropped its rank prefix; that must not pass."""
    with pytest.raises(RuntimeError, match="Duplicate weight checksum keys"):
        merge_worker_checksums([{"dp0:tp0:w": "a"}, {"dp0:tp0:w": "a"}])


@pytest.mark.parametrize(
    "baseline, current, expected",
    [
        ({"dp0:tp0:w": "a"}, {"dp0:tp0:w": "a"}, []),
        ({"dp0:tp0:w": "a"}, {"dp0:tp0:w": "b"}, ["dp0:tp0:w"]),
        # The update dropped a tensor.
        ({"dp0:tp0:w": "a", "dp0:tp0:v": "b"}, {"dp0:tp0:w": "a"}, ["dp0:tp0:v"]),
        # The update added one.
        ({"dp0:tp0:w": "a"}, {"dp0:tp0:w": "a", "dp0:tp0:u": "c"}, ["dp0:tp0:u"]),
        # Only one worker's rank came back.
        ({"dp0:tp0:w": "a", "dp0:tp1:w": "b"}, {"dp0:tp0:w": "a"}, ["dp0:tp1:w"]),
        ({}, {}, []),
    ],
)
def test_compare_treats_a_one_sided_key_as_a_mismatch(baseline, current, expected):
    matched, mismatches = compare_checksums(baseline, current)
    assert mismatches == expected
    assert matched is not expected


def test_compare_does_not_mutate_its_inputs():
    baseline = {"dp0:tp0:w": "a"}
    current = {"dp0:tp0:w": "b"}
    compare_checksums(baseline, current)
    assert baseline == {"dp0:tp0:w": "a"}
    assert current == {"dp0:tp0:w": "b"}


def test_replica_check_weights_drops_headless_servers():
    """Headless actors answer None and must not be counted as passing."""
    from verl.workers.rollout.replica import RolloutReplica

    replica = object.__new__(RolloutReplica)
    replica.servers = [_FakeServer({"success": True, "num_keys": 2}), _FakeServer(None)]

    results = asyncio.run(RolloutReplica.check_weights(replica, "snapshot"))

    assert results == [{"success": True, "num_keys": 2}]


def test_replica_check_weights_without_servers_is_empty():
    from verl.workers.rollout.replica import RolloutReplica

    replica = object.__new__(RolloutReplica)
    replica.servers = []

    assert asyncio.run(RolloutReplica.check_weights(replica, "compare")) == []
