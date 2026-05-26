# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64
import hashlib
import io
from collections import OrderedDict
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import TYPE_CHECKING

import psutil
import torch

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorMetadata,
    ECConnectorRole,
)

if TYPE_CHECKING:
    from vllm.v1.request import Request
    from vllm.platforms import current_platform

logger = init_logger(__name__)

GB = 1024 ** 3
MB = 1024 ** 2
EC_CACHE_KEY = "ec_cache"
BYTE_LENGTH = 8


def _mm_hash_to_sha256(mm_hash: str) -> str:
    if len(mm_hash) > 1000:
        raise ValueError("mm_hash exceeds the safe length limit for a shared memory identifier.")
    # Filtering out invalid characters, such as '/' and '\0'
    sanitized = mm_hash.replace('/', '_').replace('\0', '_')
    hash_hex = hashlib.sha256(sanitized.encode()).hexdigest()
    # Use Base64 encoding to avoid special characters, which is shorter and more secure.
    return base64.urlsafe_b64encode(hash_hex.encode()).decode().rstrip('=')


@dataclass
class ECSharedMemoryConnectorMetadata(ECConnectorMetadata):
    mm_hashes: list[str]

    def __init__(self) -> None:
        self.mm_hashes = []

    def add_mm_hash(self, mm_hash: str) -> None:
        self.mm_hashes.append(mm_hash)


class ECSharedMemoryConnector(ECConnectorBase):
    """EC connector backed by POSIX shared memory segments."""

    def __init__(self, vllm_config: VllmConfig, role: ECConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)
        self.count = 0
        transfer_config = vllm_config.ec_transfer_config
        if transfer_config is None:
            raise ValueError("ec_transfer_config must be set for ECConnectorBase")
        available_mem = psutil.virtual_memory().available
        _max_bytes = transfer_config.get_from_extra_config("ec_shared_memory_max_bytes", 10) * GB
        if _max_bytes > available_mem * 0.1:
            logger.info(f"shm ecconnector max bytes > available_mem * 0.1")
        self._max_bytes = min(int(available_mem * 0.1), _max_bytes)
        logger.info(f"shm ecconnector max bytes:{self._max_bytes / GB}GB")
        self._mm_hashes_need_loads: set[str] = set()
        self._mm_hash_sizes: dict[str, int] = {}
        self._mm_hash_refcounts: dict[str, int] = {}
        self._lru: OrderedDict[str, None] = OrderedDict()

    @staticmethod
    def _serialize_cache(tensor: torch.Tensor) -> bytes:
        buffer = io.BytesIO()
        torch.save({EC_CACHE_KEY: tensor.detach().cpu()}, buffer)
        return buffer.getvalue()

    @staticmethod
    def _deserialize_cache(payload: bytes) -> torch.Tensor:
        buffer = io.BytesIO(payload)
        from vllm.platforms import current_platform
        data = torch.load(buffer, map_location=current_platform.device_type)
        return data[EC_CACHE_KEY]

    def _touch_lru(self, mm_hash: str) -> None:
        if mm_hash in self._lru:
            self._lru.move_to_end(mm_hash)
        else:
            self._lru[mm_hash] = None

    def _current_bytes(self) -> int:
        return sum(self._mm_hash_sizes.values())

    def _evict_if_needed(self, extra_bytes: int) -> None:
        if not self.is_producer or self.role != ECConnectorRole.WORKER:
            return
        while self._current_bytes() + extra_bytes > self._max_bytes and self._lru:
            evict_hash, _ = self._lru.popitem(last=False)
            logger.warning(f"use total memory: {self._current_bytes() / GB}, shm evict_hash:{evict_hash}")
            self._unlink_shm(evict_hash)

    def _unlink_shm(self, mm_hash: str) -> None:
        try:
            shm = shared_memory.SharedMemory(name=_mm_hash_to_sha256(mm_hash))
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            logger.debug(f"SHM {mm_hash} already unlinked.")
        except Exception as ex:
            logger.error(f"Unlink failed for {mm_hash}: {ex}", exc_info=True)
        self._mm_hash_sizes.pop(mm_hash, None)
        # Decrease the reference count. If the reference count is 1, the value is cleared to avoid negative values.
        current_count = self._mm_hash_refcounts.get(mm_hash, 0)
        if current_count <= 1:
            self._mm_hash_refcounts.pop(mm_hash, None)
        else:
            self._mm_hash_refcounts[mm_hash] = current_count - 1

    def start_load_caches(self, encoder_cache, **kwargs) -> None:
        metadata: ECConnectorMetadata = self._get_connector_metadata()
        assert isinstance(metadata, ECSharedMemoryConnectorMetadata)
        if metadata is None:
            logger.warning("In connector.start_load_caches, but the connector metadata is None")
            return
        for mm_hash in metadata.mm_hashes:
            if mm_hash in encoder_cache:
                continue
            logger.debug(f"start load caches for mm_hash:{mm_hash}")
            try:
                shm = shared_memory.SharedMemory(name=_mm_hash_to_sha256(mm_hash))
            except FileNotFoundError:
                logger.warning("EC shared memory miss for hash %s", mm_hash)
                continue
            except (PermissionError, OSError) as exc:
                logger.warning(f"Failed to access shared memory for hash {mm_hash}: {exc}")
                continue
            try:
                size_bytes = int.from_bytes(shm.buf[:BYTE_LENGTH], "little")
                payload = bytes(shm.buf[BYTE_LENGTH: BYTE_LENGTH + size_bytes])
                encoder_cache[mm_hash] = self._deserialize_cache(payload)
                self._touch_lru(mm_hash)
                self._mm_hash_refcounts[mm_hash] = self._mm_hash_refcounts.get(mm_hash, 0) + 1
            finally:
                shm.close()

    def save_caches(self, encoder_cache, mm_hash, **kwargs) -> None:
        if not self.is_producer or self.role != ECConnectorRole.WORKER:
            return
        # 只在rank0保存
        if get_tensor_model_parallel_rank() != 0:
            return
        payload = self._serialize_cache(encoder_cache[mm_hash])
        size_bytes = len(payload)
        self._evict_if_needed(size_bytes)
        logger.debug(f"save caches for hash name:{mm_hash}")
        try:
            shm = shared_memory.SharedMemory(name=_mm_hash_to_sha256(mm_hash), create=True,
                                             size=BYTE_LENGTH + size_bytes)
        except FileExistsError:
            shm = shared_memory.SharedMemory(name=_mm_hash_to_sha256(mm_hash))
        try:
            shm.buf[:BYTE_LENGTH] = size_bytes.to_bytes(BYTE_LENGTH, "little")
            shm.buf[BYTE_LENGTH: BYTE_LENGTH + size_bytes] = payload
            self._mm_hash_sizes[mm_hash] = BYTE_LENGTH + size_bytes
            self._touch_lru(mm_hash)
            self._mm_hash_refcounts[mm_hash] = self._mm_hash_refcounts.get(mm_hash, 0) + 1
        finally:
            logger.debug(f"mm_hash:{mm_hash} shm size:{shm.size} {shm.size / MB:.2f} MB")
            shm.close()
        logger.debug("Stored EC shared memory cache for hash %s", mm_hash)

    def has_caches(self, request: "Request") -> list[bool]:
        result = []
        for feature in request.mm_features:
            try:
                mm_hash = feature.identifier
                self._mm_hashes_need_loads.add(mm_hash)
                logger.debug(f"ecconnector has caches:{mm_hash}")
                shared_memory.SharedMemory(name=_mm_hash_to_sha256(mm_hash)).close()
                result.append(True)
            except FileNotFoundError:
                result.append(False)
        return result

    def update_state_after_alloc(self, request: "Request", index: int) -> None:
        mm_hash = request.mm_features[index].identifier
        self._mm_hashes_need_loads.add(mm_hash)

    def build_connector_meta(
            self, scheduler_output: SchedulerOutput
    ) -> ECConnectorMetadata:
        meta = ECSharedMemoryConnectorMetadata()
        for mm_hash in self._mm_hashes_need_loads:
            meta.add_mm_hash(mm_hash)
        self._mm_hashes_need_loads.clear()
        return meta
