"""Core 层回归测试：异常体系、日志入口、模块导入。"""

import importlib

import pytest

from chatchat.core import (
    ChatchatError,
    ConfigurationError,
    ServiceError,
    ResourceNotFoundError,
    ValidationError,
)
from chatchat.core.logging import get_logger


class TestCoreExceptions:
    def test_errors_subclass_chatchat_error(self):
        for cls in (
            ConfigurationError,
            ServiceError,
            ResourceNotFoundError,
            ValidationError,
        ):
            assert issubclass(cls, ChatchatError)

    def test_instantiate_and_catch(self):
        with pytest.raises(ChatchatError):
            raise ChatchatError("base")
        with pytest.raises(ConfigurationError):
            raise ConfigurationError("bad config")
        with pytest.raises(ServiceError):
            raise ServiceError("service failed")
        with pytest.raises(ResourceNotFoundError):
            raise ResourceNotFoundError("not found")
        with pytest.raises(ValidationError):
            raise ValidationError("invalid")

    def test_exception_chain_preserved(self):
        try:
            try:
                raise ValueError("original")
            except ValueError as exc:
                raise ServiceError("wrapped") from exc
        except ServiceError as exc:
            assert exc.__cause__ is not None
            assert isinstance(exc.__cause__, ValueError)


class TestCoreLogging:
    def test_get_logger_basic_levels(self):
        logger = get_logger(__name__)
        logger.info("m01 info test")
        logger.warning("m01 warning test")
        logger.error("m01 error test")


def test_existing_modules_import():
    modules = [
        "chatchat",
        "chatchat.settings",
        "chatchat.core",
        "chatchat.server",
        "chatchat.server.chat",
        "chatchat.server.knowledge_base",
        "chatchat.server.agent",
        "chatchat.server.reranker",
    ]
    for m in modules:
        importlib.import_module(m)