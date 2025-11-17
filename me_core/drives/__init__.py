"""内在驱动力（drives）相关模块。

当前提供：
- DriveVector：表示一组可调节的驱动力参数；
- apply_user_command / implicit_adjust：根据用户指令与反馈平滑调整驱动力；
- DrivesConfig：驱动力相关超参数配置；
- Drive / Intent / BaseDriveSystem / SimpleDriveSystem：高层驱动力与意图接口。
"""

from .base import BaseDriveSystem, Drive, Intent, SimpleDriveSystem  # noqa: F401
from .config import DEFAULT_DRIVES_CONFIG, DrivesConfig  # noqa: F401
from .drive_update import apply_user_command, implicit_adjust  # noqa: F401
from .drive_vector import DriveVector  # noqa: F401

__all__ = [
    "DriveVector",
    "apply_user_command",
    "implicit_adjust",
    "DrivesConfig",
    "DEFAULT_DRIVES_CONFIG",
    "Drive",
    "Intent",
    "BaseDriveSystem",
    "SimpleDriveSystem",
]

