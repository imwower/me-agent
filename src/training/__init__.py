"""训练循环与配置解析模块。

为了避免在缺少 transformers 等依赖时导入失败，这里对 trainer
的导入做了轻量的保护：若导入失败，则导出一个占位函数，并在
调用时给出友好的错误提示。
"""

try:  # pragma: no cover - 降级路径仅在缺少训练依赖时触发
    from .trainer import main as train_main  # type: ignore  # noqa: F401
except Exception as _exc:  # noqa: BLE001
    def train_main() -> None:  # type: ignore[no-redef]
        raise RuntimeError(
            "导入训练模块失败，可能缺少 transformers 等依赖："
            f"{_exc}. 请先运行 env/install.sh 安装完整依赖。"
        )

__all__ = ["train_main"]
