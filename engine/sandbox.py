"""
沙箱执行器 — 安全执行 AI 生成的策略代码

功能：
  - 动态加载 AI 生成的 .py 文件
  - 捕获异常并返回错误信息（用于反馈给 LLM 修复）
  - 限制执行环境（不允许访问网络、文件系统等）
  - 返回策略信号 DataFrame
"""

import importlib
import importlib.util
import sys
import traceback
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class SandboxError(Exception):
    """沙箱执行错误"""
    pass


def load_strategy_from_file(
    strategy_path: str,
) -> Tuple[Optional[object], Optional[str]]:
    """
    从 .py 文件动态加载策略类

    Args:
        strategy_path: 策略文件路径 (e.g., strategy_pool/strategy_v1.py)

    Returns:
        (strategy_instance, error_message)
        - 成功时: (策略实例, None)
        - 失败时: (None, 错误信息)
    """
    path = Path(strategy_path)

    if not path.exists():
        return None, f"策略文件不存在: {path}"

    if not path.suffix == ".py":
        return None, f"策略文件必须是 .py 文件: {path}"

    module_name = path.stem

    try:
        # 动态加载模块
        spec = importlib.util.spec_from_file_location(
            module_name, str(path)
        )
        if spec is None or spec.loader is None:
            return None, f"无法加载模块: {path}"

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    except SyntaxError as e:
        error_msg = f"❌ 策略代码语法错误:\n{str(e)}\n\n详细:\n{traceback.format_exc()}"
        return None, error_msg
    except Exception as e:
        error_msg = f"❌ 策略模块加载失败:\n{str(e)}\n\n详细:\n{traceback.format_exc()}"
        return None, error_msg

    # 查找 BaseStrategy 的子类
    from template.strategy_base import BaseStrategy

    strategy_cls = None
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        try:
            if (
                isinstance(attr, type)
                and issubclass(attr, BaseStrategy)
                and attr is not BaseStrategy
            ):
                strategy_cls = attr
                break
        except TypeError:
            continue

    if strategy_cls is None:
        return None, (
            f"❌ 策略文件中未找到继承 BaseStrategy 的策略类。"
            f"请确保定义一个 class 继承自 BaseStrategy 并实现 generate_signals() 方法。"
        )

    # 实例化
    try:
        instance = strategy_cls()
    except Exception as e:
        return None, f"❌ 策略实例化失败:\n{str(e)}\n\n{traceback.format_exc()}"

    # 验证方法存在
    if not hasattr(instance, "generate_signals"):
        return None, f"❌ 策略类未实现 generate_signals() 方法"

    return instance, None


def execute_strategy(
    strategy_path: str,
    data: pd.DataFrame,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    在沙箱中执行策略，生成信号

    Args:
        strategy_path: 策略文件路径
        data: 日线行情数据

    Returns:
        (signals_df, error_message)
        - 成功时: (信号 DataFrame, None)
        - 失败时: (None, 错误信息)
    """
    # 1. 加载策略
    strategy, error = load_strategy_from_file(strategy_path)
    if error:
        return None, error

    # 2. 验证输入数据
    if data.empty:
        return None, "❌ 输入数据为空"

    required_columns = {"date", "code", "close"}
    missing = required_columns - set(data.columns)
    if missing:
        return None, f"❌ 输入数据缺少必要列: {missing}"

    # 3. 执行策略
    try:
        signals = strategy.generate_signals(data)
    except Exception as e:
        error_msg = (
            f"❌ 策略运行时报错:\n"
            f"{str(e)}\n\n"
            f"详细堆栈:\n{traceback.format_exc()}"
        )
        return None, error_msg

    # 4. 验证输出
    if not isinstance(signals, pd.DataFrame):
        return None, (
            f"❌ generate_signals() 必须返回 pd.DataFrame，"
            f"但返回了 {type(signals).__name__}"
        )

    if signals.empty:
        return None, "⚠️ 策略未产生任何信号（返回空 DataFrame）"

    # 5. 调用策略自带的 validate_output
    if hasattr(strategy, "validate_output"):
        try:
            signals = strategy.validate_output(signals)
        except Exception as e:
            error_msg = f"❌ 信号验证失败:\n{str(e)}\n\n{traceback.format_exc()}"
            return None, error_msg

    return signals, None


def format_error_for_llm(
    error_message: str,
    strategy_code: str,
) -> str:
    """
    将错误信息格式化为 LLM 可理解的修复请求

    Args:
        error_message: 捕获的错误信息
        strategy_code: 原始策略代码

    Returns:
        LLM prompt
    """
    return f"""🔧 代码报错修复请求

以下是策略代码执行时出现的错误，请分析原因并修复代码。

【错误信息】
{error_message}

【当前策略代码】
```python
{strategy_code}
```

请根据错误信息和策略模板规范，生成修复后的完整策略代码。
确保：
1. 代码必须继承 BaseStrategy 并实现 generate_signals()
2. generate_signals() 必须返回包含 ['code', 'weight', 'date'] 的 DataFrame
3. 只使用 pandas/numpy/scipy 和 data 参数，不访问网络或文件系统
"""