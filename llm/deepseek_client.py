"""
DeepSeek API 客户端 — LLM 交互层

功能：
  - generate_strategy(): 生成初始策略
  - analyze_and_improve(): 根据回测结果改进策略
  - fix_code(): 根据报错修复代码

使用 OpenAI 兼容客户端连接 DeepSeek API。
"""

import re
import sys
from pathlib import Path
from typing import Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class DeepSeekClient:
    """
    DeepSeek API 客户端

    Usage:
        client = DeepSeekClient(api_key="sk-xxx")
        code = client.generate_strategy("生成一个动量策略")
        new_code = client.analyze_and_improve(metrics_context, current_code)
        fixed_code = client.fix_code(error_msg, buggy_code)
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
    ):
        if not HAS_OPENAI:
            raise ImportError(
                "openai 库未安装，请运行: pip install openai"
            )

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

        # 加载提示词模板
        prompts_dir = Path(__file__).resolve().parent / "prompts"
        self.prompts = {}
        for role in ["strategist", "debugger", "analyst"]:
            prompt_path = prompts_dir / f"{role}.txt"
            if prompt_path.exists():
                self.prompts[role] = prompt_path.read_text(encoding="utf-8")
            else:
                self.prompts[role] = f"You are a {role}."

    # ------------------------------------------------------------------
    # 公有方法
    # ------------------------------------------------------------------

    def generate_strategy(
        self,
        instruction: str = "生成一个 A 股选股策略",
        temperature: float = 0.7,
        max_retries: int = 3,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        生成新的策略代码

        Args:
            instruction: 策略生成指令
            temperature: 生成温度 (0~1, 越高越有创造性)
            max_retries: 最大重试次数

        Returns:
            (code, error) — 成功时返回 (代码, None)，失败时返回 (None, 错误)
        """
        system_prompt = self.prompts.get("strategist", "")

        user_message = f"""请根据以下要求生成选股策略代码：

{instruction}

请严格遵循策略编码规范，直接输出完整的 Python 代码。
"""

        code, error = self._call_with_retry(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=temperature,
            max_tokens=4096,
            max_retries=max_retries,
        )

        if code:
            code = self._extract_code(code)

        return code, error

    def analyze_and_improve(
        self,
        feedback_context: str,
        current_code: Optional[str] = None,
        temperature: float = 0.7,
        max_retries: int = 3,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        根据回测结果分析并改进策略

        Args:
            feedback_context: 回测结果反馈文本（由 analyzer.metrics_to_feedback_context 生成）
            current_code: 当前策略代码（可选）
            temperature: 生成温度
            max_retries: 最大重试次数

        Returns:
            (improved_code, error)
        """
        system_prompt = self.prompts.get("analyst", "")

        code_section = ""
        if current_code:
            code_section = f"""

【当前策略代码】
```python
{current_code}
```
"""

        user_message = f"""请根据以下回测结果，分析策略问题并给出改进后的完整策略代码。

{feedback_context}
{code_section}

请严格遵守策略编码规范，输出改进后的完整 Python 代码。
"""

        code, error = self._call_with_retry(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=temperature,
            max_tokens=4096,
            max_retries=max_retries,
        )

        if code:
            code = self._extract_code(code)

        return code, error

    def fix_code(
        self,
        error_message: str,
        strategy_code: str,
        temperature: float = 0.3,
        max_retries: int = 3,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        根据报错信息修复策略代码

        Args:
            error_message: 报错信息
            strategy_code: 原始策略代码
            temperature: 生成温度 (较低以保持稳定性)
            max_retries: 最大重试次数

        Returns:
            (fixed_code, error)
        """
        system_prompt = self.prompts.get("debugger", "")

        user_message = f"""以下策略代码执行时出现错误，请修复它。

【错误信息】
{error_message}

【原始代码】
```python
{strategy_code}
```

请输出修复后的完整 Python 代码。
"""

        code, error = self._call_with_retry(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=temperature,
            max_tokens=4096,
            max_retries=max_retries,
        )

        if code:
            code = self._extract_code(code)

        return code, error

    def chat(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        通用对话接口（单次调用）

        Args:
            system_prompt: 系统提示
            user_message: 用户消息
            temperature: 温度
            max_tokens: 最大输出 token

        Returns:
            (response, error)
        """
        return self._call_with_retry(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=1,
        )

    # ------------------------------------------------------------------
    # 私有方法
    # ------------------------------------------------------------------

    def _call_with_retry(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        max_retries: int = 3,
    ) -> Tuple[Optional[str], Optional[str]]:
        """带重试的 API 调用"""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False,
                )

                content = response.choices[0].message.content
                return content, None

            except Exception as e:
                error_msg = str(e)

                # 如果是 API Key 错误，不重试
                if "401" in error_msg or "Invalid API Key" in error_msg:
                    return None, f"🔑 API Key 无效: {error_msg}"

                # 如果是 quota 超限，不重试
                if "429" in error_msg or "insufficient_quota" in error_msg:
                    return None, f"💰 API 配额不足: {error_msg}"

                if attempt < max_retries - 1:
                    import time
                    wait = 2 ** attempt
                    print(f"   ⚠️ API 调用失败 (重试 {attempt+1}/{max_retries}): {error_msg[:100]}")
                    print(f"      等待 {wait} 秒后重试...")
                    time.sleep(wait)
                else:
                    return None, f"❌ API 调用失败 (已重试 {max_retries} 次): {error_msg}"

        return None, "❌ 未知错误: 超过最大重试次数"

    def _extract_code(self, text: str) -> str:
        """
        从 LLM 输出中提取 Python 代码

        Args:
            text: LLM 原始输出

        Returns:
            纯净的 Python 代码
        """
        # 尝试提取 ```python ... ``` 代码块
        pattern = r'```python\s*\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)

        if matches:
            return matches[-1].strip()

        # 尝试提取 ``` ... ``` (无语言标记)
        pattern = r'```\s*\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)

        if matches:
            # 检查是否是 Python 代码（包含 class/import/def）
            for match in reversed(matches):
                if any(
                    keyword in match
                    for keyword in [
                        "class ",
                        "def generate_signals",
                        "import pandas",
                        "BaseStrategy",
                    ]
                ):
                    return match.strip()

        # 如果找不到代码块，返回原始文本（去除非代码行）
        lines = text.strip().split("\n")
        code_lines = []
        in_code = False

        for line in lines:
            if line.startswith("```"):
                in_code = not in_code
                continue
            if in_code or line.strip().startswith(("import", "from", "class", "def", "    ", "\t")):
                code_lines.append(line)

        if code_lines:
            return "\n".join(code_lines)

        return text.strip()


# ============================================================
# 便捷工厂函数
# ============================================================

def create_client_from_config() -> DeepSeekClient:
    """
    从 config.py 或环境变量创建 DeepSeek 客户端

    Returns:
        DeepSeekClient 实例

    Raises:
        ValueError: 未找到 API Key
    """
    api_key = None
    base_url = "https://api.deepseek.com"
    model = "deepseek-chat"

    # 方式 1: 尝试从 config.py 加载
    try:
        import config
        api_key = config.DEEPSEEK_API_KEY
        if hasattr(config, "DEEPSEEK_BASE_URL"):
            base_url = config.DEEPSEEK_BASE_URL
        if hasattr(config, "DEEPSEEK_MODEL"):
            model = config.DEEPSEEK_MODEL
    except ImportError:
        pass

    # 方式 2: 尝试从 .env 加载
    if api_key is None or api_key == "sk-your-api-key-here":
        try:
            from dotenv import load_dotenv
            import os
            load_dotenv()
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if api_key:
                base_url = os.getenv("DEEPSEEK_BASE_URL", base_url)
        except ImportError:
            pass

    if api_key is None or api_key in ("sk-your-api-key-here", ""):
        raise ValueError(
            "❌ 未找到 DeepSeek API Key。\n"
            "请确保以下之一:\n"
            "  1. 创建 config.py (参考 config.example.py) 并填入真实 API Key\n"
            "  2. 创建 .env 文件，设置 DEEPSEEK_API_KEY=sk-xxx\n"
            "获取 Key: https://platform.deepseek.com"
        )

    return DeepSeekClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
    )