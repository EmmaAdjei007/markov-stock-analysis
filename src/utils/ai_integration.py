"""
AI Integration module supporting multiple providers (Groq, Gemini, OpenAI).

Groq is recommended for free tier with excellent performance.
"""

import os
import logging
from typing import Optional, List, Dict, Any, Literal
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

# Provider-specific imports with graceful fallbacks
try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False
    logger.warning("Groq library not installed. Install with: pip install groq")

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    logger.warning("Gemini library not installed. Install with: pip install google-generativeai")

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    logger.warning("OpenAI library not installed. Install with: pip install openai")


@dataclass
class AIResponse:
    """Container for AI model responses."""
    content: str
    provider: str
    model: str
    tokens_used: Optional[int] = None
    finish_reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class AIIntegration:
    """
    Unified AI integration supporting multiple providers.

    Providers:
        - Groq (Recommended): Fast, free tier available, Mixtral/Llama models
        - Gemini: Google's models, free tier available
        - OpenAI: GPT models, paid only

    Example:
        >>> ai = AIIntegration(provider="groq")
        >>> response = ai.generate("Analyze AAPL stock trend", context={"price": 150})
        >>> print(response.content)
    """

    def __init__(
        self,
        provider: Literal["groq", "gemini", "openai"] = "groq",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ):
        """
        Initialize AI integration.

        Args:
            provider: AI provider to use
            api_key: API key (if None, reads from environment)
            model: Specific model name (uses provider default if None)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
        """
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Get API key
        if api_key is None:
            api_key = self._get_api_key(provider)

        if not api_key:
            raise ValueError(f"No API key found for {provider}. Set {provider.upper()}_API_KEY environment variable.")

        # Set default models
        default_models = {
            "groq": "mixtral-8x7b-32768",  # Fast and powerful
            "gemini": "gemini-1.5-flash",
            "openai": "gpt-3.5-turbo"
        }

        self.model = model or default_models.get(provider)

        # Initialize client
        self.client = self._initialize_client(provider, api_key)

    def _get_api_key(self, provider: str) -> Optional[str]:
        """Get API key from environment or Streamlit secrets."""
        env_var = f"{provider.upper()}_API_KEY"

        # Check environment variable
        api_key = os.getenv(env_var)
        if api_key:
            return api_key

        # Check Streamlit secrets
        try:
            import streamlit as st
            if hasattr(st, 'secrets') and env_var in st.secrets:
                return st.secrets[env_var]
        except (ImportError, FileNotFoundError):
            pass

        return None

    def _initialize_client(self, provider: str, api_key: str):
        """Initialize provider-specific client."""
        if provider == "groq":
            if not HAS_GROQ:
                raise ImportError("Groq library not installed. Install with: pip install groq")
            return Groq(api_key=api_key)

        elif provider == "gemini":
            if not HAS_GEMINI:
                raise ImportError("Gemini library not installed. Install with: pip install google-generativeai")
            genai.configure(api_key=api_key)
            return genai.GenerativeModel(self.model)

        elif provider == "openai":
            if not HAS_OPENAI:
                raise ImportError("OpenAI library not installed. Install with: pip install openai")
            openai.api_key = api_key
            return openai

        else:
            raise ValueError(f"Unknown provider: {provider}")

    def generate(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> AIResponse:
        """
        Generate AI response.

        Args:
            prompt: User prompt/question
            context: Additional context dictionary
            system_prompt: System prompt (role definition)
            conversation_history: Previous messages for context

        Returns:
            AIResponse with generated content
        """
        # Build enhanced prompt with context
        if context:
            context_str = "\n".join([f"- {k}: {v}" for k, v in context.items()])
            enhanced_prompt = f"{prompt}\n\nContext:\n{context_str}"
        else:
            enhanced_prompt = prompt

        # Provider-specific generation
        if self.provider == "groq":
            return self._generate_groq(enhanced_prompt, system_prompt, conversation_history)
        elif self.provider == "gemini":
            return self._generate_gemini(enhanced_prompt, system_prompt, conversation_history)
        elif self.provider == "openai":
            return self._generate_openai(enhanced_prompt, system_prompt, conversation_history)

    def _generate_groq(
        self,
        prompt: str,
        system_prompt: Optional[str],
        history: Optional[List[Dict[str, str]]]
    ) -> AIResponse:
        """Generate response using Groq."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            content = response.choices[0].message.content
            tokens = response.usage.total_tokens if hasattr(response, 'usage') else None

            return AIResponse(
                content=content,
                provider="groq",
                model=self.model,
                tokens_used=tokens,
                finish_reason=response.choices[0].finish_reason,
                metadata={"response_id": response.id}
            )

        except Exception as e:
            logger.error(f"Groq API error: {e}")
            raise

    def _generate_gemini(
        self,
        prompt: str,
        system_prompt: Optional[str],
        history: Optional[List[Dict[str, str]]]
    ) -> AIResponse:
        """Generate response using Gemini."""
        try:
            # Gemini handles system prompt differently
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            response = self.client.generate_content(
                full_prompt,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens
                }
            )

            return AIResponse(
                content=response.text,
                provider="gemini",
                model=self.model,
                tokens_used=None,  # Gemini doesn't provide token count in free tier
                finish_reason=None
            )

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

    def _generate_openai(
        self,
        prompt: str,
        system_prompt: Optional[str],
        history: Optional[List[Dict[str, str]]]
    ) -> AIResponse:
        """Generate response using OpenAI."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": prompt})

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            return AIResponse(
                content=response.choices[0].message.content,
                provider="openai",
                model=self.model,
                tokens_used=response.usage.total_tokens,
                finish_reason=response.choices[0].finish_reason,
                metadata={"response_id": response.id}
            )

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise


class StockAnalysisAgent:
    """
    Specialized agent for stock market analysis using AI.

    Provides financial analysis with proper context and role definition.
    """

    SYSTEM_PROMPT = """You are a Senior Financial Analyst with expertise in:
- Technical analysis and chart patterns
- Risk management and portfolio optimization
- Markov chain modeling for market predictions
- Statistical analysis and backtesting

Guidelines:
- Provide clear, data-driven insights
- Highlight key risks and opportunities
- Use proper financial terminology
- Include confidence levels when making predictions
- Always mention relevant disclaimers about market risks
- Format responses with markdown for readability

Remember: You're providing educational analysis, not financial advice."""

    def __init__(
        self,
        provider: Literal["groq", "gemini", "openai"] = "groq",
        api_key: Optional[str] = None
    ):
        """Initialize stock analysis agent."""
        self.ai = AIIntegration(provider=provider, api_key=api_key)
        self.conversation_history: List[Dict[str, str]] = []

    def analyze(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        include_history: bool = True
    ) -> str:
        """
        Analyze stock market query with context.

        Args:
            query: User question or analysis request
            context: Market data, metrics, portfolio state, etc.
            include_history: Use conversation history for context

        Returns:
            Analysis text

        Example:
            >>> agent = StockAnalysisAgent(provider="groq")
            >>> context = {
            ...     "ticker": "AAPL",
            ...     "current_price": 150.25,
            ...     "sharpe_ratio": 1.8,
            ...     "max_drawdown": -12.5
            ... }
            >>> analysis = agent.analyze("Should I buy this stock?", context)
        """
        history = self.conversation_history if include_history else None

        response = self.ai.generate(
            prompt=query,
            context=context,
            system_prompt=self.SYSTEM_PROMPT,
            conversation_history=history
        )

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": query})
        self.conversation_history.append({"role": "assistant", "content": response.content})

        # Limit history size
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

        return response.content

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []

    def save_history(self, filepath: str):
        """Save conversation history to file."""
        with open(filepath, 'w') as f:
            json.dump(self.conversation_history, f, indent=2)

    def load_history(self, filepath: str):
        """Load conversation history from file."""
        with open(filepath, 'r') as f:
            self.conversation_history = json.load(f)


# Convenience functions for backward compatibility

def get_groq_response(prompt: str, context: Optional[Dict] = None) -> str:
    """Quick function to get Groq response."""
    ai = AIIntegration(provider="groq")
    response = ai.generate(prompt, context)
    return response.content


def stock_analysis_agent(
    query: str,
    context: Optional[Dict[str, Any]] = None,
    provider: str = "groq"
) -> str:
    """Quick function for stock analysis."""
    agent = StockAnalysisAgent(provider=provider)
    return agent.analyze(query, context, include_history=False)


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("AI Integration Demo")
    print("=" * 60)

    # Test if API keys are available
    from src.config import get_config
    config = get_config()

    print("\nAvailable Providers:")
    for provider in ["groq", "gemini", "openai"]:
        has_key = config.has_api_key(provider)
        status = "✓" if has_key else "✗"
        print(f"  {provider.capitalize()}: {status}")

    # Try Groq if available
    if config.has_api_key("groq"):
        print("\n" + "=" * 60)
        print("Testing Groq Integration")
        print("=" * 60)

        agent = StockAnalysisAgent(provider="groq")

        context = {
            "ticker": "AAPL",
            "current_price": 150.25,
            "sharpe_ratio": 1.82,
            "max_drawdown": -12.5,
            "win_rate": 0.58
        }

        response = agent.analyze(
            "Provide a brief risk assessment for this stock based on the metrics.",
            context=context
        )

        print("\nAI Response:")
        print(response)

    print("\n" + "=" * 60)
