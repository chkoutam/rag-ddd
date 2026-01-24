from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI


_opik_configured = False


def configure_opik(
    *,
    api_key: str | None = None,
    workspace: str | None = None,
    project_name: str = "rag-ddd",
    use_local: bool = False,
) -> None:
    """Configure Opik for LLM observability.

    Args:
        api_key: Opik API key (for Comet cloud). Not needed for local.
        workspace: Comet workspace name.
        project_name: Project name for grouping traces.
        use_local: Use local self-hosted Opik instance.
    """
    global _opik_configured
    if _opik_configured:
        return

    import opik

    if use_local:
        opik.configure(use_local=True)
    elif api_key:
        opik.configure(api_key=api_key, workspace=workspace)

    import os
    os.environ["OPIK_PROJECT_NAME"] = project_name

    _opik_configured = True


def get_tracked_openai_client(client: "OpenAI", project_name: str | None = None) -> "OpenAI":
    """Wrap an OpenAI client with Opik tracing.

    Args:
        client: The OpenAI client to wrap.
        project_name: Optional project name override.

    Returns:
        The wrapped OpenAI client with tracing enabled.
    """
    from opik.integrations.openai import track_openai

    return track_openai(client, project_name=project_name)


def is_opik_configured() -> bool:
    """Check if Opik has been configured."""
    return _opik_configured
