from __future__ import annotations


class RetrievalError(RuntimeError):
    """
    Listing retrieval failed for an operational reason (provider HTTP
    error, network, timeout, missing credentials, unusable response).
    `code` uses the same vocabulary as app.llm.call_policy.LlmCallError.
    The message never contains the request URL (it carries the API token)
    or the provider's response body.
    """

    def __init__(self, code: str) -> None:
        super().__init__(f"Listing retrieval failed: {code}")
        self.code = code
