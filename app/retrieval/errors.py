from __future__ import annotations


class RetrievalError(RuntimeError):
    """
    Listing retrieval failed for an operational reason (provider HTTP
    error, network, timeout, missing credentials, unusable response).
    `code` uses the same vocabulary as app.llm.call_policy.LlmCallError,
    plus "unsupported_destination" / "unsupported_request" for searches a
    retriever cannot express. The message never contains the request URL
    (it carries the API token) or the provider's response body.

    `user_message` is an optional fixed, user-safe text for a failure the
    user can act on (e.g. a destination the retriever does not cover); the
    conversation flow shows it verbatim instead of its generic text.
    """

    def __init__(self, code: str, *, user_message: str | None = None) -> None:
        super().__init__(f"Listing retrieval failed: {code}")
        self.code = code
        self.user_message = user_message
