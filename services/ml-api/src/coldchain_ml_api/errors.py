class InputError(ValueError):
    """The request is well-formed JSON but cannot be scored (HTTP 422)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
