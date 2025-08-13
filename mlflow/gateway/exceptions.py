class AIGatewayConfigException(Exception):
    pass


class AIGatewayException(Exception):
    """
    A custom exception class for handling exceptions raised by the AI Gateway.
    This will be transformed into an HTTPException before being returned to the client.
    """

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)
