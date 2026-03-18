from app.utils.config import DEBUG


def log(message: str) -> None:
    if DEBUG:
        print(message)
