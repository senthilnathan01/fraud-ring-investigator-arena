"""Root app entrypoint shim for deployment and validator compatibility."""

from server.app import app as _app
from server.app import main as _main

app = _app


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    _main(host=host, port=port)


if __name__ == "__main__":
    main()
