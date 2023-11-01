#!/usr/bin/env python3

import logging as log
import sys
from pathlib import Path
from urllib.error import URLError

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
log.basicConfig(encoding="utf-8", format=LOG_FORMAT, level=log.INFO)

from image2map import Dashboard


def main():
    app = Dashboard()

    try:
        app()

    except URLError as e:
        app.error(
            """
            **This application requires internet access.**
            Connection error: %s
            """
            % e.reason
        )


if __name__ == "__main__":
    main()
