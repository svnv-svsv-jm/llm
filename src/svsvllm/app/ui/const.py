class PageNames:
    """UI pages' names."""

    SETTINGS = "settings"
    MAIN = "main"

    @classmethod
    def all(cls) -> list[str]:
        """Returns all values."""
        return [cls.SETTINGS, cls.MAIN]
