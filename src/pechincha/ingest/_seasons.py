def soccerdata_season(start_year: int) -> str:
    """2025 -> '2526' (época 2025/26), formato aceite pelo soccerdata."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"
