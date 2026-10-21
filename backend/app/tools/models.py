"""Typed inputs and outputs of the weather tool."""

from datetime import date
from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator


class TripWeatherRequest(BaseModel):
    """Input of the get_trip_weather tool."""

    destination: str = Field(
        min_length=2,
        max_length=200,
        description="City or place to forecast, as specific as possible, e.g. 'Lisbon, Portugal'.",
    )
    start_date: date = Field(description="First day of the trip (YYYY-MM-DD).")
    end_date: date = Field(description="Last day of the trip (YYYY-MM-DD).")

    @model_validator(mode="after")
    def _check_order(self) -> Self:
        if self.end_date < self.start_date:
            raise ValueError("end_date must not be before start_date")
        return self


class Place(BaseModel):
    name: str
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    country_code: str | None = None


class DailyWeather(BaseModel):
    date: date
    summary: str
    temperature_max_c: float | None
    temperature_min_c: float | None
    precipitation_mm: float | None
    precipitation_probability_pct: int | None


class TripWeather(BaseModel):
    """A successful forecast for (part of) the trip."""

    status: Literal["ok"] = "ok"
    source: Literal["Open-Meteo forecast"] = "Open-Meteo forecast"
    place: Place
    days: list[DailyWeather]
    note: str | None = None


class ToolError(BaseModel):
    """A structured failure the agent must relay instead of guessing."""

    status: Literal["error"] = "error"
    kind: Literal["invalid_input", "not_found", "out_of_range", "upstream_unavailable"]
    message: str


TripWeatherResult = TripWeather | ToolError
