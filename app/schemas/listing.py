# app/schemas/listing.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class RoomOption(BaseModel):
    """
    Room option = конкретная опция внутри комнаты (например, тариф/план):
    refundable, breakfast, pay-now/pay-at-property и т.д.
    В MVP мы храним гибко и не пытаемся нормализовать всё сразу.
    """
    model_config = ConfigDict(extra="allow")

    name: Optional[str] = None
    price: Optional[float] = None
    currency: Optional[str] = None

    # Все прочие поля (refundability, cancellation policy, etc.) попадут в extra


class Room(BaseModel):
    """
    Room = единица размещения (комната/апартаменты как room entity у Booking).
    Главное для structured matching: facilities.
    """
    model_config = ConfigDict(extra="allow")

    name: Optional[str] = None
    facilities: List[Any] = Field(default_factory=list)
    options: List[RoomOption] = Field(default_factory=list)


class ListingRaw(BaseModel):
    """
    ListingRaw = минимальный контракт под то, что реально приходит из Apify/Booking actor.

    Важно:
    - делаем extra="allow", чтобы не падать, если actor добавит новые поля,
      и чтобы мы могли сохранять "сырьё" для отладки / расширения.
    """
    model_config = ConfigDict(extra="allow")

    # Стабильный идентификатор: желательно иметь.
    # Если actor не даёт id — на PR#3 можно собрать hash(url) и писать сюда.
    id: Optional[str] = None
    city: Optional[str] = None

    name: Optional[str] = None
    url: Optional[str] = None
    

    # Цена/валюта
    price: Optional[float] = None
    currency: Optional[str] = None

    # Качество
    rating: Optional[float] = None  # guest rating (например 8.7)
    stars: Optional[int] = None     # star rating (например 4)

    # Тип жилья (apartment/hotel/hostel...)
    property_type: Optional[str] = None

    # Текст
    description: Optional[str] = None

    # Facilities на уровне объекта (иногда есть)
    facilities: List[Any] = Field(default_factory=list)

    # Rooms
    rooms: List[Room] = Field(default_factory=list)

    # На будущее: можно хранить “сырой” блок
    raw: Optional[Dict[str, Any]] = None
