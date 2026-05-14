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

    id: Optional[str] = None
    city: Optional[str] = None

    name: Optional[str] = None
    url: Optional[str] = None
    
    price: Optional[float] = None
    currency: Optional[str] = None

    rating: Optional[float] = None  
    stars: Optional[int] = None     

    # apartment/hotel/hostel
    property_type: Optional[str] = None

    description: Optional[str] = None
    facilities: List[Any] = Field(default_factory=list)
    rooms: List[Room] = Field(default_factory=list)
    raw: Optional[Dict[str, Any]] = None
