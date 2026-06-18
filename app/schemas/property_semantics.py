from __future__ import annotations

from enum import Enum


class PropertyType(str, Enum):
    RYOKAN = "ryokan"
    HOTEL = "hotel"
    APARTMENT = "apartment"
    RESORT = "resort"
    VILLA = "villa"
    BED_AND_BREAKFAST = "bed_and_breakfast"
    HOLIDAY_HOME = "holiday_home"
    GUEST_HOUSE = "guest_house"
    HOSTEL = "hostel"
    CAPSULE_HOTEL = "capsule_hotel"
    HOMESTAY = "homestay"
    CHALET = "chalet"
    LODGE = "lodge"
    CAMPSITE = "campsite"
    COUNTRY_HOUSE = "country_house"
    LOVE_HOTEL = "love_hotel"
    HOUSE = "house"
    APARTHOTEL = "aparthotel"
    GUESTHOUSE = "guesthouse"


class OccupancyType(str, Enum):
    ENTIRE_PLACE = "entire_place"
    PRIVATE_ROOM = "private_room"
    SHARED_ROOM = "shared_room"
    HOTEL_ROOM = "hotel_room"