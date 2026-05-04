import asyncio

from app.logic.intent_router import build_search_request_adk_async


async def run_single_case(user_message: str):
    request = await build_search_request_adk_async(user_message)

    # 👉 превращаем в простой JSON
    result = {
        "city": request.city,
        "check_in": request.check_in.isoformat() if request.check_in else None,
        "check_out": request.check_out.isoformat() if request.check_out else None,
        "adults": request.adults,
        "children": request.children,
        "rooms": request.rooms,
        "property_types": [p.value for p in request.property_types] if request.property_types else None,
    "constraints": [
    {
        "normalized_text": c.normalized_text,
        "priority": c.priority.value,
        "category": c.category.value,
        "mapping_status": c.mapping_status.value,
        "mapped_fields": [f.value for f in c.mapped_fields],
        "evidence_strategy": c.evidence_strategy.value,
    }

            for c in request.constraints
        ]
    }

    return result


    
    
async def main():
    raw_cases = [
    # 🔹 БАЗОВЫЕ (known must)
    "I want an apartment in Baku from April 20 to April 26 with a kitchen",
]

    for i, user_message in enumerate(raw_cases, 1):
        output = await run_single_case(user_message)

        print(f"\n=== CASE {i} ===")
        print("USER:", user_message)
        print("SYSTEM OUTPUT:", output)


if __name__ == "__main__":
    asyncio.run(main())