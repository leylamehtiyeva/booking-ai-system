from __future__ import annotations

from datetime import date

# "Today" for every evaluation run (see app/config/clock.py). The golden
# datasets ask for stays in April-May 2026 with explicit years; against the
# real calendar those would become "dates in the past" clarifications, and
# year inference for year-less queries would drift over time.
EVAL_REFERENCE_DATE = date(2026, 3, 1)
