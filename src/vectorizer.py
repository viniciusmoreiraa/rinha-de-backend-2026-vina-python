"""Converte payload JSON de transação em vetor int16[14] quantizado."""

import numpy as np

# MCC risk pre-quantized to int16 (value * 10000)
MCC_RISK_Q = {
    "5411": 1500, "5812": 3000, "5912": 2000, "5944": 4500,
    "7801": 8000, "7802": 7500, "7995": 8500, "4511": 3500,
    "5311": 2500, "5999": 5000,
}
MCC_RISK_Q_DEFAULT = 5000

# 2026 is not a leap year. Pre-compute base days from 2000-01-01 to 2026-01-01.
_DAYS_2026 = 9497
_CUMDAYS_2026 = (0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)

# Fallback for other years
_DAYS_FROM_2000 = {}
for _y in range(2020, 2030):
    _d = 0
    for _yr in range(2000, _y):
        _d += 366 if (_yr % 4 == 0 and (_yr % 100 != 0 or _yr % 400 == 0)) else 365
    _DAYS_FROM_2000[_y] = _d

_CUMDAYS = (0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)
_CUMDAYS_LEAP = (0, 0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335)

# Sakamoto's weekday table
_SAKAMOTO_T = (0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4)

# Pre-computed Sakamoto year term for 2026 (m >= 3): y + y//4 - y//100 + y//400
_Y_TERM_2026 = 2026 + 2026 // 4 - 2026 // 100 + 2026 // 400  # 2517
# For m < 3, y-1=2025: 2025 + 2025//4 - 2025//100 + 2025//400
_Y_TERM_2025 = 2025 + 2025 // 4 - 2025 // 100 + 2025 // 400  # 2516

# Pre-computed hour quantization: hour * 10000 / 23, rounded
_HOUR_Q = tuple(int(h * 10000.0 / 23.0 + 0.5) for h in range(24))

# Pre-computed weekday quantization: dow * 10000 / 6, rounded
_WEEKDAY_Q = tuple(int(d * 10000.0 / 6.0 + 0.5) for d in range(7))

# Reusable output buffer (single-threaded uvicorn)
_OUT_BUF = np.empty(14, dtype=np.int16)


def _ts_minutes_from_parts(y: int, m: int, d: int, h: int, mi: int, s: int) -> float:
    """Convert pre-parsed timestamp parts to minutes since epoch."""
    if y == 2026:
        days = _DAYS_2026 + _CUMDAYS_2026[m] + d - 1
    else:
        is_leap = (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0))
        days = _DAYS_FROM_2000.get(y, 0) + (_CUMDAYS_LEAP[m] if is_leap else _CUMDAYS[m]) + d - 1

    return days * 1440 + h * 60 + mi + s / 60.0


def vectorize(data: dict) -> np.ndarray:
    """Convert transaction payload to quantized int16[14] vector."""
    tx = data["transaction"]
    cust = data["customer"]
    merch = data["merchant"]
    term = data["terminal"]
    last = data.get("last_transaction")

    amount = tx["amount"]
    requested_at = tx["requested_at"]

    buf = _OUT_BUF

    # Parse timestamp once, reuse for hour, weekday, and minutes
    y = int(requested_at[0:4])
    m = int(requested_at[5:7])
    d = int(requested_at[8:10])
    h = int(requested_at[11:13])
    mi = int(requested_at[14:16])
    s = int(requested_at[17:19])

    # Inline _q: clamp to [0,1] then quantize to int16 (scale 10000)
    v = amount / 10000.0
    buf[0] = 0 if v <= 0.0 else (10000 if v >= 1.0 else int(v * 10000.0 + 0.5))

    v = tx["installments"] / 12.0
    buf[1] = 0 if v <= 0.0 else (10000 if v >= 1.0 else int(v * 10000.0 + 0.5))

    v = amount / cust["avg_amount"] / 10.0
    buf[2] = 0 if v <= 0.0 else (10000 if v >= 1.0 else int(v * 10000.0 + 0.5))

    buf[3] = _HOUR_Q[h]

    # Weekday (Sakamoto) — pre-computed year term for 2026
    if y == 2026:
        y_term = _Y_TERM_2025 if m < 3 else _Y_TERM_2026
    else:
        yw = y - 1 if m < 3 else y
        y_term = yw + yw // 4 - yw // 100 + yw // 400
    dow = (y_term + _SAKAMOTO_T[m - 1] + d) % 7
    buf[4] = _WEEKDAY_Q[(dow + 6) % 7]

    if last is None:
        buf[5] = -10000
        buf[6] = -10000
    else:
        current_min = _ts_minutes_from_parts(y, m, d, h, mi, s)
        lt = last["timestamp"]
        last_min = _ts_minutes_from_parts(
            int(lt[0:4]), int(lt[5:7]), int(lt[8:10]),
            int(lt[11:13]), int(lt[14:16]), int(lt[17:19])
        )
        v = (current_min - last_min) / 1440.0
        buf[5] = 0 if v <= 0.0 else (10000 if v >= 1.0 else int(v * 10000.0 + 0.5))

        v = last["km_from_current"] / 1000.0
        buf[6] = 0 if v <= 0.0 else (10000 if v >= 1.0 else int(v * 10000.0 + 0.5))

    v = term["km_from_home"] / 1000.0
    buf[7] = 0 if v <= 0.0 else (10000 if v >= 1.0 else int(v * 10000.0 + 0.5))

    v = cust["tx_count_24h"] / 20.0
    buf[8] = 0 if v <= 0.0 else (10000 if v >= 1.0 else int(v * 10000.0 + 0.5))

    buf[9] = 10000 if term["is_online"] else 0
    buf[10] = 10000 if term["card_present"] else 0
    buf[11] = 0 if merch["id"] in cust["known_merchants"] else 10000
    buf[12] = MCC_RISK_Q.get(merch["mcc"], MCC_RISK_Q_DEFAULT)

    v = merch["avg_amount"] / 10000.0
    buf[13] = 0 if v <= 0.0 else (10000 if v >= 1.0 else int(v * 10000.0 + 0.5))

    return buf
