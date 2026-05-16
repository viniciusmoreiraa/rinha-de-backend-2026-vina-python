"""Converte payload JSON de transação em vetor int16[14] quantizado."""

import numpy as np
from array import array

# MCC risk pre-quantized to int16 (value * 10000)
MCC_RISK_Q = {
    "5411": 1500, "5812": 3000, "5912": 2000, "5944": 4500,
    "7801": 8000, "7802": 7500, "7995": 8500, "4511": 3500,
    "5311": 2500, "5999": 5000,
}
MCC_RISK_Q_DEFAULT = 5000

# 2026 is not a leap year. Pre-compute base days from 2000-01-01 to 2026-01-01.
# This avoids any loop at runtime for the common case.
_DAYS_2026 = 9497  # days from 2000-01-01 to 2026-01-01
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

# Pre-computed hour quantization: hour * 10000 / 23, rounded
_HOUR_Q = tuple(int(h * 10000.0 / 23.0 + 0.5) for h in range(24))

# Pre-computed weekday quantization: dow * 10000 / 6, rounded
_WEEKDAY_Q = tuple(int(d * 10000.0 / 6.0 + 0.5) for d in range(7))

# Reusable output buffer (single-threaded uvicorn)
_OUT_BUF = np.empty(14, dtype=np.int16)


def _q(x: float) -> int:
    """Clamp to [0,1] and quantize to int16 (scale 10000)."""
    if x <= 0.0:
        return 0
    if x >= 1.0:
        return 10000
    return int(x * 10000.0 + 0.5)


def _ts_minutes(ts: str) -> float:
    """Parse ISO 8601 timestamp to minutes. Optimized for 2026."""
    # Fast path: all competition data is 2026
    y = int(ts[0:4])
    m = int(ts[5:7])
    d = int(ts[8:10])

    if y == 2026:
        days = _DAYS_2026 + _CUMDAYS_2026[m] + d - 1
    else:
        is_leap = (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0))
        days = _DAYS_FROM_2000.get(y, 0) + (_CUMDAYS_LEAP[m] if is_leap else _CUMDAYS[m]) + d - 1

    return days * 1440 + int(ts[11:13]) * 60 + int(ts[14:16]) + int(ts[17:19]) / 60.0


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

    buf[0] = _q(amount / 10000.0)
    buf[1] = _q(tx["installments"] / 12.0)
    buf[2] = _q(amount / cust["avg_amount"] / 10.0)
    buf[3] = _HOUR_Q[int(requested_at[11:13])]

    # Weekday (Sakamoto)
    y = int(requested_at[0:4])
    m = int(requested_at[5:7])
    d = int(requested_at[8:10])
    if m < 3:
        y -= 1
    dow = (y + y // 4 - y // 100 + y // 400 + _SAKAMOTO_T[m - 1] + d) % 7
    buf[4] = _WEEKDAY_Q[(dow + 6) % 7]

    if last is None:
        buf[5] = -10000
        buf[6] = -10000
    else:
        minutes_diff = _ts_minutes(requested_at) - _ts_minutes(last["timestamp"])
        buf[5] = _q(minutes_diff / 1440.0)
        buf[6] = _q(last["km_from_current"] / 1000.0)

    buf[7] = _q(term["km_from_home"] / 1000.0)
    buf[8] = _q(cust["tx_count_24h"] / 20.0)
    buf[9] = 10000 if term["is_online"] else 0
    buf[10] = 10000 if term["card_present"] else 0
    buf[11] = 0 if merch["id"] in cust["known_merchants"] else 10000
    buf[12] = MCC_RISK_Q.get(merch["mcc"], MCC_RISK_Q_DEFAULT)
    buf[13] = _q(merch["avg_amount"] / 10000.0)

    return buf
