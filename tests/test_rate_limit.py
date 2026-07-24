from rate_limit import InMemoryRateLimiter


def test_limiter_blocks_after_limit() -> None:
    limiter = InMemoryRateLimiter(limit=2, window_seconds=60, enabled=True)

    first = limiter.check("client-a", now=0)
    second = limiter.check("client-a", now=1)
    third = limiter.check("client-a", now=2)

    assert first.allowed is True
    assert second.allowed is True
    assert third.allowed is False
    assert third.remaining == 0
    assert third.reset_after_seconds >= 1


def test_limiter_window_expiration_allows_again() -> None:
    limiter = InMemoryRateLimiter(limit=1, window_seconds=10, enabled=True)

    blocked = limiter.check("client-a", now=1)
    assert blocked.allowed is True

    denied = limiter.check("client-a", now=2)
    assert denied.allowed is False

    allowed_again = limiter.check("client-a", now=12)
    assert allowed_again.allowed is True
