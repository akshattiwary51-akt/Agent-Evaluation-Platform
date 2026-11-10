import httpx
import pytest

from app.accounting.cost import Price, estimate_cost, total_cost
from app.providers.anthropic import AnthropicProvider, parse_response, to_wire_messages
from app.providers.base import ProviderError, Usage

PRICE = Price("p", "m", input_per_million=3.0, output_per_million=15.0,
              cached_input_per_million=0.3)


def test_cost_counts_cached_separately_without_double_counting():
    # 1M uncached in, 1M cached in, 1M out
    assert estimate_cost(Usage(1_000_000, 1_000_000, 1_000_000), PRICE) == pytest.approx(18.3)


def test_missing_usage_is_unknown_not_free():
    assert estimate_cost(Usage(), PRICE) is None
    assert total_cost([Usage(10, 10, 0), Usage()], PRICE) is None


def test_cached_tokens_without_a_cached_rate_are_unknown():
    p = Price("p", "m", 3.0, 15.0)
    assert estimate_cost(Usage(10, 10, 5), p) is None


def test_retries_add_to_total():
    one = estimate_cost(Usage(1000, 100, 0), PRICE)
    assert total_cost([Usage(1000, 100, 0)] * 3, PRICE) == pytest.approx(3 * one)


def test_wire_format_merges_tool_results_into_one_user_turn():
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "text": "ok", "tool_calls": [
            {"id": "a", "name": "get_order", "arguments": {"order_id": "o"}},
            {"id": "b", "name": "get_return_status", "arguments": {"order_id": "o"}}]},
        {"role": "tool", "tool_call_id": "a", "content": "{}"},
        {"role": "tool", "tool_call_id": "b", "content": "{}"},
    ]
    wire = to_wire_messages(msgs)
    assert [w["role"] for w in wire] == ["user", "assistant", "user"]
    assert [b["type"] for b in wire[1]["content"]] == ["text", "tool_use", "tool_use"]
    assert [b["tool_use_id"] for b in wire[2]["content"]] == ["a", "b"]


def test_parse_response_normalizes_usage():
    r = parse_response({"id": "x", "model": "resolved-1", "stop_reason": "tool_use",
                        "content": [{"type": "text", "text": "hi"},
                                    {"type": "tool_use", "id": "t1", "name": "get_order",
                                     "input": {"order_id": "o"}}],
                        "usage": {"input_tokens": 50, "output_tokens": 9,
                                  "cache_read_input_tokens": 400}})
    assert r.text == "hi" and r.tool_calls[0].arguments == {"order_id": "o"}
    assert (r.usage.input_tokens, r.usage.output_tokens, r.usage.cached_input_tokens) == (50, 9, 400)
    assert r.resolved_model == "resolved-1"


async def test_provider_http_error_becomes_provider_error():
    transport = httpx.MockTransport(lambda req: httpx.Response(529, text="overloaded"))
    prov = AnthropicProvider("k", httpx.AsyncClient(transport=transport))
    with pytest.raises(ProviderError, match="529"):
        await prov.generate(model="m", system="s", messages=[{"role": "user", "content": "x"}],
                            tools=[], timeout=5, max_output_tokens=10)


async def test_request_shape_and_secret_stays_in_header():
    seen = {}
    def handler(req: httpx.Request):
        seen["headers"], seen["body"] = req.headers, req.read().decode()
        return httpx.Response(200, json={"content": [{"type": "text", "text": "ok"}],
                                         "usage": {"input_tokens": 1, "output_tokens": 1}})
    prov = AnthropicProvider("sekret", httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    out = await prov.generate(model="m", system="s", messages=[{"role": "user", "content": "x"}],
                              tools=[{"name": "t", "description": "d", "parameters": {"type": "object"}}],
                              timeout=5, max_output_tokens=10)
    assert out.text == "ok" and seen["headers"]["x-api-key"] == "sekret"
    assert "sekret" not in seen["body"] and '"input_schema"' in seen["body"]
