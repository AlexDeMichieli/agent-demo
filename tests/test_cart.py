"""Tests for the cart module. Some of these intentionally fail due to bugs."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.cart import calculate_total, apply_discount, format_receipt


def test_calculate_total():
    items = [{"price": 10, "quantity": 2}, {"price": 5, "quantity": 1}]
    assert calculate_total(items) == 25


def test_calculate_total_empty():
    assert calculate_total([]) == 0


def test_apply_discount_valid():
    assert apply_discount(100, "SAVE10") == 90
    assert apply_discount(100, "SAVE20") == 80


def test_apply_discount_invalid_code():
    # Invalid code should return original total (no discount)
    assert apply_discount(100, "BOGUS") == 100


def test_apply_discount_none():
    # BUG: this crashes with TypeError instead of returning 100
    assert apply_discount(100, None) == 100


def test_format_receipt_basic():
    items = [
        {"name": "Widget", "price": 10, "quantity": 2},
        {"name": "Gadget", "price": 5, "quantity": 1},
    ]
    receipt = format_receipt(items)
    assert "TOTAL: $25.00" in receipt


def test_format_receipt_with_discount():
    items = [{"name": "Widget", "price": 100, "quantity": 1}]
    receipt = format_receipt(items, "SAVE10")
    assert "TOTAL: $90.00" in receipt


def test_format_receipt_no_discount_explicit_none():
    # BUG: this crashes because format_receipt passes None to apply_discount
    items = [{"name": "Widget", "price": 10, "quantity": 1}]
    receipt = format_receipt(items, None)
    assert "TOTAL: $10.00" in receipt
