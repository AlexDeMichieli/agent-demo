"""
Shopping cart module.

This is a small intentional codebase with some bugs for the agent to find and fix.
The agent will read issues, analyze this code, and generate fixes.
"""


def calculate_total(items):
    """Calculate the total price of items in a cart.
    
    Args:
        items: list of dicts with 'price' and 'quantity' keys
        
    Returns:
        float: total price
    """
    total = 0
    for item in items:
        total += item["price"] * item["quantity"]
    return total


def apply_discount(total, discount_code):
    """Apply a discount code to the total.
    
    Known bug: crashes with TypeError when discount_code is None
    instead of returning the original total.
    """
    discounts = {
        "SAVE10": 0.10,
        "SAVE20": 0.20,
        "VIP50": 0.50,
    }
    discount = discounts.get(discount_code, 0)
    return total - (total * discount)


def format_receipt(items, discount_code=None):
    """Format a receipt string for the given items.
    
    Known bugs:
    - Crashes on empty items list (no meaningful output)
    - Calls apply_discount even when discount_code is None (triggers the bug above)
    """
    total = calculate_total(items)

    # BUG: when discount_code is explicitly passed as None,
    # this still tries to call apply_discount which crashes
    final = apply_discount(total, discount_code)

    lines = []
    lines.append("=== RECEIPT ===")
    for item in items:
        lines.append(f"  {item['name']}: ${item['price']} x {item['quantity']}")
    lines.append(f"  Subtotal: ${total:.2f}")
    if discount_code:
        lines.append(f"  Discount ({discount_code}): -${total - final:.2f}")
    lines.append(f"  TOTAL: ${final:.2f}")
    lines.append("===============")
    return "\n".join(lines)
