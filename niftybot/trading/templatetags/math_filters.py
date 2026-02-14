# trading/templatetags/math_filters.py

from django import template
from decimal import Decimal, ROUND_HALF_UP
from django.utils.safestring import mark_safe

register = template.Library()


@register.filter(name='get_item')
def get_item(dictionary, key):
    """
    Safely get item from dictionary in template:
    {{ my_dict|get_item:'some_key' }}
    """
    if dictionary is None:
        return None
    return dictionary.get(key)


@register.filter(name='greater_than')
def greater_than(value, arg):
    """Returns True if value > arg"""
    try:
        return Decimal(str(value)) > Decimal(str(arg))
    except (ValueError, TypeError):
        return False


@register.filter(name='less_than')
def less_than(value, arg):
    """Returns True if value < arg"""
    try:
        return Decimal(str(value)) < Decimal(str(arg))
    except (ValueError, TypeError):
        return False


@register.filter(name='round_decimal')
def round_decimal(value, places=2):
    """
    Rounds a number (int/float/Decimal) to specified decimal places
    Usage: {{ value|round_decimal:0 }}  → integer
           {{ value|round_decimal:2 }} → two decimals
    """
    if value is None:
        return "0.00" if places > 0 else "0"

    try:
        decimal_value = Decimal(str(value))
        rounded = decimal_value.quantize(
            Decimal('1.' + '0' * places),
            rounding=ROUND_HALF_UP
        )
        if places == 0:
            return str(int(rounded))
        return f"{rounded:.{places}f}"
    except (ValueError, TypeError):
        return "0" if places == 0 else "0.00"


@register.filter(name='percentage')
def percentage(part, whole):
    """
    Calculate percentage: part / whole * 100
    Returns float rounded to 1 decimal place
    Usage: {{ wins|percentage:total }} → e.g. 42.5
    """
    if whole == 0 or whole is None:
        return 0.0
    try:
        return round((part / whole) * 100, 1)
    except (TypeError, ZeroDivisionError):
        return 0.0


@register.filter(name='div')
def div(value, divisor):
    """Simple division filter"""
    if divisor == 0 or divisor is None:
        return 0
    try:
        return Decimal(str(value)) / Decimal(str(divisor))
    except:
        return 0


@register.filter(name='mul')
def mul(value, multiplier):
    """Simple multiplication filter"""
    try:
        return Decimal(str(value)) * Decimal(str(multiplier))
    except:
        return 0


@register.filter(name='intcomma')
def intcomma(value):
    """
    Alias to django.contrib.humanize intcomma (for consistency)
    Already available via {% load humanize %}, but added here as fallback
    """
    from django.contrib.humanize.templatetags.humanize import intcomma as humanize_intcomma
    return humanize_intcomma(value)