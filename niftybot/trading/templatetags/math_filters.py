# trading/templatetags/math_filters.py
from django import template
from decimal import Decimal, InvalidOperation, DivisionByZero

register = template.Library()


@register.filter(name='div')
def divide(value, arg):
    """
    Safely divides value by arg.
    Returns 0 if division by zero or invalid types.
    
    Usage: {{ value|div:arg }}
    Example: {{ total_pnl|div:total_trades }}
    """
    try:
        # Convert to Decimal for financial precision
        val = Decimal(str(value)) if value is not None else Decimal('0')
        divisor = Decimal(str(arg)) if arg is not None else Decimal('0')
        
        if divisor == 0:
            return Decimal('0')
            
        return val / divisor
    
    except (InvalidOperation, DivisionByZero, TypeError, ValueError):
        return Decimal('0')


@register.filter(name='multiply')
def multiply(value, arg):
    """
    Multiplies value by arg safely.
    Returns 0 on invalid input.
    
    Usage: {{ win_rate|multiply:100 }}
    """
    try:
        val = Decimal(str(value)) if value is not None else Decimal('0')
        factor = Decimal(str(arg)) if arg is not None else Decimal('0')
        return val * factor
    except (InvalidOperation, TypeError, ValueError):
        return Decimal('0')


@register.filter(name='sub')
def subtract(value, arg):
    """
    Subtracts arg from value safely.
    
    Usage: {{ total|sub:fees }}
    """
    try:
        val = Decimal(str(value)) if value is not None else Decimal('0')
        subtrahend = Decimal(str(arg)) if arg is not None else Decimal('0')
        return val - subtrahend
    except (InvalidOperation, TypeError, ValueError):
        return Decimal('0')


@register.filter(name='abs')
def absolute(value):
    """
    Returns the absolute value.
    
    Usage: {{ pnl|abs }}
    """
    try:
        val = Decimal(str(value)) if value is not None else Decimal('0')
        return abs(val)
    except (InvalidOperation, TypeError, ValueError):
        return Decimal('0')


@register.filter(name='percentage')
def percentage(value, total):
    """
    Calculates percentage: (value / total) * 100
    Returns 0 if total is 0 or invalid.
    
    Usage: {{ wins|percentage:total_trades }}
    """
    try:
        val = Decimal(str(value)) if value is not None else Decimal('0')
        tot = Decimal(str(total)) if total is not None else Decimal('0')
        
        if tot == 0:
            return Decimal('0')
            
        return (val / tot) * Decimal('100')
    except (InvalidOperation, DivisionByZero, TypeError, ValueError):
        return Decimal('0')