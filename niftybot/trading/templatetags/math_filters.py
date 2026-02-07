from django import template

register = template.Library()

@register.filter
def div(value, arg):
    try:
        return float(value) / float(arg) if arg != 0 else 0
    except (ValueError, ZeroDivisionError):
        return 0

@register.filter
def multiply(value, arg):
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return 0