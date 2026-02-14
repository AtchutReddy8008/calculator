# trading/templatetags/calendar_tags.py

from django import template
from calendar import monthcalendar

register = template.Library()


@register.simple_tag(name='calendar')
def get_calendar(year, month):
    """
    Custom template tag that returns a calendar grid for a given year and month.
    
    Returns:
        list of lists (weeks), each containing 7 integers (day numbers or 0)
    
    Usage:
        {% calendar year month as cal %}
        {% for week in cal %}
            {% for day in week %}
                {{ day }}
            {% endfor %}
        {% endfor %}
    """
    try:
        y = int(year)
        m = int(month)
        if not (1900 <= y <= 2100 and 1 <= m <= 12):
            raise ValueError("Invalid year or month")
        return monthcalendar(y, m)
    except (ValueError, TypeError):
        # Return empty grid on error (prevents template crash)
        return [[0] * 7 for _ in range(6)]