from django import template

register = template.Library()

@register.filter(name='filter_date')
def filter_date(records, target_date_str):
    """
    Returns the first DailyPnL record matching the given YYYY-MM-DD string
    """
    for record in records:
        if record.date.strftime('%Y-%m-%d') == target_date_str:
            return record
    return None