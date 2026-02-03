# from django.contrib import admin
# from django.urls import path, include
# from django.contrib.auth import views as auth_views
# from trading import views

# urlpatterns = [
#     path('admin/', admin.site.urls),
#     path('', views.home, name='home'),
#     path('dashboard/', views.dashboard, name='dashboard'),
#     path('broker/', views.broker_page, name='broker'),
#     path('pnl-calendar/', views.pnl_calendar, name='pnl_calendar'),
#     path('algorithms/', views.algorithms_page, name='algorithms'),
#     path('login/', auth_views.LoginView.as_view(template_name='registration/login.html'), name='login'),
#     path('logout/', auth_views.LogoutView.as_view(), name='logout'),
#     path('signup/', views.signup, name='signup'),
#     path('connect-zerodha/', views.connect_zerodha, name='connect_zerodha'),
#     path('start-bot/', views.start_bot, name='start_bot'),
#     path('stop-bot/', views.stop_bot, name='stop_bot'),
#     path('bot-status/', views.bot_status, name='bot_status'),
# ]
# niftybot/urls.py
from django.contrib import admin
from django.urls import path
from django.contrib.auth import views as auth_views
from trading import views

urlpatterns = [
    path('admin/', admin.site.urls),

    # Public pages (no login required)
    path('', views.home, name='home'),
    path('signup/', views.signup, name='signup'),
    path('login/', auth_views.LoginView.as_view(template_name='registration/login.html'), name='login'),
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),

    # Protected pages (login required)
    path('dashboard/', views.dashboard, name='dashboard'),
    path('broker/', views.broker_page, name='broker'),
    path('pnl-calendar/', views.pnl_calendar, name='pnl_calendar'),
    path('algorithms/', views.algorithms_page, name='algorithms'),
    path('connect-zerodha/', views.connect_zerodha, name='connect_zerodha'),
    path('start-bot/', views.start_bot, name='start_bot'),
    path('stop-bot/', views.stop_bot, name='stop_bot'),
    path('bot-status/', views.bot_status, name='bot_status'),
    path('update-max-lots/', views.update_max_lots, name='update_max_lots'),
]