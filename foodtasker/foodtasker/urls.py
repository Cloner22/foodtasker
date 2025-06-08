"""
URL configuration for foodtasker project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from coreapp import views , apis
from django.contrib.auth import views as auth_views
from django.conf import settings
from django.conf.urls.static import static




urlpatterns = [
    path('admin/', admin.site.urls),
    path('',views.home , name = 'home' ),
    path('restaurant/sign_in/', auth_views.LoginView.as_view(template_name='restaurant/sign_in.html'), name='restaurant_sign_in'),
    path('restaurant/sign_out/', auth_views.LogoutView.as_view(next_page='/'), name='restaurant_sign_out'),
    path('restaurant/sign_up/',views.restaurant_sign_up , name = 'restaurant_sign_up' ),
    path('restaurant/',views.restaurant_home , name = 'restaurant_home' ),
    
    path('restaurant/account/',views.restaurant_account, name = 'restaurant_account' ),
    path('restaurant/meal/',views.restaurant_meal , name = 'restaurant_meal' ),
    path('restaurant/meal/add/',views.restaurant_add_meal , name = 'restaurant_add_meal' ),
    path('restaurant/meal/edit/<int:meal_id>',views.restaurant_edit_meal , name = 'restaurant_edit_meal' ),

    path('restaurant/order/',views.restaurant_order, name = 'restaurant_order' ),
    path('restaurant/report/',views.restaurant_report , name = 'restaurant_report' ),
    
    # APIS
    path('api/restaurant/order/notification/<last_request_time>/', apis.restaurant_order_notification),

    # APIS for CUSTOMERS
    path('api/customer/restaurants/', apis.custmoer_get_restaurant),
    path('api/customer/meals/<int:restaurant_id>', apis.custmoer_get_meals),
    path('api/customer/order/add/', apis.custmoer_add_order),
    path('api/customer/order/latest/', apis.custmoer_get_latest_order),
    path('api/customer/order/latest_status/', apis.custmoer_get_latest_order_status),
    path('api/customer/driver/location/', apis.custmoer_get_driver_location),

    # APIS for DRIVRER
    path('api/driver/order/ready/', apis.driver_get_ready_orders),
    path('api/driver/order/pick/', apis.driver_pick_order),
    path('api/driver/order/latest/', apis.driver_get_latest_order),
    path('api/driver/order/complete/', apis.driver_complete_order),
    path('api/driver/revenue/', apis.driver_get_revenue),
    path('api/driver/location/update/', apis.driver_update_location),
    path('api/driver/profile/', apis.driver_get_profile),
    path('api/driver/profile/update/', apis.driver_update_profile),




]

#locations media
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

