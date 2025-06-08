import json
from django.http import JsonResponse
from coreapp.models import Restaurant ,Meal , Order , OrderDetails
from coreapp.serializers import RestaurantSerializer ,\
MealSerializer ,OrdersSerializer,OrderStatusSerializer,OrderDriverSerializer

from django.utils import timezone
from oauth2_provider.models import AccessToken
from django.views.decorators.csrf import csrf_exempt
# ==========
# RESTAURANT
# ==========

def restaurant_order_notification (request,last_request_time):
    notification = Order.objects.filter(
        restaurant  = request.user.restaurant,
        created_at__gt = last_request_time
    ).count()

    return JsonResponse({"notification": notification})

# ==========
# CUSTOMER
# ==========

def custmoer_get_restaurant (request):
    restaurants = RestaurantSerializer(
        Restaurant.objects.all().order_by("-id"),
        many=True,
        context = {"request": request}
    ).data
    return JsonResponse({"restaurant":restaurants})


def custmoer_get_meals(request,restaurant_id):
    meals = MealSerializer(
        Meal.objects.filter(restaurant_id=restaurant_id).order_by("-id"),
        many = True,
        context ={"request": request}
    ).data
    return JsonResponse({"meals":meals})
@csrf_exempt
def custmoer_add_order(request):

    """
    params:
        1.access_token
        2. restaurant_id
        3. address
        4. order_details (json format), example:
            [{"meal_id": 1, "quantity": 2},{"meal_id": 2, "quantity": 3}}]
    return:  
         {"status": "success"}  
    
    """

    if request.method == "POST":
        # Get access token
        access_token = AccessToken.objects.get(
            token=request.POST.get("access_token"),
            expires__gt=timezone.now()

            )
        # Get customer profile
        customer = access_token.user.customer
        # Check whether customer has any outstanding order
        if Order.objects.filter(customer=customer).exclude(status = Order.DELIVERED):
            return JsonResponse({"status": "filed", "error": "Your last order must be complated"})
        # check orders address
        if not request.POST["address"]:
            return JsonResponse({"status": "failed", "error": "Address is required"})

        # get order details 
        order_details = json.loads(request.POST["order_details"])
        
        # check if meals in only one restaurant and then calculate the order total
        order_total= 0
        for meal in order_details:
            if not Meal.objects.filter(id=meal["meal_id"], restaurant_id = request.POST["restaurant_id"]):
                return JsonResponse({"status": "failed","error":"meals must be in only one restaurant "})
            else:
                order_total += Meal.objects.get(id=meal["meal_id"]).price * meal["quantity"]
        # create order 
        if len(order_details)> 0 :
            
            #  step 1 - create an order
            order = Order.objects.create(
                customer = customer,
                restaurant_id = request.POST["restaurant_id"],
                total = order_total,
                status = Order.COOKING,
                address = request.POST["address"]
            )

            #  step 2- create order details
            for meal in order_details:
                OrderDetails.objects.create(
                   order  =order,
                    meal_id = meal["meal_id"],
                    quantity = meal["quantity"],
                    sub_total = Meal.objects.get(id = meal["meal_id"]).price*meal["quantity"]
                )
            return JsonResponse({"status":"success"})
                    


    return JsonResponse({})
    
def custmoer_get_latest_order(request):

    """
    
    params:
    1 . access_token
    return:
    {json data with all details of an order}
    """
    access_token = AccessToken.objects.get(
        token = request.GET.get("access_token"),
        expires__gt = timezone.now()
    )
    customer = access_token.user.customer

    order = OrdersSerializer(
        Order.objects.filter(customer = customer).last()
    ).data
    return JsonResponse({
        "last_order":order
    })

def custmoer_get_latest_order_status(request):

    """
        params:
             1 . access_token
        return:
            {json data with all details of an order}
    """
    access_token = AccessToken.objects.get(
        token = request.GET.get("access_token"),
        expires__gt = timezone.now()
    )
    customer = access_token.user.customer

    order_status = OrderStatusSerializer(
        Order.objects.filter(customer = customer).last()
    ).data

    return JsonResponse({
        "last_order_status":order_status
    })

def custmoer_get_driver_location(request):
    access_token = AccessToken.objects.get(
        token = request.GET.get("access_token"),
        expires__gt = timezone.now()
    )

    customer = access_token.user.customer 
    
    current_order = Order.objects.filter(customer = customer,status = Order.ONTHEWAY).last()
    if current_order :
        location = current_order.driver.location
    else:
        location = None


    return JsonResponse({
        "location" : location,
    })


# ==========
# DRIVER
# ==========

def driver_get_ready_orders(request):
    orders = OrdersSerializer(
        Order.objects.filter(status = Order.READY,  driver = None).order_by ('-id'),
        many = True
    ).data
    return JsonResponse({
        "orders" : orders
    })

@csrf_exempt
def driver_pick_order(request):
    
    """
        params:
            1. access_token
            2. order_id
        return:
            {"status" : "success"}
    """
    if request.method == "POST" :
        # Get access token
        access_token = AccessToken.objects.get(
            token = request.POST.get("access_token"),
            expires__gt = timezone.now()
        )
        
        # Get Driver
        driver = access_token.user.driver
    
        # check if this driver
        if Order.objects.filter(driver=driver, status=Order.ONTHEWAY):
            return JsonResponse({
                "status": "failed",
                "error" : "Your outstanding order is not delivered yet."
            })
        # process the picking up order
        try:
            order = Order.objects.get(
            id = request.POST["order_id"],
            driver = None,
            status = Order.READY
            )

            order.driver = driver
            order.status = Order.ONTHEWAY
            order.picked_at = timezone.now()
            order.save()
            
            return JsonResponse({
                "status" : "success"
            }) 
        
        except Order.DoesNotExist:
            return JsonResponse({
                "status" : "failed",
                "error": "this order has been picked up by another"
            })

def driver_get_latest_order(request):
    # Get access_token
    access_token = AccessToken.objects.get(
        token = request.GET['access_token'],
        expires__gt = timezone.now()
    )
    # Get Driver
    driver = access_token.user.driver

    # Get the latest order of this driver
    order = OrdersSerializer(
        Order.objects.filter(driver=driver, status = Order.ONTHEWAY).last()
    ).data
    return JsonResponse({
        "ordre" : order
    })
@csrf_exempt
def driver_complete_order(request):
    """ 
    params:
        1. access_token
        2. order_idd
    return:
    {"status": "success"}
    
    """
    if request.method == "POST":
         # Get access token
        access_token = AccessToken.objects.get(
            token = request.POST.get("access_token"),
            expires__gt = timezone.now()
        )
        
        # Get Driver
        driver = access_token.user.driver

        # complete an order
        order = Order.objects.get(id = request.POST["order_id"],driver = driver)
        order.status = Order.DELIVERED
        order.save()

    return JsonResponse({
        "status":"success"
    })

def driver_get_revenue(request):
    # Get access token
    access_token = AccessToken.objects.get(
        token = request.GET.get("access_token"),
        expires__gt = timezone.now()
    )
    
    # Get Driver
    driver = access_token.user.driver

    from datetime import timedelta

    revenue = {}
    today = timezone.now()
    current_weekdays = [today + timedelta(days=i) for i in range (0 - today.weekday(), 7 -today.weekday())]

    for day in current_weekdays:
        orders = Order.objects.filter(
            driver = driver,
            status = Order.DELIVERED,
            created_at__year = day.year,
            created_at__month = day.month,
            created_at__day = day.day,
        )

        revenue[day.strftime("%a")] = sum(order.total for order in orders)

    return JsonResponse({
        "revenue" : revenue
    })


@csrf_exempt
def driver_update_location(request):

    """
    params:
      1.access_token
      2.location EX:  lat, lng
    return:
    {"status":"success"}

        """
    if request.method == "POST":
        access_token = AccessToken.objects.get(
            token = request.POST["access_token"],
            expires__gt = timezone.now()
        )

        driver = access_token.user.driver
        driver.location = request.POST.get("location")
        driver.save()

    return JsonResponse({
        "status":"success"
    })

def driver_get_profile(request):
    access_token = AccessToken.objects.get(
        token = request.GET["access_token"],
        expires__gt = timezone.now()
    )

    driver = OrderDriverSerializer(
        access_token.user.driver
    ).data
    
    return JsonResponse({
        "driver": driver
    })

@csrf_exempt
def driver_update_profile(request):
    """
     params:
        1.access_token
        2.car_model
        3.palte_number
    return:
    {"status" : "success"}
    
    """

    if request.method == "POST":
        access_token = AccessToken.objects.get(
            token = request.POST["access_token"],
            expires__gt = timezone.now()
        )

        driver = access_token.user.driver

        #update drivers profile 
        driver.car_models  = request.POST["car_models"]
        driver.plate_number = request.POST["plate_number"]
        driver.save()
        
    return JsonResponse({
        "status" : "success"
    })