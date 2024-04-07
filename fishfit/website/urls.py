from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name="home"),
    path('about.html', views.about, name="about"),
    path('index.html', views.home, name="home"),
    path('blog.html', views.blog, name="blog"),
    path('contact.html', views.contact, name="contact"),
    path('watch.html', views.watch, name="watch"),
    path('track', views.track, name="func"),
    path('fishs.html',views.fishs, name="fishs"),
    path('testimonial.html',views.testi, name="testimonial"),
    path('login.html',views.handlelogin, name="handlelogin"),
    path('signup.html', views.handlesignup, name="handlesignup"),
    path('logout',views.handlelogout, name="handlelogout")

]
