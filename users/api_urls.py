from django.urls import path
from .api_views import RegisterAPIView, LoginAPIView, PredictionAPIView, DatasetAPIView, AdminLoginAPIView, AdminUsersListAPIView, UserActivateAPIView, TrainingAPIView

urlpatterns = [
    path('register/', RegisterAPIView.as_view(), name='api_register'),
    path('login/', LoginAPIView.as_view(), name='api_login'),
    path('predict/', PredictionAPIView.as_view(), name='api_predict'),
    path('dataset/', DatasetAPIView.as_view(), name='api_dataset'),
    path('admin/login/', AdminLoginAPIView.as_view(), name='api_admin_login'),
    path('admin/users/', AdminUsersListAPIView.as_view(), name='api_admin_users'),
    path('admin/activate/', UserActivateAPIView.as_view(), name='api_admin_activate'),
    path('training/', TrainingAPIView.as_view(), name='api_training'),
    path('health/', lambda r: Response({"status": "ok"}), name='api_health'),
]
