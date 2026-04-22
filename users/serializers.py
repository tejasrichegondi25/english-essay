from rest_framework import serializers
from .models import UserRegistrationModel

class UserRegistrationSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserRegistrationModel
        fields = ['name', 'loginid', 'password', 'mobile', 'email', 'locality', 'address', 'city', 'state']
        extra_kwargs = {'password': {'write_only': True}}

    def create(self, validated_data):
        validated_data['status'] = 'pending'
        return UserRegistrationModel.objects.create(**validated_data)

class UserLoginSerializer(serializers.Serializer):
    loginid = serializers.CharField()
    pswd = serializers.CharField()
