from django.shortcuts import render
from users.forms import UserRegistrationForm


def index(request):
    try:
        return render(request, 'index.html', {})
    except Exception as e:
        import traceback
        traceback.print_exc()
        from django.http import HttpResponse
        return HttpResponse(f"Error: {str(e)}", status=500)

def AdminLogin(request):
    return render(request, 'AdminLogin.html', {})

def UserLogin(request):
    return render(request, 'UserLogin.html', {})


def UserRegister(request):
    form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})