"""
ASGI config for Automatic_English_Essay_Scoring_Algorithm_Based_On_Ml project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.1/howto/deployment/asgi/
"""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Automatic_English_Essay_Scoring_Algorithm_Based_On_Ml.settings')

application = get_asgi_application()
