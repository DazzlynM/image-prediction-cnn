from django.db import models

class FileData(models.Model):
    name = models.CharField(max_length=50)
    message = models.CharField(max_length=100)
    image = models.ImageField(null=True, blank=True, upload_to='img/%y')