from django import forms
from .models import FileData

class FileDataForm(forms.ModelForm):
    class Meta:
        model=FileData
        fields=("name", "message", "image")
    