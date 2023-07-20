from django.shortcuts import render
from django.http import HttpResponse
import joblib
# Create your views here.
def home(request):
  return render(request, 'home.html')

def result(request):
  cls= joblib.load('C:\\Users\\MINISTER JOHN\\Downloads\\pickled_model.pkl')
  lis = []
  lis.append(request.GET['year'])
  lis.append(request.GET['kilometers'])
  lis.append(request.GET['engine'])
  lis.append(request.GET['power'])
  lis.append(request.GET['seats'])
  lis.append(request.GET['transmission'])
  lis.append(request.GET['owner_type'])

  ans = cls.predict([lis])

  return render (request, 'result.html', {'ans': ans})