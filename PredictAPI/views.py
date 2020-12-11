from django.shortcuts import render
from django.http import JsonResponse

import json
import pandas as pd
import joblib

heartModel=joblib.load('heart_model.pkl')

def predictHeart(request):
    data = json.loads(request.body)
    dataF = pd.DataFrame({'x':data}).transpose()
    score = heartModel.predict(dataF)[0]
    print(score)
    score = float(score)
    return JsonResponse({'score':score})


def predictLiver(request):
    return JsonResponse({'score': 1})


def predictKidney(request):
    return JsonResponse({'score': 1})


# 63	1	3	145	233	1	0	150	0	2.3	0	0	1	1
