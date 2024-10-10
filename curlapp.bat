curl -X POST http://127.0.0.1:8000/predict_diabetes/ \
-H "Content-Type: application/json" \
-d '{
    "feature1": 0.038075906433423,
    "feature2": 0.050680118739818,
    "feature3": 0.0616962065186875,
    "feature4": 0.0218723549949558,
    "feature5": -0.044223498424446,
    "feature6": -0.034820762837698,
    "feature7": -0.043400846128719,
    "feature8": -0.00259226199818282,
    "feature9": 0.0199074893129092,
    "feature10": -0.017646125159806
}'
