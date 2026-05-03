import importlib, gemini_client
importlib.reload(gemini_client)
import ml_engine as ml

answers = {"q1":9,"q2":"2","q3":"0","q4":"3","q5":"2","q6":"0","q7":10,"q8":"1","q9":"2","q10":"0","q11":"1","q12":9}
pred = ml.predict(answers)

summary = gemini_client.generate_summary(pred["cluster"], pred["match_pct"], pred["subjects"], answers)
print("Gemini summary:")
print(summary)
